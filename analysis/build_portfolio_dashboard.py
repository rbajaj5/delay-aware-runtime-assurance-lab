"""Build a local Tableau package and review preview from a frozen simulation.

No simulation is run and nothing is published by this script.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import itertools
import json
import shutil
import sys
import uuid
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"/"tableau_deps"))
from lxml import etree as ET
from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode, TableDefinition, TableName, SqlType, Inserter
from tableaudocumentapi import Workbook

OUT=ROOT/"artifacts"/"20261013"/"portfolio_rta"
DASH=OUT/"tableau"
NS="http://www.tableausoftware.com/xml/user"
DS="riskdemo"


def node(parent,tag,text=None,**attrs):
    e=ET.SubElement(parent,tag,{k.replace("_","-"):str(v) for k,v in attrs.items()})
    if text is not None:e.text=str(text)
    return e


def profile_csv(path):
    with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    info={}
    for key in rows[0]:
        vals=[r[key] for r in rows]; present=[x for x in vals if x!=""]
        if set(present)<= {"True","False"}:kind="boolean"
        else:
            try:
                numbers=[float(x) for x in present]
                kind="integer" if all(x.is_integer() for x in numbers) else "real"
            except ValueError:kind="string"
        info[key]={"type":kind,"missing":len(vals)-len(present),"unique":len(set(present))}
        if kind=="string":info[key]["values"]=sorted(set(present))
    return {"file":path.name,"rows":len(rows),"columns":len(info),"fields":info}


def flat_rows(records):
    rows=[]
    for r in records:
        for j,action in enumerate(["High pick-and-roll","Five-out drive","Weak-side cut","Post split","Advance","Reset"]):
            for kind,key in [("Proposed","weights_proposed"),("Applied","weights_applied")]:
                rows.append({"Frame":r["frame"],"Time (s)":r["time_s"],"Phase":r["phase"],
                    "Coverage":r["coverage"],"Controller":r["controller_decision"],
                    "Action":action,"Allocation":kind,"Weight":r[key][j],
                    "Leading share":r["leading_share"],"Eigenvector overlap":r["eigenvector_overlap"],
                    "Lambda 1":r["eigenvalues"][0],"Lambda 2":r["eigenvalues"][1],"Lambda 3":r["eigenvalues"][2],
                    "Covariance condition":r["covariance_condition_number"],
                    "Conditioning":r["conditioning_proposed"] if kind=="Proposed" else r["conditioning_applied"],
                    "Shadow conditioning":r["conditioning_shadow"],"Shadow RMSE (ft)":r["shadow_tracking_rmse_ft"],
                    "Shadow spacing (ft)":r["spacing_shadow_ft"],"Hidden collapse":r["hidden_collapse"],
                    "Applied RMSE (ft)":r["tracking_rmse_ft"],"Valid observation fraction":r["observation_valid_fraction"]})
    return rows


def hyper_and_checks(rows,records):
    fields=list(rows[0]); first=rows[0]
    kinds={k:('boolean' if isinstance(first[k],bool) else 'integer' if isinstance(first[k],int) else 'real' if isinstance(first[k],float) else 'string') for k in fields}
    types={"boolean":SqlType.bool(),"integer":SqlType.big_int(),"real":SqlType.double(),"string":SqlType.text()}
    target=DASH/"Data"/"risk.hyper"
    target.parent.mkdir(parents=True,exist_ok=True)
    table=TableDefinition(TableName("Extract","Extract"),[TableDefinition.Column(k,types[kinds[k]]) for k in fields])
    validations=[]
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as proc:
        with Connection(proc.endpoint,str(target),CreateMode.CREATE_AND_REPLACE) as conn:
            conn.catalog.create_schema("Extract"); conn.catalog.create_table(table)
            with Inserter(conn,table) as ins:
                ins.add_rows([[row[k] for k in fields] for row in rows]); ins.execute()
            phases=["All"]+list(dict.fromkeys(r["phase"] for r in records))
            modes=["All","ACCEPT","RTA HOLD"]
            for phase,mode in itertools.product(phases,modes):
                chosen=[r for r in records if (phase=="All" or r["phase"]==phase) and (mode=="All" or r["controller_decision"]==mode)]
                clauses=[]
                for key,value in [("Phase",phase),("Controller",mode)]:
                    if value!="All":clauses.append('"'+key+'" = '+"'"+value.replace("'","''")+"'")
                where=" WHERE "+" AND ".join(clauses) if clauses else ""
                query='SELECT COUNT(DISTINCT "Frame"), AVG("Leading share"), COUNT(DISTINCT CASE WHEN "Hidden collapse" THEN "Frame" ELSE NULL END) FROM "Extract"."Extract"'+where
                n,avg,hidden=conn.execute_list_query(query)[0]
                expected_avg=float(np.mean([r["leading_share"] for r in chosen])) if chosen else None
                passed=n==len(chosen) and hidden==sum(r["hidden_collapse"] for r in chosen)
                passed=passed and ((avg is None and expected_avg is None) or (avg is not None and abs(avg-expected_avg)<1e-10))
                grouped=conn.execute_list_query('SELECT "Action", "Allocation", AVG("Weight") FROM "Extract"."Extract"'+where+' GROUP BY "Action", "Allocation"')
                for action,kind,weight in grouped:
                    j=list(dict.fromkeys(row["Action"] for row in rows)).index(action)
                    key="weights_proposed" if kind=="Proposed" else "weights_applied"
                    expected=float(np.mean([r[key][j] for r in chosen]))
                    passed=passed and abs(weight-expected)<1e-10
                validations.append({"phase":phase,"controller":mode,"frames":n,"mean_leading_share":avg,
                                    "hidden_frames":hidden,"small_group":0<n<120,"passed":bool(passed)})
            count=conn.execute_scalar_query('SELECT COUNT(*) FROM "Extract"."Extract"')
    return kinds,{"engine":"Tableau Hyper SQL", "rows":count,"distinct_frames":len(records),
                  "filter_combinations":validations,"all_numeric_checks_passed":all(x["passed"] for x in validations),
                  "native_desktop_render_tested":False,"native_desktop_filters_tested":False,
                  "note":"Hyper query results were checked against frame JSON. Native Tableau Desktop is not installed; workbook layout and native filter controls still require Desktop acceptance."}


def parameters(parent,name,values):
    col=node(parent,"column",caption=name,datatype="string",name=f"[{name}]",param_domain_type="list",role="measure",type="nominal",value='"All"')
    node(col,"calculation",**{"class":"tableau","formula":'"All"'})
    members=node(col,"members")
    for value in values:node(members,"member",value='"'+value+'"')
    return col


def make_workbook(kinds,records):
    root=ET.Element("workbook",{"original-version":"18.1","source-platform":"win","version":"18.1"},nsmap={"user":NS})
    node(root,"preferences")
    datasources=node(root,"datasources")
    params=node(datasources,"datasource",name="Parameters",hasconnection="false",inline="true",version="18.1")
    p1=parameters(params,"Phase filter",["All"]+list(dict.fromkeys(r["phase"] for r in records)))
    p2=parameters(params,"Controller filter",["All","ACCEPT","RTA HOLD"])
    source=node(datasources,"datasource",caption="Synthetic agent simulation",inline="true",name=DS,version="18.1")
    conn=node(source,"connection",**{"class":"hyper","dbname":"Data/risk.hyper","schema":"Extract","tablename":"Extract","access_mode":"readonly","authentication":"auth-none"})
    node(conn,"relation",name="Extract",table="[Extract].[Extract]",type="table")
    meta=node(conn,"metadata-records")
    columns={}
    for i,(field,kind) in enumerate(kinds.items()):
        role="dimension" if kind in ["string","boolean"] or field=="Frame" else "measure"
        typ="nominal" if role=="dimension" else "quantitative"
        rec=node(meta,"metadata-record",**{"class":"column"})
        for tag,value in [("remote-name",field),("remote-type",130 if kind=="string" else 11 if kind=="boolean" else 20 if kind=="integer" else 5),
                          ("local-name",f"[{field}]"),("parent-name","[Extract]"),("remote-alias",field),("ordinal",i),
                          ("local-type",kind),("aggregation","Count" if role=="dimension" else "Avg"),("contains-null","false")]:node(rec,tag,value)
        columns[field]=node(source,"column",name=f"[{field}]",caption=field,datatype=kind,role=role,type=typ)
    formula='([Parameters].[Phase filter] = "All" OR [Phase] = [Parameters].[Phase filter]) AND ([Parameters].[Controller filter] = "All" OR [Controller] = [Parameters].[Controller filter])'
    keep=node(source,"column",name="[Keep frame]",datatype="boolean",role="dimension",type="nominal")
    node(keep,"calculation",**{"class":"tableau","formula":formula}); columns["Keep frame"]=keep
    for name,formula in [("Selected frames","COUNTD([Frame])"),("Hidden frames","COUNTD(IF [Hidden collapse] THEN [Frame] END)"),
                         ("Hold fraction",'COUNTD(IF [Controller] = "RTA HOLD" THEN [Frame] END) / COUNTD([Frame])'),
                         ("Group note",'IF COUNTD([Frame]) < 120 THEN "Small group: fewer than 120 correlated frames" ELSE "One episode; frames are serially dependent" END')]:
        col=node(source,"column",name=f"[{name}]",datatype="string" if name=="Group note" else "real",role="measure",type="nominal" if name=="Group note" else "quantitative")
        node(col,"calculation",**{"class":"tableau","formula":formula}); columns[name]=col
    node(source,"aliases",enabled="yes")
    source_style=node(source,"style")
    color_rule=node(source_style,"style-rule",element="mark")
    for field,values in [("Allocation",{'"Proposed"':"#aa6a05",'"Applied"':"#167451"}),
                         ("Hidden collapse",{"true":"#c23442","false":"#346ea7"})]:
        encoding=node(color_rule,"encoding",attr="color",field=f"[none:{field}:nk]",type="palette")
        for value,color in values.items():node(node(encoding,"map",to=color),"bucket",value)
    sheets=node(root,"worksheets")
    names=[]
    def sheet(name,x,y,kind="Line",color=None,detail=None):
        names.append(name)
        ws=node(sheets,"worksheet",name=name)
        title=node(node(node(ws,"layout-options"),"title"),"formatted-text");node(title,"run",name,bold="true")
        table=node(ws,"table"); view=node(table,"view")
        ds=node(view,"datasources");node(ds,"datasource",caption="Synthetic agent simulation",name=DS);node(ds,"datasource",name="Parameters")
        dep=node(view,"datasource-dependencies",datasource=DS)
        used=set([x,y,"Keep frame","Phase","Controller"]+[v for v in [color,detail] if v])
        for field in used:dep.append(deepcopy(columns[field]))
        dep_p=node(view,"datasource-dependencies",datasource="Parameters");dep_p.append(deepcopy(p1));dep_p.append(deepcopy(p2))
        refs={}
        for field in used:
            is_dim=columns[field].get("role")=="dimension"
            derivation="None" if is_dim or field=="Time (s)" else "Avg"
            typ="nominal" if is_dim else "quantitative"
            suffix="nk" if is_dim else "qk"
            instance=f"[{derivation.lower()}:{field}:{suffix}]"
            node(dep,"column-instance",column=f"[{field}]",derivation=derivation,name=instance,pivot="key",type=typ)
            refs[field]=f"[{DS}].{instance}"
        fil=node(view,"filter",**{"class":"categorical","column":refs["Keep frame"]})
        node(fil,"groupfilter",function="member",level="[none:Keep frame:nk]",member="true")
        node(view,"aggregation",value="true")
        style=node(table,"style")
        sr=node(style,"style-rule",element="worksheet");node(sr,"format",attr="font-family",value="Segoe UI");node(sr,"format",attr="font-size",value="11")
        panes=node(table,"panes");pane=node(panes,"pane")
        node(node(pane,"view"),"breakdown",value="auto")
        node(pane,"mark",**{"class":kind})
        enc=node(pane,"encodings")
        if color:node(enc,"color",column=refs[color])
        if detail:node(enc,"lod",column=refs[detail])
        rows=refs[y]+" / "+refs[color] if kind=="Bar" and color else refs[y]
        node(table,"rows",rows);node(table,"cols",refs[x])
        return ws
    sheet("1  How concentrated is motion?","Time (s)","Leading share")
    sheet("2  Which actions receive weight?","Weight","Action","Bar","Allocation")
    sheet("3  Proposed and applied geometry","Time (s)","Conditioning","Line","Allocation")
    sheet("4  Can accurate tracking hide collapse?","Shadow RMSE (ft)","Shadow conditioning","Circle","Hidden collapse","Frame")
    dashboards=node(root,"dashboards");dash=node(dashboards,"dashboard",name="Multi-agent risk and assurance")
    dashboard_sources=node(dash,"datasources");node(dashboard_sources,"datasource",name="Parameters")
    dashboard_parameters=node(dash,"datasource-dependencies",datasource="Parameters")
    dashboard_parameters.append(deepcopy(p1));dashboard_parameters.append(deepcopy(p2))
    node(dash,"style");node(dash,"size",maxheight="900",maxwidth="1440",minheight="900",minwidth="1440")
    zones=node(dash,"zones")
    def text_zone(identifier,x,y,w,h,text,size=14):
        zone=node(zones,"zone",id=identifier,x=x,y=y,w=w,h=h,type_v2="text")
        formatted=node(zone,"formatted-text");node(formatted,"run",text,fontsize=size,fontname="Segoe UI")
        return zone
    text_zone(1,1800,1500,95000,7000,"Multi-agent risk and runtime assurance | Synthetic teaching simulation",23)
    text_zone(2,1800,8500,94000,4000,"One recorded episode • Six-second rolling windows • Proposed = gold; applied = green",13)
    for ident,col,x in [(3,"Phase filter",1800),(4,"Controller filter",40000)]:
        node(zones,"zone",id=ident,x=x,y="13500",w="36500",h="6500",type_v2="paramctrl",param=f"[Parameters].[{col}]",mode="compact")
    for i,(name,x,y) in enumerate(zip(names,[1800,50500,1800,50500],[22000,22000,57500,57500]),10):
        node(zones,"zone",id=i,x=x,y=y,w="47700",h="33500",name=name)
    text_zone(20,1800,92500,95000,6000,"Source: synchronized_frames.json, seed 20261013. Ratios are unitless; tracking error is in feet. Threshold: 0.12; low error: <0.5 ft; spacing: >=2 ft. Frames are dependent; small groups of 14 and 6 occur. No profitability or flight-validation claim.",11)
    windows=node(root,"windows",source_height="40")
    for name in names:
        window=node(windows,"window",**{"class":"worksheet","name":name})
        node(window,"viewpoint",zoom="entire-view")
    window=node(windows,"window",**{"class":"dashboard","name":"Multi-agent risk and assurance","maximized":"true"})
    node(window,"viewpoints")
    node(root,"thumbnails")
    file=DASH/"Multi_agent_risk_and_assurance.twb"
    ET.ElementTree(root).write(str(file),encoding="utf-8",xml_declaration=True,pretty_print=True)
    workbook=Workbook(str(file))
    parsed={"worksheets":workbook.worksheets,"datasources":len(workbook.datasources)}
    with zipfile.ZipFile(DASH/"Multi_agent_risk_and_assurance.twbx","w",zipfile.ZIP_DEFLATED) as package:
        package.write(file,file.name);package.write(DASH/"Data"/"risk.hyper","Data/risk.hyper")
    return parsed


def main():
    global OUT,DASH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=OUT)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    OUT=args.source.resolve()
    DASH=args.out.resolve() if args.out else OUT/"tableau"
    DASH.mkdir(parents=True,exist_ok=True)
    bundle=json.loads((OUT/"synchronized_frames.json").read_text(encoding="utf-8"));records=bundle["frames"]
    audit=json.loads((OUT/"validation.json").read_text())
    if not audit["checks_passed"]:raise RuntimeError("The simulation audit must pass before building a dashboard")
    profiles=[profile_csv(OUT/"frame_metrics.csv"),profile_csv(OUT/"action_weights.csv")]
    (DASH/"dataset_profile.json").write_text(json.dumps({"tables":profiles,"issues":audit["quality_notes"]},indent=2),encoding="utf-8")
    rows=flat_rows(records)
    with (DASH/"dashboard_data.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    kinds,checks=hyper_and_checks(rows,records)
    checks["workbook_parse"]=make_workbook(kinds,records)
    checks["source_sha256"]=hashlib.sha256((OUT/"synchronized_frames.json").read_bytes()).hexdigest()
    (DASH/"dashboard_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    preview_keys=["frame","time_s","phase","coverage","controller_decision","leading_share","eigenvalues",
                  "eigenvector_overlap","conditioning_proposed","conditioning_applied","conditioning_shadow",
                  "shadow_tracking_rmse_ft","spacing_shadow_ft","hidden_collapse","weights_proposed","weights_applied",
                  "offense_xy_ft","defense_xy_ft","proposed_xy_ft","ball_xy_ft"]
    compact=[{k:r[k] for k in preview_keys} for r in records]
    template=Path(__file__).with_name("portfolio_dashboard_preview.html")
    html=template.read_text(encoding="utf-8")
    html=html.replace("__FRAME_DATA__",json.dumps(compact,separators=(",",":"))).replace("__SOURCE_HASH__",checks["source_sha256"])
    (DASH/"dashboard_review.html").write_text(html,encoding="utf-8")
    if Path(__file__).resolve()!=DASH/Path(__file__).name:shutil.copy2(__file__,DASH/Path(__file__).name)
    if template.resolve()!=DASH/template.name:shutil.copy2(template,DASH/template.name)
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
