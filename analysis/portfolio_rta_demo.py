"""Deterministic motion, covariance, Markowitz allocation, and affine RTA.

Run: python portfolio_rta_demo.py --out OUTPUT --encoder h264_nvenc
Requires Python 3.11+, numpy, scipy, Pillow, and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize


FPS = 24
DURATION = 90
SEED = 20261013
WINDOW_SECONDS = 6
GAMMA = 5.0
SHRINKAGE = 0.12
RIDGE = 0.015
COND_MIN = 0.12
COND_RESUME = 0.16
SPACING_MIN = 2.0
AREA_MIN = 0.07
TRACKING_MAX = 0.5
DT = 1 / FPS
ACTION_NAMES = ["High pick-and-roll", "Five-out drive", "Weak-side cut", "Post split", "Advance", "Reset"]
ACTION_CODES = ["PNR", "DRV", "CUT", "PST", "ADV", "RST"]
PLAYER_IDS = ["L1", "L2", "L3", "F1", "F2"]
ROLES = ["primary guard", "impact wing", "impact center", "weak-side wing", "spacing wing"]
RATINGS = [96, 93, 91, 86, 84]
NOMINAL = np.array([[25, 35], [3, 8], [47, 8], [7, 27], [43, 27]], dtype=float)
ORIGIN = np.array([25., 22.])
Q = NOMINAL - ORIGIN
DESIGN_EDGES = np.column_stack((Q[1] - Q[0], Q[2] - Q[0]))
DESIGN_INV = np.linalg.inv(DESIGN_EDGES)
BARY = np.linalg.solve(np.vstack((Q[:3].T, np.ones(3))), np.vstack((Q[3:].T, np.ones(2)))).T
ACTION_MAPS = np.array([
    [[.94, .07], [-.05, .83]], [[1., .04], [.04, .98]],
    [[.96, -.07], [.04, .92]], [[.97, .02], [.08, .79]],
    [[1.01, .0], [-.02, .96]], [[1., .0], [.0, 1.]],
])
PHASES = [(0, 15, "Stable five-out"), (15, 40, "Coverage changes"),
          (40, 65, "Allocation adapts"), (65, 80, "Geometry under stress"),
          (80, 90, "General multi-agent system")]
BG = "#11171b"
INK = "#eff4f4"
MUTED = "#a9b7bf"
GRID = "#33424a"
GREEN = "#5de0ac"
RED = "#ff6d78"
GOLD = "#f8c15d"
BLUE = "#70caff"
VIOLET = "#cfacf8"
COLORS = [BLUE, GOLD, VIOLET, "#168176", "#cb5872"]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def geometry(points):
    edges = np.column_stack((points[1] - points[0], points[2] - points[0]))
    a = edges @ DESIGN_INV
    sv = np.linalg.svd(a, compute_uv=False)
    dd = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    dd += np.eye(len(points)) * 1e6
    return a, sv, float(sv[1] / sv[0]), float(abs(np.linalg.det(a))), float(dd.min())


def optimal_weights(mu, cov, initial):
    result = minimize(lambda w: .5 * GAMMA * w @ cov @ w - mu @ w,
                      initial, jac=lambda w: GAMMA * cov @ w - mu,
                      method="SLSQP", bounds=[(0, 1)] * 6,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                    "jac": lambda w: np.ones(6)}],
                      options={"ftol": 1e-11, "maxiter": 100})
    if not result.success:
        raise RuntimeError(f"Allocation solver failed: {result.message}")
    w = np.maximum(result.x, 0)
    w /= w.sum()
    gradient = GAMMA * cov @ w - mu
    lagrange = -float(np.mean(gradient[w > 1e-7]))
    stationarity = gradient + lagrange
    kkt = max(float(abs(w.sum()-1)), float(np.max(abs(stationarity[w > 1e-7]))),
              float(np.max(np.maximum(-stationarity[w <= 1e-7], 0), initial=0)))
    return w, kkt


def simulate():
    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, 2*np.pi, (5, 2))
    window = deque(maxlen=WINDOW_SECONDS * FPS)
    init_w = np.full(6, 1/6)
    applied_w = init_w.copy()
    held_a = np.eye(2)
    held_center = ORIGIN.copy()
    actual = NOMINAL.copy()
    shadow = NOMINAL.copy()
    defense = NOMINAL + np.array([0., -3.])
    prev_off = actual.copy()
    prev_def = defense.copy()
    smooth_v_off = np.zeros((5, 2))
    smooth_v_def = np.zeros((5, 2))
    prev_v1 = None
    holding = False
    frozen_cov = None
    records = []
    feature_archive = []
    for step in range(-WINDOW_SECONDS * FPS - 48, FPS * DURATION):
        t = step * DT
        regime = float(smooth((t - 15) / 25))
        # A scripted environmental demand creates the geometric stress test.
        compression = 1 - .975 * smooth((t-60) / 7) * (1-smooth((t-79) / 5))
        measured_off = actual + rng.normal(0, .002, (5, 2))
        measured_def = defense + rng.normal(0, .002, (5, 2))
        smooth_v_off = .85 * smooth_v_off + .15 * (measured_off-prev_off) / DT
        smooth_v_def = .85 * smooth_v_def + .15 * (measured_def-prev_def) / DT
        prev_off, prev_def = measured_off.copy(), measured_def.copy()
        rel_v = (smooth_v_def - smooth_v_off) / 3.0
        # Six observable directional relative-velocity channels, all in units of 3 ft/s.
        features = np.array([rel_v[0, 0], rel_v[1, 1], rel_v[2, 0], rel_v[3, 1],
                             rel_v[4, 0], (rel_v[0, 1] + rel_v[4, 1]) / np.sqrt(2)])
        window.append(features.copy())
        feature_archive.append({"time_s": t, "features": features.tolist()})
        if len(window) < window.maxlen:
            cov = np.eye(6) * .08
        else:
            raw_cov = np.cov(np.array(window), rowvar=False, ddof=1)
            cov = (1-SHRINKAGE)*raw_cov + SHRINKAGE*np.trace(raw_cov)/6*np.eye(6) + RIDGE*np.eye(6)
        evals, evecs = np.linalg.eigh(cov)
        evals, evecs = evals[::-1], evecs[:, ::-1]
        v1 = evecs[:, 0]
        if prev_v1 is not None and v1 @ prev_v1 < 0:
            evecs[:, 0] *= -1
            v1 = evecs[:, 0]
        overlap = 1. if prev_v1 is None else float(abs(v1 @ prev_v1))
        prev_v1 = v1.copy()
        if frozen_cov is None or t < 14:
            frozen_cov = cov.copy()
        clearance = np.linalg.norm(measured_def - measured_off, axis=1)
        mu = np.array([.53, .50, .47, .45, .42, .38])
        mu += regime*np.array([-.14, -.12, .14, .08, -.03, .09])
        mu += .025 * np.array([clearance[0], clearance[1], clearance[3], clearance[2], clearance[4], 3.])
        mu += .025*np.array([np.sin(.22*t), np.cos(.16*t), np.sin(.19*t),
                             np.cos(.21*t), np.sin(.12*t), 0.])
        proposed_w, kkt = optimal_weights(mu, cov, init_w)
        init_w = proposed_w.copy()
        fixed_cov_w, _ = optimal_weights(mu, frozen_cov, proposed_w)
        proposed_a = np.diag([1., compression]) @ np.einsum("i,ijk->jk", proposed_w, ACTION_MAPS)
        proposed_center = ORIGIN + np.array([.45*np.sin(.23*t), .35*np.sin(.17*t)])
        proposed = Q @ proposed_a.T + proposed_center
        _, psv, pcond, parea, pspace = geometry(proposed)
        observed_a, osv, ocond, oarea, ospace = geometry(measured_off)
        # Weyl's singular-value perturbation bound for bounded leader-position uncertainty.
        eps_a = 2*np.sqrt(2)*.015*np.linalg.norm(DESIGN_INV, 2)
        sv_lower = max(0., psv[1]-eps_a) / (psv[0]+eps_a)
        identity_ok = len(set(PLAYER_IDS)) == 5
        observation_ok = np.isfinite(measured_off).all() and np.isfinite(measured_def).all()
        threshold = COND_RESUME if holding else COND_MIN
        reasons = []
        if sv_lower < threshold:
            reasons.append("affine conditioning")
        if parea < AREA_MIN:
            reasons.append("area scale")
        if pspace < SPACING_MIN:
            reasons.append("spacing")
        if not identity_ok:
            reasons.append("identity")
        if not observation_ok:
            reasons.append("observation quality")
        holding = bool(reasons)
        if not holding:
            held_a, held_center, applied_w = proposed_a.copy(), proposed_center.copy(), proposed_w.copy()
        safe_target = Q @ held_a.T + held_center
        disturbance = .018*np.column_stack((np.sin(1.1*t + phases[:, 0]), np.cos(.9*t+phases[:, 1])))
        gain = 1 - np.exp(-DT/.16)
        actual += gain * (safe_target + disturbance - actual)
        shadow += gain * (proposed + disturbance - shadow)
        toward = np.array([25., 5.25])-actual
        toward /= np.linalg.norm(toward, axis=1, keepdims=True)
        drop = actual + 3.0*toward
        switched = actual[[2, 0, 1, 4, 3]] + 3.0*toward
        center_help = .1*(ORIGIN-actual)
        load_angles = np.array([0., .35, -.4, .65, -.2]) + .95*regime
        common = (np.sin(1.6*t)+.3*np.sin(2.4*t))*(3.4*regime)
        dirs = np.column_stack((np.cos(load_angles), np.sin(load_angles)))
        independent = .48*np.column_stack((np.sin(t*np.array([1.17,1.51,1.83,2.09,2.47])+phases[:, 0]),
                                            np.cos(t*np.array([1.37,1.67,1.97,2.23,2.61])+phases[:, 1])))
        defense_target = (1-regime)*drop + regime*(switched+center_help) + independent + common*dirs
        defense_target = np.clip(defense_target, [1., 1.], [49., 46.])
        defense += (1-np.exp(-DT/.2))*(defense_target-defense)
        _, asv, acond, aarea, aspace = geometry(actual)
        _, ssv, scond, sarea, sspace = geometry(shadow)
        rmse = float(np.sqrt(np.mean(np.sum((actual-safe_target)**2, axis=1))))
        shadow_rmse = float(np.sqrt(np.mean(np.sum((shadow-proposed)**2, axis=1))))
        follower_expected = BARY @ safe_target[:3]
        follower_rmse = float(np.sqrt(np.mean(np.sum((actual[3:]-follower_expected)**2, axis=1))))
        pass_cycle = [0, 2, 3, 1, 4, 0]
        pass_idx = int(max(t, 0)//3.2) % len(pass_cycle)
        ball_from, ball_to = pass_cycle[pass_idx], pass_cycle[(pass_idx+1)%len(pass_cycle)]
        pass_fraction = float(smooth((max(t, 0)%3.2-2.65)/.55))
        ball = (1-pass_fraction)*actual[ball_from] + pass_fraction*actual[ball_to]
        ball += np.array([.45, .45]) * (1-pass_fraction)
        if step < 0:
            continue
        phase = next(p[2] for p in PHASES if p[0] <= t < p[1])
        records.append({
            "frame": step, "time_s": t, "possession_id": "SYNTH-001", "phase": phase,
            "coverage": "DROP" if t < 15 else ("TRANSITION" if t < 40 else "SWITCH + HELP"),
            "coverage_blend": regime, "compression_request": float(compression),
            "application_view": "BASKETBALL" if t < 80 else "SCHEMATIC DRONES",
            "offense_xy_ft": actual.tolist(), "defense_xy_ft": defense.tolist(), "ball_xy_ft": ball.tolist(),
            "proposed_xy_ft": proposed.tolist(), "unshielded_shadow_xy_ft": shadow.tolist(),
            "observed_offense_xy_ft": measured_off.tolist(), "observed_defense_xy_ft": measured_def.tolist(),
            "features": features.tolist(), "covariance": cov.tolist(), "eigenvalues": evals.tolist(),
            "leading_eigenvector": v1.tolist(), "leading_share": float(evals[0]/evals.sum()),
            "eigenvector_overlap": overlap, "eigenvalue_gap": float(evals[0]-evals[1]),
            "covariance_condition_number": float(evals[0]/evals[-1]),
            "utility": mu.tolist(), "weights_proposed": proposed_w.tolist(), "weights_applied": applied_w.tolist(),
            "weights_frozen_covariance": fixed_cov_w.tolist(),
            "risk_contributions": (evals*(evecs.T@proposed_w)**2).tolist(),
            "risk_variance": float(proposed_w@cov@proposed_w), "optimizer_kkt_residual": kkt,
            "proposed_affine_map": proposed_a.tolist(), "applied_affine_map": held_a.tolist(),
            "applied_center_ft": held_center.tolist(), "sigma_proposed": psv.tolist(), "sigma_applied": asv.tolist(),
            "conditioning_proposed": pcond, "conditioning_applied": acond, "conditioning_shadow": scond,
            "conditioning_lower_bound": sv_lower, "area_proposed": parea, "area_applied": aarea,
            "spacing_proposed_ft": pspace, "spacing_applied_ft": aspace, "spacing_shadow_ft": sspace,
            "tracking_rmse_ft": rmse, "shadow_tracking_rmse_ft": shadow_rmse,
            "follower_rmse_ft": follower_rmse, "controller_decision": "RTA HOLD" if holding else "ACCEPT",
            "reject_reasons": reasons, "identity_continuity": bool(identity_ok),
            "observation_valid_fraction": float(observation_ok), "observation_age_s": DT,
            "hidden_collapse": bool(scond < COND_MIN and shadow_rmse < TRACKING_MAX and sspace >= SPACING_MIN),
        })
    return records, feature_archive


def write_data(out, records, features):
    metadata = {"synthetic": True, "seed": SEED, "fps": FPS, "duration_s": DURATION,
                "gamma": GAMMA, "window_seconds": WINDOW_SECONDS, "shrinkage": SHRINKAGE,
                "ridge": RIDGE, "conditioning_threshold": COND_MIN, "resume_threshold": COND_RESUME,
                "spacing_threshold_ft": SPACING_MIN, "area_threshold": AREA_MIN,
                "tracking_threshold_ft": TRACKING_MAX, "action_names": ACTION_NAMES,
                "player_ids": PLAYER_IDS, "roles": ROLES, "impact_ratings": RATINGS,
                "nominal_xy_ft": NOMINAL.tolist(), "follower_affine_weights": BARY.tolist(),
                "note": "Synthetic kinematic teaching scenario; no real players, returns, or operational flight validation."}
    with (out/"synchronized_frames.json").open("w", encoding="utf-8") as stream:
        json.dump({"metadata": metadata, "frames": records, "feature_prewarm": features[:192]}, stream, separators=(",", ":"), allow_nan=False)
    fields = ["frame", "time_s", "possession_id", "phase", "coverage", "application_view", "controller_decision",
              "leading_share", "eigenvector_overlap", "covariance_condition_number", "conditioning_proposed",
              "conditioning_applied", "area_proposed", "area_applied", "spacing_proposed_ft", "spacing_applied_ft",
              "tracking_rmse_ft", "shadow_tracking_rmse_ft", "follower_rmse_ft", "hidden_collapse",
              "identity_continuity", "observation_valid_fraction", "observation_age_s"]
    with (out/"frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["lambda1", "lambda2", "lambda3"])
        writer.writeheader()
        for r in records:
            writer.writerow({**{f:r[f] for f in fields}, **{f"lambda{i+1}":r["eigenvalues"][i] for i in range(3)}})
    with (out/"action_weights.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "phase", "coverage", "controller_decision", "action", "proposed_weight", "applied_weight"])
        for r in records:
            for j, name in enumerate(ACTION_NAMES):
                writer.writerow([r["frame"],r["time_s"],r["phase"],r["coverage"],r["controller_decision"],name,r["weights_proposed"][j],r["weights_applied"][j]])


def audit(out, records, features):
    errors = []
    max_cov_error, max_risk_error, max_map_error = 0., 0., 0.
    all_features = np.array([r["features"] for r in features])
    hold_start = None
    for i, r in enumerate(records):
        vals = np.array(r["eigenvalues"])
        cov = np.array(r["covariance"])
        original_index = i+192
        segment = all_features[original_index-143:original_index+1]
        sample = np.cov(segment, rowvar=False, ddof=1)
        rebuilt = (1-SHRINKAGE)*sample + SHRINKAGE*np.trace(sample)/6*np.eye(6) + RIDGE*np.eye(6)
        max_cov_error = max(max_cov_error, float(abs(cov-rebuilt).max()))
        max_risk_error = max(max_risk_error, abs(sum(r["risk_contributions"])-r["risk_variance"]))
        if not np.all(vals > 0): errors.append(f"Non-SPD covariance at {i}")
        for key in ["weights_proposed", "weights_applied"]:
            if min(r[key]) < -1e-8 or abs(sum(r[key])-1) > 1e-8: errors.append(f"Invalid weights {i}")
        if r["controller_decision"] == "RTA HOLD":
            if hold_start is None: hold_start = r["time_s"]
            if i:
                max_map_error = max(max_map_error, float(abs(np.array(r["applied_affine_map"])-records[i-1]["applied_affine_map"]).max()))
                if not np.allclose(r["weights_applied"], records[i-1]["weights_applied"], atol=1e-12): errors.append(f"Weights changed during HOLD at {i}")
        if r["conditioning_applied"] < COND_MIN: errors.append(f"Applied affine map below threshold at {i}")
        if r["spacing_applied_ft"] < SPACING_MIN: errors.append(f"Applied spacing below threshold {i}")
    def avg(key, start, end):
        return float(np.mean([r[key] for r in records if start <= r["time_s"] < end]))
    phase_counts = {p[2]:sum(r["phase"]==p[2] for r in records) for p in PHASES}
    result = {
        "frames": len(records), "duration_s": len(records)/FPS, "seed": SEED,
        "errors": errors, "checks_passed": not errors and max_cov_error < 1e-10 and max_map_error < 1e-12,
        "covariance_recomputation_max_abs_error": max_cov_error,
        "spectral_risk_recomputation_max_abs_error": max_risk_error,
        "held_map_max_change": max_map_error,
        "max_optimizer_kkt_residual": max(r["optimizer_kkt_residual"] for r in records),
        "first_hold_time_s": hold_start,
        "hold_frames": sum(r["controller_decision"]=="RTA HOLD" for r in records),
        "hidden_collapse_shadow_frames": sum(r["hidden_collapse"] for r in records),
        "minimum_applied_conditioning": min(r["conditioning_applied"] for r in records),
        "minimum_proposed_conditioning": min(r["conditioning_proposed"] for r in records),
        "minimum_shadow_spacing_ft": min(r["spacing_shadow_ft"] for r in records),
        "stable_mean_leading_share": avg("leading_share", 5, 15),
        "switch_help_mean_leading_share": avg("leading_share", 35, 55),
        "stable_mean_lambda1": float(np.mean([r["eigenvalues"][0] for r in records if 5<=r["time_s"]<15])),
        "switch_help_mean_lambda1": float(np.mean([r["eigenvalues"][0] for r in records if 35<=r["time_s"]<55])),
        "mean_L1_weight_difference_from_frozen_covariance": float(np.mean([sum(abs(np.array(r["weights_proposed"])-r["weights_frozen_covariance"])) for r in records if 35<=r["time_s"]<55])),
        "phase_counts": phase_counts,
        "independent_episodes": 1,
        "null_scalar_values": sum(v is None for r in records for v in r.values()),
        "quality_notes": ["Overlapping windows are serially dependent; frames are not independent trials.",
                          "Adjacent-frame eigenvector overlap can stay near one during gradual cumulative rotation.",
                          "Covariance is regularized; its condition number is estimator-dependent.",
                          "Basketball-to-drone morph reuses the same coordinates; it is not a second experiment.",
                          "Advance is a half-court surrogate for transition, not a full-court play."]}
    (out/"validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["checks_passed"]:
        raise AssertionError("Data audit failed")
    return result


class Movie:
    def __init__(self, records):
        self.records = records
        self.fonts = {}
        self.data = {k:np.array([r[k] for r in records]) for k in ["eigenvalues", "leading_share", "eigenvector_overlap", "covariance_condition_number", "conditioning_proposed", "conditioning_applied"]}
        self.cov_limit = max(np.max(np.abs(r["covariance"])) for r in records)
        self.base = self.make_base()

    def font(self, size, bold=False):
        key=(size,bold)
        if key not in self.fonts:
            paths = [Path("C:/Windows/Fonts")/("segoeuib.ttf" if bold else "segoeui.ttf"),
                     Path("/usr/share/fonts/truetype/dejavu")/("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
            self.fonts[key] = ImageFont.truetype(str(next(p for p in paths if p.exists())), size)
        return self.fonts[key]

    def text(self, draw, xy, value, size=21, color=INK, bold=False, anchor=None):
        draw.text(xy, value, font=self.font(size,bold), fill=color, anchor=anchor)

    def xy(self, p):
        return (65+float(p[0])*14.6, 226+(47-float(p[1]))*14.0)

    def make_base(self):
        im=Image.new("RGB",(1920,1080),BG)
        d=ImageDraw.Draw(im)
        self.text(d,(36,24),"A formation under supervision",38,bold=True)
        self.text(d,(38,77),"Synthetic agents  /  rolling covariance  /  risk-aware allocation  /  independent runtime assurance",22,MUTED)
        d.line((866,163,866,954),fill=GRID,width=2)
        self.text(d,(900,169),"Risk-aware allocation",24,bold=True)
        self.text(d,(900,208),"max  μᵀw − (γ/2) wᵀΣ̂w     subject to w ≥ 0,  1ᵀw = 1",24)
        self.text(d,(901,250),"Covariance Σ̂  ·  6 s window",19,MUTED)
        self.text(d,(1398,250),"Action mix",22,bold=True)
        self.text(d,(1650,252),"proposed",17,GOLD)
        self.text(d,(1762,252),"applied",17,GREEN)
        self.text(d,(900,493),"Risk uses every mode:  wᵀΣ̂w = Σᵢ λᵢ(vᵢᵀw)²     γ = 5",21,MUTED)
        return im

    def plot(self,d,i,top,bottom,series,limits,title,threshold=None):
        left,right=954,1875
        lo,hi=limits
        self.text(d,(900,top-29),title,19,MUTED)
        for val in [lo,(lo+hi)/2,hi]:
            y=bottom-(val-lo)/(hi-lo)*(bottom-top)
            d.line((left,y,right,y),fill=GRID,width=1)
            self.text(d,(946,y),f"{val:.1f}",15,MUTED,anchor="rm")
        for sec in [0,15,40,65,80,90]:
            x=left+sec/90*(right-left)
            d.line((x,top,x,bottom),fill=GRID,width=1)
        if threshold is not None:
            yy=bottom-(threshold-lo)/(hi-lo)*(bottom-top)
            for xx in range(left,right,12): d.line((xx,yy,min(xx+6,right),yy),fill=RED,width=1)
        indices=np.arange(i+1)
        if len(indices)>1000: indices=np.unique(np.r_[indices[::2],i])
        for values,color in series:
            points=[(left+j/(FPS*90)*(right-left),bottom-(np.clip(float(values[j]),lo,hi)-lo)/(hi-lo)*(bottom-top)) for j in indices]
            if len(points)>1: d.line(points,fill=color,width=3)
        xx=left+i/(FPS*90)*(right-left)
        d.line((xx,top-2,xx,bottom+2),fill=INK,width=2)
        if top>830:
            for sec in [0,15,40,65,80,90]:
                self.text(d,(left+sec/90*(right-left),bottom+7),str(sec),15,MUTED,anchor="mt")

    def frame(self,i):
        r=self.records[i]
        t=r["time_s"]
        im=self.base.copy(); d=ImageDraw.Draw(im)
        self.text(d,(1878,31),f"{t:05.1f} / 90.0 s",25,INK,anchor="rt")
        phase_number=next(k for k,p in enumerate(PHASES) if p[0]<=t<p[1])
        for k,(start,end,label) in enumerate(PHASES):
            x0=38+start/90*1836; x1=38+end/90*1836
            d.rectangle((x0,119,x1-3,151),fill=GREEN if k==phase_number else GRID)
            self.text(d,((x0+x1)/2,133),"Domain transfer" if k==4 else label,17,BG if k==phase_number else INK,anchor="mm")
        morph=float(smooth((t-80)/4))
        hold=r["controller_decision"]=="RTA HOLD"
        self.text(d,(39,170),"Synthetic five-on-five" if morph<.5 else "Same state · schematic quadrotors",25,bold=True)
        self.text(d,(39,204),"L1 guard   L2 wing   L3 center   |   F1, F2 followers",19,MUTED)
        d.rectangle((65,226,795,884),fill="#e7eef0",outline="#83949c",width=2)
        def line(points,fill="#aabac1",width=2): d.line([self.xy(p) for p in points],fill=fill,width=width)
        if morph<1:
            court_color=tuple(int((1-morph)*x+morph*y) for x,y in zip((125,146,154),(231,238,240)))
            line([(17,0),(17,19),(33,19),(33,0)],court_color)
            hoop=self.xy((25,5.25)); d.ellipse((hoop[0]-10,hoop[1]-10,hoop[0]+10,hoop[1]+10),outline=court_color,width=3)
            line([(22,4),(28,4)],court_color,3)
            for theta in np.linspace(0,np.pi,90)[:-1]:
                p=(25+23.75*np.cos(theta),5.25+23.75*np.sin(theta))
                p2=(25+23.75*np.cos(theta+.035),5.25+23.75*np.sin(theta+.035))
                line([p,p2],court_color)
        if morph>.1:
            for xx in np.arange(5,50,5): line([(xx,0),(xx,47)],"#d2dfe2",1)
            for yy in np.arange(5,47,5): line([(0,yy),(50,yy)],"#d2dfe2",1)
        self.text(d,(79,239),r["coverage"] if morph<.5 else "Schematic top-down view; same trajectory",20,"#344b57",True)
        # Followers retain their design-time affine coefficients.
        actual=np.array(r["offense_xy_ft"]); prop=np.array(r["proposed_xy_ft"])
        for follower in [3,4]:
            for leader in [0,1,2]: line([actual[follower],actual[leader]],"#b0c9cb",1)
        line([actual[0],actual[1],actual[2],actual[0]],"#147c61",3)
        if hold:
            line([prop[0],prop[1],prop[2],prop[0]],"#d12442",4)
            for j,p in enumerate(prop):
                px,py=self.xy(p); d.ellipse((px-13,py-13,px+13,py+13),outline="#d12442",width=2)
        if morph<1:
            for j,p in enumerate(r["defense_xy_ft"]):
                px,py=self.xy(p)
                color=tuple(int((1-morph)*x+morph*y) for x,y in zip((75,87,98),(231,238,240)))
                d.ellipse((px-14,py-14,px+14,py+14),fill=color,outline="#e7eef0",width=1)
                if morph<.5:self.text(d,(px,py),f"D{j+1}",12,"#ffffff",anchor="mm")
        for j,p in enumerate(actual):
            px,py=self.xy(p)
            if morph>.4:
                for dx,dy in [(-12,-12),(12,-12),(-12,12),(12,12)]:
                    d.line((px,py,px+dx,py+dy),fill="#33484d",width=2)
                    d.ellipse((px+dx-7,py+dy-7,px+dx+7,py+dy+7),outline=COLORS[j],width=2)
            d.ellipse((px-17,py-17,px+17,py+17),fill=COLORS[j],outline="#10262b",width=3 if j<3 else 1)
            self.text(d,(px,py),PLAYER_IDS[j],16,"#10262b" if j<3 else "#ffffff",True,anchor="mm")
        if morph<.5:
            bx,by=self.xy(r["ball_xy_ft"])
            d.ellipse((bx-7,by-7,bx+7,by+7),fill="#ed7d31",outline="#522a16",width=2)
        self.text(d,(79,849),"Solid: applied   Red outline: rejected proposal" if hold else "Leader triangle + fixed follower affine links",18,"#344b57")
        self.text(d,(37,909),r["controller_decision"],26,RED if hold else GREEN,True)
        self.text(d,(307,908),f"Spacing: {r['spacing_applied_ft']:.2f} ft    Area scale: {r['area_applied']:.3f}",20)
        self.text(d,(38,944),f"Applied tracking: {r['tracking_rmse_ft']:.3f} ft    Follower RMSE: {r['follower_rmse_ft']:.3f} ft",20,MUTED)
        self.text(d,(38,975),f"Unshielded tracking: {r['shadow_tracking_rmse_ft']:.3f} ft  |  {'HIDDEN COLLAPSE' if r['hidden_collapse'] else 'geometry monitored'}",20,RED if r['hidden_collapse'] else MUTED)
        self.text(d,(38,1007),"Identities: 5/5 continuous    Observations: 100% valid",18,MUTED)
        # Fixed heatmap color scale ensures changes are not due to recoloring each frame.
        cov=np.array(r["covariance"])
        for a in range(6):
            for b in range(6):
                val=float(cov[a,b]/self.cov_limit)
                color0=np.array([43,56,63]); color1=np.array([94,224,174] if val>=0 else [250,113,132])
                color=tuple((color0+(color1-color0)*np.sqrt(abs(val))).astype(int))
                x,y=919+b*32,300+a*29
                d.rectangle((x,y,x+30,y+27),fill=color)
            self.text(d,(912,313+a*29),str(a+1),14,MUTED,anchor="rm")
            self.text(d,(934+a*32,288),str(a+1),14,MUTED,anchor="mm")
        self.text(d,(900,477),f"fixed scale ±{self.cov_limit:.2f}",15,MUTED)
        self.text(d,(1150,285),f"λ₁ / trace   {r['leading_share']:.3f}",21,GREEN)
        self.text(d,(1150,323),f"Overlap      {r['eigenvector_overlap']:.5f}",20)
        self.text(d,(1150,361),f"κ(Σ̂)          {r['covariance_condition_number']:.2f}",20)
        self.text(d,(1150,400),f"σ min/max    {r['conditioning_proposed']:.3f}",19,RED if hold else GOLD)
        self.text(d,(1150,439),f"σ = {r['sigma_proposed'][1]:.3f}, {r['sigma_proposed'][0]:.3f}",18,MUTED)
        self.text(d,(1150,472),"PC1",14,MUTED)
        for a,loading in enumerate(r["leading_eigenvector"]):
            xx=1200+26*a
            yy=480-20*loading
            d.line((xx,480,xx+18,480),fill=GRID,width=1)
            d.rectangle((xx,min(480,yy),xx+18,max(480,yy)+1),fill=BLUE)
        for a,name in enumerate(["Pick & roll","Five-out drive","Weak-side cut","Post split","Advance","Reset"]):
            y=289+a*31
            self.text(d,(1398,y+6),name,16)
            d.rectangle((1515,y,1765,y+8),fill=GRID)
            d.rectangle((1515,y+12,1765,y+20),fill=GRID)
            d.rectangle((1515,y,1515+250*r["weights_proposed"][a],y+8),fill=GOLD)
            d.rectangle((1515,y+12,1515+250*r["weights_applied"][a],y+20),fill=GREEN)
            self.text(d,(1790,y+5),f"{100*r['weights_proposed'][a]:2.0f}/{100*r['weights_applied'][a]:2.0f}%",16)
        eig=self.data["eigenvalues"]
        self.plot(d,i,548,606,[(eig[:,0],GREEN),(eig[:,1],GOLD),(eig[:,2],VIOLET)],(0,float(eig[:,0].max()*1.05)),f"Top eigenvalues  ·  λ₁ {r['eigenvalues'][0]:.3f}   λ₂ {r['eigenvalues'][1]:.3f}   λ₃ {r['eigenvalues'][2]:.3f}")
        self.plot(d,i,648,706,[(self.data["leading_share"],GREEN),(self.data["eigenvector_overlap"],BLUE)],(0,1),"Leading share (green) + adjacent-frame overlap (blue)")
        self.plot(d,i,750,802,[(self.data["covariance_condition_number"],VIOLET)],(0,float(self.data["covariance_condition_number"].max()*1.08)),"Covariance condition number  κ(Σ̂)")
        self.plot(d,i,847,905,[(self.data["conditioning_proposed"],GOLD),(self.data["conditioning_applied"],GREEN)],(0,1),"Affine σmin / σmax  ·  proposed / applied   ·   red threshold 0.12",COND_MIN)
        if t<15:
            caption="Independent motion keeps risk diversified. The controller accepts this geometry."
        elif t<40:
            caption="Switch-and-help couples motions. The rolling covariance spectrum changes with the agents."
        elif t<65:
            caption="Risk = covariance weighted by the action mix. New estimates change the proposed allocation."
        elif t<80:
            caption="Low tracking error can coexist with collapse. RTA holds the last accepted map AND weights."
        else:
            caption="Players → robots → software agents: covariance → allocation → runtime assurance."
        lines=[""]
        for word in caption.split():
            trial=(lines[-1]+" "+word).strip()
            if d.textlength(trial,font=self.font(20))>970: lines.append(word)
            else: lines[-1]=trial
        for n,line_text in enumerate(lines):self.text(d,(900,959+29*n),line_text,20,INK)
        self.text(d,(38,1047),"SYNTHETIC KINEMATIC DEMO  |  One 90-second teaching sequence; no game clock  |  No betting-profit or flight-validation claim",18,MUTED)
        return im


def render(out, records, encoder):
    painter=Movie(records)
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg: raise RuntimeError("ffmpeg must be installed and on PATH")
    target=out/"portfolio_rta_synchronized.mp4"
    args=[ffmpeg,"-y","-hide_banner","-loglevel","warning","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(FPS),"-i","-","-an","-c:v",encoder]
    args += ["-preset","p5","-cq","21","-b:v","0"] if encoder=="h264_nvenc" else ["-preset","fast","-crf","20"]
    args += ["-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
    with (out/"ffmpeg.log").open("w") as log:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
        try:
            for i in range(len(records)):
                im=painter.frame(i)
                if i in [8*FPS,30*FPS,52*FPS,72*FPS,87*FPS]: im.save(out/f"preview_{i//FPS:02d}s.png")
                proc.stdin.write(im.tobytes())
                if i % (10*FPS)==0:print(f"Rendered {i//FPS:02d}/90 seconds",flush=True)
        finally:
            proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError((out/"ffmpeg.log").read_text())
    print(f"Movie: {target}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"artifacts"/"20261013"/"portfolio_rta")
    parser.add_argument("--encoder",choices=["libx264","h264_nvenc"],default="libx264")
    parser.add_argument("--data-only",action="store_true")
    parser.add_argument("--render-existing",action="store_true")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if args.render_existing:
        records=json.loads((args.out/"synchronized_frames.json").read_text())["frames"]
    else:
        records,features=simulate()
        write_data(args.out,records,features)
        audit(args.out,records,features)
    if not args.data_only:render(args.out,records,args.encoder)
    source=Path(__file__)
    if source.resolve()!= (args.out/source.name).resolve():shutil.copy2(source,args.out/source.name)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file() and p.name!="manifest.json"}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
