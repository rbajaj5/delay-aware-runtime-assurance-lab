"""Read-only loopback server for saved-run dashboard and artifacts."""
import argparse
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, unquote
ROOT=Path(__file__).resolve().parent

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path.startswith('/api/jobs/'):
            from scenario_runner import status
            try:self.json_response(status(path.rsplit('/',1)[-1]))
            except ValueError as e:self.json_response({'error':str(e)},400)
            return
        if path=='/':self.path='/app/index.html'
        elif not path.startswith(('/app/','/output/','/data/')):self.path='/app'+self.path
        if self.path.endswith('.json.gz'):
            target=Path(self.translate_path(self.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Encoding','gzip');self.send_header('Content-Length',str(target.stat().st_size))
            self.send_header('Cache-Control','no-cache');self.end_headers()
            with target.open('rb') as f:self.copyfile(f,self.wfile)
        else:super().do_GET()
    def log_message(self,*args):pass
    def json_response(self,value,code=200):
        raw=json.dumps(value,allow_nan=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
    def do_POST(self):
        if self.path!='/api/run':self.send_error(404);return
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:
            self.send_error(403);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=8192:raise ValueError('Invalid request size')
            from scenario_runner import submit
            self.json_response(submit(json.loads(self.rfile.read(size))))
        except (ValueError,KeyError,TypeError) as e:self.json_response({'error':str(e)},400)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    print(f'Local: http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
