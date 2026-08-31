"""Local console for the agent: replay a recorded run, edit the prompt, launch a new one.

    .venv/Scripts/python.exe -m web.server        then open http://127.0.0.1:8765

Stdlib only, on purpose - the competition environment installs from
requirements.txt and this must never become a reason that fails. No Flask, no
build step, no node.

Read-only over the run records. The one thing it writes is
agent/prompt_override.txt, and the one thing it executes is `python -m agent`,
which is the same command a human would type. There is no separate code path
for a run started here: it produces an ordinary run folder that the ledger,
gendiffs.py and every other tool already understand.
"""
import http.server
import io
import json
import mimetypes
import os
import queue
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, 'web')
sys.path.insert(0, ROOT)

PORT = int(os.environ.get('WEB_PORT', 8765))
LOG_ROOTS = {'pure': 'logs', '1k': 'logs-1k', '27k': 'logs-27k'}


# --------------------------------------------------------------------------
# reading the record
# --------------------------------------------------------------------------

def _iter_files(run_dir):
    if not os.path.isdir(run_dir):
        return []
    return sorted(f for f in os.listdir(run_dir)
                  if re.fullmatch(r'\d{4}\.json', f))


def _read_json(path):
    try:
        with io.open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def list_runs():
    """Every run folder that has at least one iteration record."""
    out = []
    for ds, root in LOG_ROOTS.items():
        base = os.path.join(ROOT, root)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            run_dir = os.path.join(base, name)
            files = _iter_files(run_dir)
            if not files:
                continue
            best = None
            kept = failed = noop = 0
            cost = 0.0
            model = None
            first = last = None
            for fn in files:
                rec = _read_json(os.path.join(run_dir, fn))
                if not rec:
                    continue
                model = rec.get('model') or model
                v = rec.get('valid_primary')
                verdict = rec.get('verdict')
                # A dev-screen or duplicate row is not a comparable score; it
                # must not be allowed to set a run's headline number.
                if v is not None and verdict not in ('screen', 'duplicate'):
                    best = v if best is None else max(best, v)
                if verdict == 'KEPT':
                    kept += 1
                elif verdict == 'failed':
                    failed += 1
                elif verdict == 'no-op':
                    noop += 1
                cost += rec.get('cost_usd') or 0.0
                ts = rec.get('timestamp')
                if ts:
                    first = first or ts
                    last = ts
            out.append({
                'id': '%s/%s' % (root, name),
                'name': name,
                'dataset': ds,
                'iterations': len(files),
                'best': best,
                'kept': kept,
                'failed': failed,
                'noop': noop,
                'cost_usd': round(cost, 2),
                'model': model,
                'started': first,
                'ended': last,
                'events': _count_events(run_dir),
                'diffs': os.path.isdir(os.path.join(run_dir, 'diffs')),
            })
    return out


def _count_events(run_dir):
    p = os.path.join(run_dir, 'events.jsonl')
    if not os.path.isfile(p):
        return 0
    try:
        with io.open(p, encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _safe_run_dir(run_id):
    """Resolve a run id to a directory, refusing anything outside the log roots."""
    if not run_id or not re.fullmatch(r'[\w.-]+/[\w.-]+', run_id):
        return None
    root, name = run_id.split('/', 1)
    if root not in LOG_ROOTS.values():
        return None
    run_dir = os.path.normpath(os.path.join(ROOT, root, name))
    if not run_dir.startswith(os.path.join(ROOT, root)):
        return None
    return run_dir if os.path.isdir(run_dir) else None


def read_run(run_id):
    run_dir = _safe_run_dir(run_id)
    if not run_dir:
        return None
    iterations = [_read_json(os.path.join(run_dir, fn))
                  for fn in _iter_files(run_dir)]
    iterations = [r for r in iterations if r]
    events = []
    ev_path = os.path.join(run_dir, 'events.jsonl')
    if os.path.isfile(ev_path):
        with io.open(ev_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    diffs = {}
    ddir = os.path.join(run_dir, 'diffs')
    if os.path.isdir(ddir):
        for fn in sorted(os.listdir(ddir)):
            m = re.fullmatch(r'(\d{4})\.diff', fn)
            if not m:
                continue
            try:
                with io.open(os.path.join(ddir, fn), encoding='utf-8') as f:
                    diffs[str(int(m.group(1)))] = f.read()
            except Exception:
                pass
    return {'id': run_id, 'iterations': iterations,
            'events': events, 'diffs': diffs}


def read_solution(run_id, iteration):
    """The full source the agent wrote for one iteration."""
    run_dir = _safe_run_dir(run_id)
    if not run_dir:
        return None
    rec = _read_json(os.path.join(run_dir, '%04d.json' % int(iteration)))
    if not rec:
        return None
    rel = rec.get('solution') or ''
    path = os.path.normpath(os.path.join(ROOT, rel))
    if not path.startswith(ROOT) or not os.path.isfile(path):
        return None
    with io.open(path, encoding='utf-8') as f:
        return {'path': rel, 'source': f.read()}


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

def prompt_state():
    from agent import prompt as P
    text, source = P.raw_system_prompt()
    ident = P.prompt_identity()
    return {
        'text': text,
        'shipped': P.SYSTEM_PROMPT,
        'source': source,
        'hash': ident['prompt_hash'],
        'chars': len(text),
        'lines': text.count('\n') + 1,
        'edited': source == 'override' and text != P.SYSTEM_PROMPT,
    }


def save_prompt(text):
    """Validate then write the override. Invalid templates are refused, not saved.

    The agent formats this string, so an unbalanced brace is a crash 40 minutes
    into a run. Catching it here costs nothing and turns a wasted run into an
    inline error message.
    """
    from agent import prompt as P
    for ds, cfg in P._DATASET_CONFIG.items():
        try:
            text.format(ds_name=cfg['name'], baseline=cfg['baseline'],
                        target=cfg['target'], suffix=cfg['suffix'],
                        data_rel=cfg['data_rel'])
        except (KeyError, IndexError, ValueError) as e:
            return {'ok': False, 'error': (
                'Not a usable prompt template (%s: %s). Literal { and } must '
                'be doubled as {{ and }}. The only single-brace names allowed '
                'are ds_name, baseline, target, suffix, data_rel.'
                % (ds, e))}
    if text.strip() == P.SYSTEM_PROMPT.strip():
        if os.path.isfile(P.OVERRIDE_PATH):
            os.remove(P.OVERRIDE_PATH)
        return {'ok': True, 'reverted': True}
    with io.open(P.OVERRIDE_PATH, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return {'ok': True}


def reset_prompt():
    from agent import prompt as P
    if os.path.isfile(P.OVERRIDE_PATH):
        os.remove(P.OVERRIDE_PATH)
    return {'ok': True}


def preflight(dataset='pure'):
    """Exactly what the model will be sent, assembled by the agent's own code.

    Not a description of the prompt and not a copy of it - system_prompt() and
    build_user_message() are the functions the loop itself calls, so this
    screen cannot drift away from what actually runs.
    """
    import harness.ledger as ledger
    from agent import prompt as P
    from agent import tools
    prev = getattr(ledger, 'RUN_DIR', None)
    try:
        ledger.use_dataset(dataset)
        system = P.system_prompt(dataset)
        try:
            user = P.build_user_message()
        except Exception as e:
            user = '(could not assemble: %s)' % e
    finally:
        if prev:
            try:
                ledger.init_run_dir(prev)
            except Exception:
                pass
    ident = P.prompt_identity(dataset)
    total_chars = len(system) + len(user)
    return {
        'system': system,
        'user': user,
        'identity': ident,
        'chars': total_chars,
        # ~4 chars/token is the usual English rough cut; this is a sanity
        # figure for the confirm screen, not an accounting number.
        'est_tokens': int(total_chars / 4),
        'tools': [t['function']['name'] for t in tools.TOOL_SCHEMAS],
        'baseline': ledger.BASELINE_VALID,
        'epsilon': getattr(ledger, 'EPSILON', 0.002),
    }


# --------------------------------------------------------------------------
# launching a run
# --------------------------------------------------------------------------

class Runner:
    """One agent subprocess at a time, with its output fanned out to browsers."""

    def __init__(self):
        self.proc = None
        self.run_id = None
        self.run_dir = None
        self.dataset = 'pure'
        self.started = None
        self.stopped_reason = None
        self.lines = []
        self.subscribers = []
        self.lock = threading.Lock()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        return {
            'running': self.alive(),
            'run_id': self.run_id,
            'dataset': self.dataset,
            'started': self.started,
            'stopped_reason': self.stopped_reason,
            'exit_code': (self.proc.poll() if self.proc else None),
            'lines': self.lines[-200:],
        }

    def publish(self, obj):
        with self.lock:
            dead = []
            for q in self.subscribers:
                try:
                    q.put_nowait(obj)
                except Exception:
                    dead.append(q)
            for q in dead:
                self.subscribers.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def start(self, opts):
        if self.alive():
            return {'ok': False, 'error': 'a run is already in progress'}
        dataset = opts.get('dataset', 'pure')
        if dataset not in LOG_ROOTS:
            return {'ok': False, 'error': 'unknown dataset'}
        root = LOG_ROOTS[dataset]
        name = opts.get('run_name') or 'web-run'
        if not re.fullmatch(r'[\w-]{1,60}', name):
            return {'ok': False, 'error': 'run name must be letters, digits, - and _'}
        base = os.path.join(ROOT, root)
        os.makedirs(base, exist_ok=True)
        n = 1
        while os.path.exists(os.path.join(base, '%s-%d' % (name, n))):
            n += 1
        run_id_name = '%s-%d' % (name, n)

        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        # The recorded crash in logs/void-run-1 was a cp1252 console choking on
        # an arrow character. A run launched from here gets a UTF-8 stdout so
        # that failure mode cannot come back through this path.
        for key, opt, cast in (
            ('AGENT_MODEL', 'model', str),
            ('AGENT_MAX_EXPERIMENTS', 'max_experiments', int),
            ('AGENT_MAX_WALL_SECONDS', 'max_wall_seconds', float),
            ('AGENT_MAX_COST_USD', 'max_cost_usd', float),
        ):
            if opts.get(opt) not in (None, ''):
                try:
                    env[key] = str(cast(opts[opt]))
                except (TypeError, ValueError):
                    return {'ok': False, 'error': 'bad value for %s' % opt}

        try:
            max_iter = int(opts.get('max_iter') or 3)
        except (TypeError, ValueError):
            return {'ok': False, 'error': 'max_iter must be a number'}
        max_iter = max(1, min(max_iter, 200))

        cmd = [sys.executable, '-m', 'agent',
               '--run-id', run_id_name,
               '--dataset', dataset,
               '--max-iter', str(max_iter)]
        if opts.get('supervised'):
            cmd.append('--supervised')

        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Binary pipe on purpose. The loop draws a carriage-return spinner,
        # and Python's universal-newline text mode treats a bare CR as a line
        # ending - every animation frame would arrive as its own console
        # line. Reading bytes and splitting only on LF lets _emit collapse
        # each redraw to the frame the terminal would actually be showing.
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
            creationflags=creationflags)
        self.run_id = '%s/%s' % (root, run_id_name)
        self.run_dir = os.path.join(base, run_id_name)
        self.dataset = dataset
        self.started = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.stopped_reason = None
        self.lines = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._watch_files, daemon=True).start()
        return {'ok': True, 'run_id': self.run_id, 'cmd': ' '.join(cmd),
                'max_iter': max_iter}
    def _pump(self):
        proc = self.proc
        buf = b''
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            if chunk == b'\n':
                self._emit(buf)
                buf = b''
            else:
                buf += chunk
                if len(buf) > 16384:   # this long means a runaway redraw
                    self._emit(buf)
                    buf = b''
        if buf:
            self._emit(buf)
        code = proc.wait()
        self.publish({'type': 'exit', 'code': code})

    def _emit(self, raw):
        """One console line, with any spinner redraw collapsed to its last frame.

        Keep only what survives the final carriage return - that is what the
        terminal would actually be showing.
        """
        # rstrip the CR of a Windows CRLF first, or the split below
        # returns the empty string after it and every line vanishes.
        line = raw.rstrip(b'\r').split(b'\r')[-1]
        line = line.decode('utf-8', 'replace').rstrip()
        if not line:
            return
        self.lines.append(line)
        if len(self.lines) > 5000:
            del self.lines[:1000]
        self.publish({'type': 'stdout', 'line': line})


    def _watch_files(self):
        """Tail the run's own records, so the page shows what the ledger shows.

        The console renders from events.jsonl and NNNN.json - the same files a
        judge would read - rather than from parsed stdout. If they disagree,
        the files win, because the files are the deliverable.
        """
        seen_iters = set()
        ev_offset = 0
        while True:
            alive = self.alive()
            run_dir = self.run_dir
            if run_dir and os.path.isdir(run_dir):
                ev = os.path.join(run_dir, 'events.jsonl')
                if os.path.isfile(ev):
                    try:
                        with io.open(ev, encoding='utf-8') as f:
                            f.seek(ev_offset)
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    self.publish({'type': 'event',
                                                  'event': json.loads(line)})
                                except Exception:
                                    pass
                            ev_offset = f.tell()
                    except Exception:
                        pass
                for fn in _iter_files(run_dir):
                    if fn in seen_iters:
                        continue
                    rec = _read_json(os.path.join(run_dir, fn))
                    if rec:
                        seen_iters.add(fn)
                        self.publish({'type': 'iteration', 'record': rec})
            if not alive:
                break
            time.sleep(1.0)

    def stop(self):
        if not self.alive():
            return {'ok': False, 'error': 'nothing running'}
        self.stopped_reason = 'stopped from the console'
        pid = self.proc.pid
        try:
            if os.name == 'nt':
                # The agent spawns a child python per experiment; /T takes the
                # tree, otherwise a 900s training run outlives the stop button.
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                               capture_output=True)
            else:
                self.proc.terminate()
        except Exception as e:
            return {'ok': False, 'error': str(e)}
        return {'ok': True}


RUNNER = Runner()


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'agent-console'

    def log_message(self, fmt, *args):
        pass

    # -- helpers ----------------------------------------------------------
    def _send(self, code, body, ctype='application/json; charset=utf-8',
              extra=None):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str))

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8'))
        except Exception:
            return {}

    def _static(self, path):
        rel = path.lstrip('/') or 'index.html'
        full = os.path.normpath(os.path.join(WEB, rel))
        if not full.startswith(WEB) or not os.path.isfile(full):
            return self._send(404, 'not found', 'text/plain; charset=utf-8')
        ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
        if ctype.startswith('text/') or ctype.endswith('javascript'):
            ctype += '; charset=utf-8'
        with open(full, 'rb') as f:
            self._send(200, f.read(), ctype)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path

        if p == '/api/runs':
            return self._json({'runs': list_runs()})
        if p == '/api/run':
            data = read_run((q.get('id') or [''])[0])
            return self._json(data) if data else self._json(
                {'error': 'no such run'}, 404)
        if p == '/api/solution':
            data = read_solution((q.get('id') or [''])[0],
                                 (q.get('iteration') or ['0'])[0])
            return self._json(data) if data else self._json(
                {'error': 'no source on disk'}, 404)
        if p == '/api/prompt':
            return self._json(prompt_state())
        if p == '/api/preflight':
            ds = (q.get('dataset') or ['pure'])[0]
            try:
                return self._json(preflight(ds))
            except Exception as e:
                return self._json({'error': str(e)}, 500)
        if p == '/api/status':
            return self._json(RUNNER.status())
        if p == '/api/stream':
            return self._stream()
        if p.startswith('/api/'):
            return self._json({'error': 'unknown endpoint'}, 404)
        return self._static(p)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        body = self._body()
        if u.path == '/api/prompt':
            if body.get('reset'):
                return self._json(reset_prompt())
            text = body.get('text')
            if not isinstance(text, str) or not text.strip():
                return self._json({'ok': False, 'error': 'empty prompt'}, 400)
            return self._json(save_prompt(text))
        if u.path == '/api/run':
            return self._json(RUNNER.start(body))
        if u.path == '/api/stop':
            return self._json(RUNNER.stop())
        return self._json({'error': 'unknown endpoint'}, 404)

    def _stream(self):
        q = RUNNER.subscribe()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        try:
            self.wfile.write(b': connected\n\n')
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                    payload = json.dumps(msg, default=str)
                    self.wfile.write(('data: %s\n\n' % payload).encode('utf-8'))
                except queue.Empty:
                    self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
        except Exception:
            pass
        finally:
            RUNNER.unsubscribe(q)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    httpd = Server(('127.0.0.1', PORT), Handler)
    url = 'http://127.0.0.1:%d' % PORT
    print('agent console  ->  %s' % url)
    print('serving records from %s' % ROOT)
    print('Ctrl-C to stop')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
