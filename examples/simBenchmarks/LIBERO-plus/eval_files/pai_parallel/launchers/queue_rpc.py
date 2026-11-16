"""Local evaluation queue with idempotent RPC and short in-memory lock sections."""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import time
import urllib.error
import urllib.request


class ResultNotReady(Exception):
    """Committed artifacts are not yet visible to the controller."""


class ResultCoordinator:
    def __init__(self, queue, rows, read_result):
        self.queue = queue
        self.rows = rows
        self.read_result = read_result

    def snapshot(self):
        with self.queue.lock:
            return dict(self.rows), self.queue.snapshot()

    def complete(self, worker, key):
        with self.queue.lock:
            if key not in self.queue.jobs:
                raise ValueError('Unknown result')
            if key in self.queue.done:
                return {'ok': True}
            if self.queue.active.get(worker) != key:
                raise ValueError('Wrong result owner')
            job = self.queue.jobs[key]
        # NAS metadata reads can stall. Never hold the global queue lock here.
        row = self.read_result(job)
        if row is None:
            raise ResultNotReady('Incomplete episode artifacts')
        with self.queue.lock:
            if key in self.queue.done:
                return {'ok': True}
            if self.queue.active.get(worker) != key:
                raise ValueError('Wrong result owner')
            self.rows[key] = row
            self.queue.finish(worker, key)
        return {'ok': True}

    def dispatch(self, path, body):
        worker = body['worker']
        if path == '/claim':
            return self.queue.claim(worker, body.get('last_suite'))
        if path == '/complete':
            return self.complete(worker, body['key'])
        if path == '/error':
            with self.queue.lock:
                if self.queue.active.get(worker) != body['key']:
                    raise ValueError('Wrong failed job owner')
                # Python episode errors remain fatal; only infrastructure exits
                # are eligible for bounded worker recovery.
                self.queue.fail(worker, body['error'], retry=False)
            return {'ok': True}
        raise ValueError('Unknown operation')

    def recover_worker(self, worker, error):
        with self.queue.lock:
            key = self.queue.active.get(worker)
        if key is None:
            return {'key': None, 'recovered': False, 'requeued': False}
        try:
            self.complete(worker, key)
            return {'key': key, 'recovered': True, 'requeued': False}
        except ResultNotReady:
            with self.queue.lock:
                if key in self.queue.done:
                    return {'key': key, 'recovered': True, 'requeued': False}
                if self.queue.active.get(worker) == key:
                    self.queue.fail(worker, error)
                return {'key': key, 'recovered': False,
                        'requeued': key not in self.queue.failed}


def make_queue_server(address, coordinator, token, valid_workers):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.headers.get('Authorization') != token:
                    raise ValueError('Wrong queue token')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1000000:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if body['worker'] not in valid_workers:
                    raise ValueError('Unknown worker')
                answer = coordinator.dispatch(self.path, body)
                status = 200
            except (ResultNotReady, OSError) as exc:
                status, answer = 503, {'error': str(exc)}
            except (ValueError, KeyError, TypeError) as exc:
                status, answer = 400, {'error': str(exc)}
            except Exception:
                logging.exception('Queue request failed')
                status, answer = 500, {'error': 'Queue request failed'}
            data = json.dumps(answer).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                # A timed-out client may retry an already committed operation.
                logging.debug('Queue response receiver disconnected')

    class QueueServer(ThreadingHTTPServer):
        request_queue_size = 256
        daemon_threads = True

    return QueueServer(address, Handler)


class QueueRPC:
    def __init__(self, url, worker, token, *, timeout=120, max_attempts=4,
                 backoff=1, opener=None):
        if timeout <= 0 or max_attempts < 1 or backoff < 0:
            raise ValueError('Invalid queue RPC retry policy')
        self.url, self.worker, self.token = url, worker, token
        self.timeout, self.max_attempts, self.backoff = timeout, max_attempts, backoff
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, path, payload):
        data = json.dumps(dict(payload, worker=self.worker)).encode()
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(self.url + path, data=data,
                headers={'Content-Type': 'application/json', 'Authorization': self.token})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (408, 429, 500, 502, 503, 504):
                    raise
                if attempt == self.max_attempts:
                    raise
                failure = 'HTTP ' + str(status)
            except (OSError, urllib.error.URLError, http.client.HTTPException) as exc:
                if attempt == self.max_attempts:
                    raise
                failure = type(exc).__name__
            logging.warning('Queue %s retry %s/%s after %s',
                            path, attempt, self.max_attempts, failure)
            time.sleep(min(10, self.backoff * 2 ** (attempt - 1)))
