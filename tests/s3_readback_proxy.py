"""Forward S3 requests while holding selected shard completions for readback."""

import http.client
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, unquote, urlsplit


class S3ReadbackProxy:
    def __init__(self, endpoint: str):
        upstream = urlsplit(endpoint)
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.key = None
        self.parts = {}
        self.errors = []
        self.lock = Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def body(self):
                if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
                    return self.rfile.read(int(self.headers.get("Content-Length", 0)))
                # Preserve wire chunks, including signed AWS chunks and trailers.
                chunks = bytearray()
                while True:
                    line = self.rfile.readline()
                    if not line:
                        raise EOFError("Truncated chunked request")
                    chunks.extend(line)
                    size = int(line.split(b";", 1)[0], 16)
                    if size:
                        chunk = self.rfile.read(size + 2)
                        if len(chunk) != size + 2:
                            raise EOFError("Truncated request chunk")
                        chunks.extend(chunk)
                    else:
                        while True:
                            line = self.rfile.readline()
                            if not line:
                                raise EOFError("Truncated request trailers")
                            chunks.extend(line)
                            if line == b"\r\n":
                                return bytes(chunks)

            def forward(self):
                connection = None
                try:
                    self.connection.settimeout(30)
                    body = self.body()
                    request = urlsplit(self.path)
                    key = unquote(request.path).lstrip("/")
                    query = parse_qs(request.query, keep_blank_values=True)
                    complete = (
                        self.command == "PUT"
                        and "partNumber" not in query
                        or self.command == "POST"
                        and "uploadId" in query
                    )
                    if complete and key == proxy.key:
                        proxy.entered.set()
                        if not proxy.release.wait(30):
                            raise TimeoutError(f"Reader did not release {key}")
                    cls = (
                        http.client.HTTPSConnection
                        if upstream.scheme == "https"
                        else http.client.HTTPConnection
                    )
                    connection = cls(upstream.hostname, upstream.port, timeout=30)
                    headers = dict(self.headers)
                    headers["Connection"] = "close"
                    connection.request(
                        self.command,
                        upstream.path.rstrip("/") + self.path,
                        body=body,
                        headers=headers,
                    )
                    response = connection.getresponse()
                    data = response.read()
                    if response.status < 300 and "partNumber" in query:
                        with proxy.lock:
                            proxy.parts.setdefault(key, set()).add(
                                query["partNumber"][0]
                            )
                    self.send_response(response.status)
                    for name, value in response.getheaders():
                        if name.lower() not in (
                            "connection",
                            "transfer-encoding",
                            "content-length",
                        ):
                            self.send_header(name, value)
                    length = (
                        response.getheader("Content-Length", "0")
                        if self.command == "HEAD"
                        else str(len(data))
                    )
                    self.send_header("Content-Length", length)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                except (
                    OSError,
                    EOFError,
                    ValueError,
                    http.client.HTTPException,
                ) as error:
                    with proxy.lock:
                        proxy.errors.append(str(error))
                    with suppress(OSError):
                        self.send_error(502, str(error))
                finally:
                    if connection:
                        connection.close()
                    self.close_connection = True

            do_GET = do_HEAD = do_PUT = do_POST = do_DELETE = forward

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Join in-flight requests before the test removes its bucket.
        self.server.daemon_threads = False
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def arm(self, key: str):
        self.key = key
        self.entered.clear()
        self.release.clear()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
