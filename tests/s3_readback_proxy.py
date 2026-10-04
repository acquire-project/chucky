"""Delay selected S3Mock shard completions while the reader checks S3 directly."""

import asyncio
from contextlib import asynccontextmanager

from aiohttp import ClientSession, ClientTimeout, web


class S3ReadbackProxy:
    def __init__(self, upstream):
        self.upstream = upstream.rstrip("/")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.key = None
        self.parts = {}

    def arm(self, key):
        self.key = key
        self.entered.clear()
        self.release.clear()

    async def forward(self, request):
        # aiohttp handles HTTP framing; retain the AWS payload and headers.
        body = await request.read()
        key = request.path.lstrip("/")
        complete = (
            request.method == "PUT"
            and "partNumber" not in request.query
            or request.method == "POST"
            and "uploadId" in request.query
        )
        if complete and key == self.key:
            self.entered.set()
            await asyncio.wait_for(self.release.wait(), timeout=30)
        headers = request.headers.copy()
        headers.popall("Transfer-Encoding", None)
        headers.popall("Content-Length", None)
        async with self.client.request(
            request.method,
            self.upstream + request.raw_path,
            data=body,
            headers=headers,
            allow_redirects=False,
        ) as response:
            data = await response.read()
            if response.status < 300 and "partNumber" in request.query:
                self.parts.setdefault(key, set()).add(request.query["partNumber"])
            headers = response.headers.copy()
            headers.popall("Transfer-Encoding", None)
            if request.method != "HEAD":
                headers.popall("Content-Length", None)
            return web.Response(status=response.status, headers=headers, body=data)

    @asynccontextmanager
    async def run(self):
        app = web.Application(client_max_size=0)
        app.router.add_route("*", "/{path:.*}", self.forward)
        runner = web.AppRunner(app, shutdown_timeout=30)
        async with ClientSession(
            timeout=ClientTimeout(total=30), auto_decompress=False
        ) as self.client:
            try:
                await runner.setup()
                await web.TCPSite(runner, "127.0.0.1", 0).start()
                self.endpoint = f"http://127.0.0.1:{runner.addresses[0][1]}"
                yield self
            finally:
                self.release.set()
                await runner.cleanup()
