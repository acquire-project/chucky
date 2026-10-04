# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "zarr>=3",
#   "numcodecs",
#   "tensorstore>=0.1.76",
#   "ome-zarr-models>=1.6",
# ]
# ///
"""Validate every published NGFF extent while a CPU/GPU writer streams to S3.

Uses AWS_ENDPOINT_URL (default http://localhost:9000) and a unique test bucket.
The writer uses a completion gate; the reader accesses S3 directly. No writer
flush, close, or sink drain is allowed until all streaming checkpoints pass.
"""

import argparse
import json
import os
import subprocess
import time
import uuid
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from s3_readback_proxy import S3ReadbackProxy
from validate_ngff_readback import (
    expect_rejected,
    expected_level,
    require,
    validate_ngff_metadata,
)
from validate_zarr import validate_tensorstore


class Writer:
    def __init__(self, command):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.lines = Queue()

        def collect():
            for line in self.process.stdout:
                self.lines.put(line.rstrip())
            self.lines.put(None)

        self.thread = Thread(target=collect, daemon=True)
        self.thread.start()

    def line(self):
        try:
            line = self.lines.get(timeout=30)
        except Empty:
            raise TimeoutError("Writer checkpoint timed out") from None
        require(line is not None, f"Writer exited early: {self.process.poll()}")
        return line

    def send(self, frames):
        self.process.stdin.write(f"{frames}\n")
        self.process.stdin.flush()

    def appended(self, frames):
        require(self.line() == f"appended {frames}", "Incorrect writer checkpoint")

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.thread.join(timeout=5)
        self.process.stdout.close()


class Reader:
    def __init__(self, s3, endpoint, bucket, prefix, shape):
        self.s3, self.bucket, self.prefix = s3, bucket, prefix
        self.kvstore = {
            "driver": "s3",
            "bucket": bucket,
            "endpoint": endpoint,
            "aws_region": "us-east-1",
            "aws_credentials": {"type": "environment"},
            "s3_request_retries": {
                "max_retries": 1,
                "initial_delay": "10ms",
                "max_delay": "100ms",
            },
        }
        self.reference = [expected_level(level, shape) for level in range(3)]
        self.previous = [0, 0, 0]
        self.paths = None

    def get(self, key):
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as body:
            return body.read()

    def array_store(self, path):
        return {**self.kvstore, "path": f"{self.prefix}/pyramid/{path}/"}

    def snapshot(self, maximum):
        metadata = json.loads(self.get(f"{self.prefix}/pyramid/zarr.json"))
        datasets = validate_ngff_metadata(metadata)
        paths = [dataset["path"] for dataset in datasets]
        if self.paths is not None:
            require(paths == self.paths, "Dataset paths changed while streaming")
        self.paths = paths
        sizes = []
        for level, path in enumerate(paths):
            metadata = json.loads(self.get(f"{self.prefix}/pyramid/{path}/zarr.json"))
            require(
                metadata["dimension_names"] == ["t", "y", "x"],
                f"Level {level}: incorrect array axis names",
            )
            shape = validate_tensorstore(
                self.array_store(path),
                self.reference[level],
                allow_prefix=True,
            )
            require(
                self.previous[level] <= shape[0] <= maximum[level],
                f"Level {level}: unexpected published extent {shape[0]}, "
                f"previous {self.previous[level]}, maximum {maximum[level]}",
            )
            sizes.append(shape[0])
        self.previous = sizes
        return sizes

    def wait_for(self, frames):
        deadline = time.monotonic() + 30
        while True:
            # Every observed extent is read immediately. An invalid read fails;
            # only valid, lagging metadata is polled again.
            sizes = self.snapshot([frames] * 3)
            if sizes == [frames] * 3:
                return
            require(time.monotonic() < deadline, f"Publication stalled at {sizes}")
            time.sleep(0.01)

    def reject_early_metadata(self, frames):
        for level, path in enumerate(self.paths):
            key = f"{self.prefix}/pyramid/{path}/zarr.json"
            original = self.get(key)
            metadata = json.loads(original)
            metadata["shape"][0] = frames
            try:
                self.s3.put_object(
                    Bucket=self.bucket, Key=key, Body=json.dumps(metadata)
                )
                expect_rejected(
                    f"S3 level {level} metadata ahead of its shards",
                    partial(
                        validate_tensorstore,
                        self.array_store(path),
                        self.reference[level],
                        allow_prefix=True,
                    ),
                )
            finally:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=original)

    def reject_partial_shard_damage(self, shard_index):
        for level, path in enumerate(self.paths):
            key = f"{self.prefix}/pyramid/{path}/c/{shard_index}/0/0"
            original = self.get(key)
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
                expect_rejected(
                    f"missing S3 level {level} final partial shard",
                    partial(
                        validate_tensorstore,
                        self.array_store(path),
                        self.reference[level],
                    ),
                )
                for offset, label in ((0, "payload"), (-1, "index checksum")):
                    corrupt = bytearray(original)
                    corrupt[offset] ^= 1
                    self.s3.put_object(Bucket=self.bucket, Key=key, Body=bytes(corrupt))
                    expect_rejected(
                        f"corrupt S3 level {level} {label}",
                        partial(
                            validate_tensorstore,
                            self.array_store(path),
                            self.reference[level],
                        ),
                    )
            finally:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=original)


def run_case(executable, s3, endpoint, bucket, proxy, codec, buffered, multipart=False):
    prefix = f"{codec}-{buffered}{'-multipart' if multipart else ''}"
    writer = Writer(
        [
            str(executable),
            proxy.endpoint,
            bucket,
            prefix,
            codec,
            str(buffered),
            str(int(multipart)),
        ]
    )
    try:
        fixture = json.loads(writer.line())
        nt, ny, nx = (fixture[key] for key in ("nt", "ny", "nx"))
        shard_frames = fixture["shard_frames"]
        reader = Reader(s3, endpoint, bucket, prefix, (nt, ny, nx))
        reader.wait_for(0)
        writer.send(1)
        writer.appended(1)
        reader.wait_for(0)
        for shard_index in range(2):
            complete = (shard_index + 1) * shard_frames
            held_key = f"{prefix}/pyramid/{reader.paths[0]}/c/{shard_index}/0/0"
            proxy.arm(f"{bucket}/{held_key}")
            writer.send(complete + 1)
            require(
                proxy.entered.wait(30), "Writer did not reach the held shard upload"
            )
            # The held object is absent; previously published shards stay readable.
            try:
                s3.head_object(Bucket=bucket, Key=held_key)
            except ClientError as error:
                require(
                    error.response["ResponseMetadata"]["HTTPStatusCode"] == 404,
                    f"Unexpected HEAD failure: {error}",
                )
            else:
                raise AssertionError("Held shard became visible before completion")
            deadline = time.monotonic() + 0.2
            while True:
                reader.snapshot([complete - shard_frames, complete, complete])
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            proxy.release.set()
            writer.appended(complete + 1)
            reader.wait_for(complete)
            if codec == "none" and buffered == 0 and not multipart and shard_index == 0:
                reader.reject_early_metadata(complete + 1)
        if buffered == 2:
            writer.send(nt + 1)
            writer.appended(nt + 1)
        # All earlier checkpoints deliberately precede flush and close.
        writer.send(0)
        require(writer.process.wait(timeout=30) == 0, "Writer close failed")
        reader.wait_for(nt)
        if multipart:
            for shard_index in range(2):
                key = f"{bucket}/{prefix}/pyramid/{reader.paths[0]}/c/{shard_index}/0/0"
                require(
                    len(proxy.parts.get(key, ())) >= 2,
                    "Multipart upload was not exercised",
                )
        elif codec == "none" and buffered == 0:
            reader.reject_partial_shard_damage(2)
            reader.wait_for(nt)
        require(not proxy.errors, f"S3 proxy failed: {proxy.errors}")
        print(
            f"PASS {prefix}: streaming extents 0/{shard_frames}/{2 * shard_frames}, "
            f"final {nt}; all three levels",
            flush=True,
        )
    finally:
        proxy.release.set()
        writer.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    args = parser.parse_args()
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        config=Config(
            connect_timeout=5,
            read_timeout=15,
            retries={"max_attempts": 0},
            s3={"addressing_style": "path"},
        ),
    )
    bucket = "chucky-ngff-readback-" + uuid.uuid4().hex
    s3.create_bucket(Bucket=bucket)
    proxy = S3ReadbackProxy(endpoint)
    try:
        codecs = subprocess.check_output(
            [str(args.executable), "--list"],
            text=True,
            timeout=30,
        ).splitlines()
        require(len(codecs) == 10 and len(set(codecs)) == 10, "Incomplete codec matrix")
        for buffered in range(3):
            for codec in codecs:
                run_case(args.executable, s3, endpoint, bucket, proxy, codec, buffered)
        run_case(
            args.executable, s3, endpoint, bucket, proxy, "none", 0, multipart=True
        )
    finally:
        proxy.close()
        # The unique bucket isolates parallel CPU/GPU tests and existing S3 tests.
        uploads = s3.list_multipart_uploads(Bucket=bucket).get("Uploads", [])
        for upload in uploads:
            s3.abort_multipart_upload(
                Bucket=bucket, Key=upload["Key"], UploadId=upload["UploadId"]
            )
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        s3.delete_bucket(Bucket=bucket)


if __name__ == "__main__":
    main()
