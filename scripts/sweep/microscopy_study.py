# /// script
# requires-python = ">=3.11"
# dependencies = ["click", "rich", "pydantic"]
# ///
"""Prepare and run a retained microscopy experiment, separate from regular sweeps."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import time

import sweep
from microscopy_data import check_machine, check_observation, estimate_seconds, validate_study
from microscopy_plan import DEFAULT_DEFINITION, fingerprint, make_plan, read_json, validate_plan


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def execute_plan(document, measure, checkpoint, max_seconds, *, clock=time.monotonic):
    validate_study(document, complete=False)
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("The process budget must be finite and positive")
    plan = document["plan"]
    deadline = clock() + max_seconds
    document["status"] = "running"
    checkpoint(document)
    for task in plan["schedule"][len(document["records"]):]:
        remaining = deadline - clock()
        if remaining < 1:
            document["status"] = "budget-exhausted"
            checkpoint(document)
            return False
        record = {**task, "started": utc_now()}
        try:
            result = measure(plan["cases"][task["case_id"]], task, min(600, remaining))
            record["result"] = result
            check_observation(record, task, plan["cases"][task["case_id"]], plan["definition"])
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            record.setdefault("result", {"status": "error", "error": str(error)})
            record["error"] = str(error)
            document["records"].append(record)
            document["status"] = "failed"
            checkpoint(document)
            raise ValueError(f"{task['id']} failed; observations are retained: {error}") from error
        document["records"].append(record)
        checkpoint(document)
    document["status"] = "complete"
    document["finished"] = utc_now()
    validate_study(document)
    checkpoint(document)
    return True


def prepare_document(plan, build_dir, build_path, registry, corpus_path, machine_name, study_id):
    executable = sweep.image_executable(build_dir)
    build = sweep.image_build_record(executable)
    expected = read_json(build_path)
    for key in ("revision", "source_tree_sha256", "executable_sha256", "cmake_cache_sha256"):
        if not build.get(key) or build[key] != expected.get(key):
            raise ValueError(f"Build changed ({key}); rebuild and record it before measuring")
    if build.get("worktree_status"):
        raise ValueError("Commit the study code before running a retained experiment")
    for spec in plan["cases"].values():
        sweep.RunSpec(**spec)
    definition = plan["definition"]
    corpus = sweep.load_image_corpus(registry, definition["dataset"], corpus_path)
    members = [(pack["id"], pack["input_id"], pack["dtype"].removesuffix("le"))
               for pack in corpus.manifest["packs"]]
    if make_plan(definition, members, plan["phase"]) != plan:
        raise ValueError("Verified corpus differs from the planned input selection")
    _, image_runner = sweep._image_modules()
    machine = image_runner.machine_record(machine_name, "gpu" in definition["backends"])
    machine["logical_cpu_count"] = machine["cpu_count"]
    machine["physical_cpu_count"] = machine.get("cpu_topology", {}).get("physical_cores")
    machine["cpu_count"] = sweep.cpu_count()
    gpu, driver = sweep.gpu_and_driver() if "gpu" in definition["backends"] else (None, None)
    machine.update(gpu=gpu, driver=driver)
    check_machine(machine, definition)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", study_id):
        raise ValueError("Study id must contain lowercase letters, digits, and hyphens")
    return ({"version": 1, "benchmark": "microscopy-study", "id": study_id,
             "status": "running", "created": utc_now(), "machine": machine,
             "build": build, "corpus": sweep.image_corpus_record(corpus),
             "plan": plan, "plan_sha256": fingerprint(plan), "records": []}, corpus)


def resume_document(existing, prepared):
    validate_study(existing, complete=False)
    for key in ("id", "machine", "build", "plan_sha256", "plan"):
        if existing[key] != prepared[key]:
            raise ValueError(f"Resume changed {key}; use a new output directory")
    recorded = {key: value for key, value in existing["corpus"].items() if key != "verify_s"}
    verified = {key: value for key, value in prepared["corpus"].items() if key != "verify_s"}
    if recorded != verified:
        raise ValueError("Resume changed corpus; use a new output directory")
    return existing


def export_document(document, storage):
    validate_study(document)
    if not storage.strip() or any(character in storage for character in ("/", "\\", "\n", "\r")):
        raise ValueError("Describe storage without filesystem paths")
    host_path = re.compile(r"(?:^|[\s\"'[(=;,:])(?:[A-Za-z]:[\\/]|\\\\|/(?![/\s*])|~[\\/]|file://)")
    flag_path = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/[^\s/]+/)")

    def absolute(value):
        return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()

    def clean(value, field=""):
        if isinstance(value, dict):
            result, counts = {}, {}
            for key, item in value.items():
                name = key
                if absolute(key):
                    basename = PureWindowsPath(key).name
                    counts[basename] = counts.get(basename, 0) + 1
                    name = f"{basename} ({counts[basename]})"
                if name in result:
                    raise ValueError("Public metadata names collide after removing paths")
                result[name] = clean(item, key)
            return result
        if isinstance(value, list):
            return [clean(item, field) for item in value]
        if isinstance(value, str):
            pattern = flag_path if "FLAGS" in field else host_path
            return "\n".join("PATH_REMOVED" if pattern.search(line) else line
                             for line in value.split("\n"))
        return value

    exported = clean(document)
    exported.setdefault("storage", {})["description"] = storage.strip()
    validate_study(exported)
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="Expand configurations and execution order without loading image bytes")
    plan_parser.add_argument("--definition", type=Path, default=DEFAULT_DEFINITION)
    plan_parser.add_argument("--phase", choices=("pilot", "discovery", "comparison"))
    plan_parser.add_argument("--backend", action="append", choices=("cpu", "gpu"), help="Select measured backends")
    plan_parser.add_argument("--sink", action="append", choices=("discard", "fs"), help="Select measured sinks")
    plan_parser.add_argument("--cpu-count", type=int, help="Require this allowed CPU count when running")
    plan_parser.add_argument("--cpu-workers", action="append", type=int,
                             help="CPU compression threads; repeat to compare counts in a selected study")
    plan_parser.add_argument("--gpu", help="Require this GPU name when running")
    plan_parser.add_argument("--data-registry", type=Path, default=sweep.DEFAULT_DATA_REGISTRY)
    plan_parser.add_argument("--corpus", type=Path)
    plan_parser.add_argument("--estimate-from", type=Path, help="Completed pilot study.json")
    plan_parser.add_argument("--output", type=Path, required=True)
    build_parser = commands.add_parser("record-build", help="Record a successfully completed build before measurement")
    build_parser.add_argument("--build-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run", help="Execute a prepared study on the defined machine")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--build-dir", type=Path, required=True)
    run_parser.add_argument("--build-record", type=Path, required=True)
    run_parser.add_argument("--data-registry", type=Path, default=sweep.DEFAULT_DATA_REGISTRY)
    run_parser.add_argument("--corpus", type=Path)
    run_parser.add_argument("--machine", required=True)
    run_parser.add_argument("--id", required=True, help="Unique archive id for this phase and session")
    run_parser.add_argument("--output", type=Path, required=True, help="Directory for study.json checkpoints")
    run_parser.add_argument("--max-seconds", type=float, required=True, help="Process budget; corpus/build verification is extra")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--tmpdir", type=Path)
    for key in ("bucket", "region", "endpoint"):
        run_parser.add_argument(f"--s3-{key}")
    export_parser = commands.add_parser("export", help="Write a public copy without host filesystem paths")
    export_parser.add_argument("--study", type=Path, required=True)
    export_parser.add_argument("--storage", required=True, help="Path-free storage description, including local or network type")
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            if args.output.exists():
                raise ValueError("Export output exists; choose a new public-copy filename")
            document = export_document(read_json(args.study), args.storage)
            write_json(args.output, document)
            return
        if args.command == "record-build":
            write_json(args.output, sweep.image_build_record(sweep.image_executable(args.build_dir.resolve())))
            return
        if args.command == "plan":
            definition = read_json(args.definition)
            if args.backend:
                definition["backends"] = list(dict.fromkeys(args.backend))
                if definition["version"] == 1:
                    definition["profiles"] = {key: definition["profiles"][key] for key in definition["backends"]}
                else:
                    from microscopy_comparison import selected_settings
                    definition["inputs"] = [input_id for input_id in definition["inputs"]
                                            if any(selected_settings(definition, input_id, backend)
                                                   for backend in definition["backends"])]
                    for selected in (definition["configurations"], definition["reference"]["configurations"]):
                        for input_id in set(selected) - set(definition["inputs"]):
                            del selected[input_id]
            if args.cpu_workers is not None:
                if definition["version"] == 1:
                    raise ValueError("CPU worker comparisons require a selected comparison definition")
                definition.update(version=3, cpu_workers=args.cpu_workers)
                for selected in definition["configurations"].values():
                    for settings in selected:
                        settings.pop("cpu_workers", None)
            if args.sink:
                if definition["version"] in (2, 3):
                    definition["sinks"] = list(dict.fromkeys(args.sink))
                elif len(set(args.sink)) == 1:
                    definition["sink"] = args.sink[0]
                else:
                    raise ValueError("Multiple sinks require a selected comparison definition")
            if args.cpu_count is not None:
                definition["environment"]["cpu_count"] = args.cpu_count
            if args.gpu is not None:
                definition["environment"]["gpu"] = args.gpu
            members = sweep.load_image_members(args.data_registry, definition["dataset"], args.corpus)
            plan = make_plan(definition, members, args.phase)
            for spec in plan["cases"].values():
                sweep.RunSpec(**spec)
            write_json(args.output, plan)
            report = {"phase": plan["phase"], **plan["counts"],
                      "requested_seconds_floor": plan["requested_seconds"]}
            if args.estimate_from:
                report["estimate"] = estimate_seconds(read_json(args.estimate_from), plan)
            print(json.dumps(report, indent=2))
            return
        if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
            raise ValueError("--max-seconds must be finite and positive")
        plan = validate_plan(read_json(args.plan))
        if any(spec["sink"] == "fs" for spec in plan["cases"].values()):
            if args.tmpdir is None or not args.tmpdir.is_dir():
                raise ValueError("Filesystem studies require --tmpdir pointing to an existing storage directory")
        path = args.output / "study.json"
        if path.exists() and not args.resume:
            raise ValueError("Output exists; use --resume or a new output directory")
        if args.resume and not path.exists():
            raise ValueError("No study checkpoint to resume")
        document, corpus = prepare_document(plan, args.build_dir.resolve(), args.build_record,
                                            args.data_registry, args.corpus, args.machine, args.id)
        document["sink_options"] = {"tmpdir": str(args.tmpdir.resolve()) if args.tmpdir else None,
                                    "s3_bucket": args.s3_bucket, "s3_region": args.s3_region,
                                    "s3_endpoint": args.s3_endpoint}
        if args.resume:
            existing = read_json(path)
            if existing.get("sink_options") != document["sink_options"]:
                raise ValueError("Resume changed sink destination")
            document = resume_document(existing, document)
        layouts = sweep.existing_image_layouts([record["result"] for record in document["records"]])

        def measure(config, task, timeout):
            profile = task["profile"]
            result = sweep.run_image_one(
                sweep.RunSpec(**config), args.build_dir.resolve(), corpus,
                profile["min_gib"], 1, False, layouts, warmup=profile["warmup_s"],
                duration=profile["duration_s"], geometry_frames=plan["definition"]["geometry_frames"],
                tmpdir_root=args.tmpdir, s3_bucket=args.s3_bucket, s3_region=args.s3_region,
                s3_endpoint=args.s3_endpoint, timeout=timeout, record_command=True, calibration=True,
                max_attempts=1,
            )
            if result is None:
                raise ValueError("Image benchmark executable is missing")
            result.pop("repetitions", None)
            print(f"{task['id']} / {len(plan['schedule'])}: {config['input_id']} "
                  f"{config['backend']} {config['codec']} {config['chunk_label']} "
                  f"block={config['blosc_block_bytes']} workers={result['worker_threads']} "
                  f"{task['role']}", flush=True)
            return result

        complete = execute_plan(document, measure, lambda data: write_json(path, data), args.max_seconds)
        if not complete:
            raise SystemExit("Process budget reached; completed observations are retained")
        print(f"Complete: {len(document['records'])} observations in {path}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Microscopy study: {error}\n")


if __name__ == "__main__":
    main()
