#!/usr/bin/env python3
"""Check fixed-input ASTRA-sim/ns-3 result repeatability without modifying the original case."""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

FINISHED = re.compile(r"sys\[(\d+)\] finished,\s*(\d+) cycles")

def require(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return path

def digest(path):
    return hashlib.sha256(require(path).read_bytes()).hexdigest()

def cycle_count(log, n):
    found = [(int(r), int(c)) for r, c in FINISHED.findall(require(log).read_text(errors="replace"))]
    if len(found) != n or {r for r, _ in found} != set(range(n)):
        raise RuntimeError(f"Incomplete rank finishes in {log}: got {len(found)}, expected {n}")
    return max(c for _, c in found)

def flow_summary(path):
    rows, ends = [], []
    for line in require(path).read_text().splitlines():
        if not line.strip():
            continue
        w = line.split()
        if len(w) != 8:
            raise RuntimeError(f"Invalid ns-3 FCT row in {path}: {line}")
        try:
            int(w[0], 16); int(w[1], 16)
            nums = [int(x) for x in w[2:]]
        except ValueError as exc:
            raise RuntimeError(f"Non-numeric FCT row in {path}: {line}") from exc
        if nums[2] <= 0 or nums[3] < 0 or nums[4] <= 0:
            raise RuntimeError(f"Invalid FCT duration/size in {path}: {line}")
        ends.append(nums[3] + nums[4])
        rows.append(" ".join(w))
    if not rows:
        raise RuntimeError(f"No FCT flows in {path}")
    signature = hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()
    return {"flows": len(rows), "last_end": max(ends), "flow_sha256": signature}

def isolated_network(original, fct_path):
    text = require(original).read_text()
    rows, matches = [], 0
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == "FCT_OUTPUT_FILE":
            if len(fields) != 2:
                raise RuntimeError("Invalid FCT_OUTPUT_FILE configuration")
            matches += 1
            line = f"FCT_OUTPUT_FILE {fct_path}"
        rows.append(line)
    if matches != 1:
        raise RuntimeError(f"Expected one FCT_OUTPUT_FILE, got {matches}")
    return "\n".join(rows) + "\n"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="3x4")
    ap.add_argument("--runs", type=int, default=0, help="new validation runs; 0 = read-only static audit")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--base", default="/app/astra-sim/astra-sim/bai_multinode_remote_core")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", args.spec):
        ap.error("--spec should be like 3x4")
    if args.runs < 0 or args.runs > 3 or args.timeout <= 0:
        ap.error("--runs must be 0..3 and --timeout must be positive")
    k, m = (int(x) for x in args.spec.split("x"))
    n = k * m
    base = Path(args.base).resolve()
    astra = base.parent
    ns3 = astra / "extern/network_backend/ns-3"
    case = base / "perf_star_16MiB" / args.spec / "16777216B"
    out = Path(args.out or base / "r3_repeatability_20261014" / args.spec).resolve()
    if out == case or out in case.parents or case in out.parents:
        raise RuntimeError("--out may not overlap the original case")
    binfile = require(ns3 / "build/scratch/ns3.42-AstraSimNetwork-default")
    network = require(case / "network.txt")
    system = require(case / "system.json")
    logical = require(case / "logical.json")
    remote = require(astra / "examples/remote_memory/analytical/no_memory_expansion.json")
    workload = case / "workload/all_gather"
    inputs = [binfile, network, system, logical, remote]
    inputs += [require(f"{workload}.{r}.et") for r in range(n)]
    report = {"spec": args.spec, "original_case": str(case),
              "input_sha256": {str(x): digest(x) for x in inputs},
              "environment": {"NS_GLOBAL_VALUE": os.environ.get("NS_GLOBAL_VALUE", "<unset>")}}
    original = {"cycles": cycle_count(case / "run.log", n),
                **flow_summary(case / "fct.txt")}
    documented = json.loads(require(case / "result.json").read_text())
    if original["cycles"] != int(documented["cycles"]):
        raise RuntimeError("Original run.log does not match result.json cycles")
    if original["last_end"] != original["cycles"]:
        raise RuntimeError("Original FCT end time differs from completion cycles")
    if "fct_flows" in documented and original["flows"] != int(documented["fct_flows"]):
        raise RuntimeError("Original FCT count differs from result.json")
    report["original"] = original
    out.mkdir(parents=True, exist_ok=True)
    rng_pattern = re.compile(r"RngSeedManager|SetSeed\s*\(|SetRun\s*\(|NS_GLOBAL_VALUE|random_device|srand\s*\(|/dev/(?:u)?random")
    source_hits = []
    for directory in (ns3 / "scratch", astra / "astra-sim/network_frontend/ns3"):
        if not directory.is_dir():
            continue
        for source in directory.rglob("*"):
            if not source.is_file() or source.suffix not in (".h", ".cc", ".cpp", ".c", ".py"):
                continue
            if source.stat().st_size > 2_000_000:
                continue
            for number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
                if rng_pattern.search(line):
                    source_hits.append(f"{source}:{number}:{line.strip()}")
    hits_path = out / "r3_rng_source_hits.txt"
    hits_path.write_text("\n".join(source_hits) + ("\n" if source_hits else ""))
    report["seed_source_hits"] = {"count": len(source_hits), "file": str(hits_path)}
    print(f"SEED_SOURCE_SCAN: hits={len(source_hits)} file={hits_path}", flush=True)
    print(f"BASELINE {args.spec}: cycles={original['cycles']} flows={original['flows']}", flush=True)
    if args.runs == 0:
        print("AUDIT_ONLY: no simulator execution; inspect source for seed overrides", flush=True)
    report["validation_runs"] = []
    for index in range(1, args.runs + 1):
        job = out / f"repeat_{index:02d}"
        job.mkdir(parents=True, exist_ok=True)
        cfg = job / "network.txt"
        cfg.write_text(isolated_network(network, job / "fct.txt"))
        cmd = [str(binfile), f"--workload-configuration={workload}",
               f"--system-configuration={system}",
               f"--network-configuration={cfg}",
               f"--remote-memory-configuration={remote}",
               f"--logical-topology-configuration={logical}",
               "--comm-group-configuration=empty"]
        with (job / "run.log").open("w") as f:
            try:
                proc = subprocess.run(cmd, cwd=ns3 / "build/scratch", stdout=f,
                                      stderr=subprocess.STDOUT, timeout=args.timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"SIM_TIMEOUT for {job}") from exc
        if proc.returncode:
            raise RuntimeError(f"SIM_FAILED for {job} (exit {proc.returncode}); see run.log")
        value = {"cycles": cycle_count(job / "run.log", n), **flow_summary(job / "fct.txt")}
        value["cycles_match_original"] = value["cycles"] == original["cycles"]
        value["fct_matches_original"] = value["flow_sha256"] == original["flow_sha256"]
        report["validation_runs"].append(value)
        print(f"REPEAT {index}: cycles={value['cycles']} flows={value['flows']} "
              f"cycles_match={value['cycles_match_original']} "
              f"fct_match={value['fct_matches_original']}", flush=True)
    vals = [original["cycles"]] + [r["cycles"] for r in report["validation_runs"]]
    report["summary"] = {"samples_including_original": len(vals),
                         "all_cycles_equal": (len(set(vals)) == 1 if len(vals) > 1 else None),
                         "cycle_range": max(vals) - min(vals),
                         "population_std_cycles": statistics.pstdev(vals)}
    for input_path in inputs:
        if digest(input_path) != report["input_sha256"][str(input_path)]:
            raise RuntimeError(f"Frozen source input changed during check: {input_path}")
    (out / "r3_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"SUMMARY identical={report['summary']['all_cycles_equal']} "
          f"range={report['summary']['cycle_range']} "
          f"report={out / 'r3_report.json'}", flush=True)

if __name__ == "__main__":
    main()
