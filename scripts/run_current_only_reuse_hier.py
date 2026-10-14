#!/usr/bin/env python3
"""Current MultiNode-RemoteCore 2x8/12/16; reuse recorded Hier Ring."""
import argparse
from contextlib import ExitStack
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

SIZE = 16777216
KEY = "all-gather-implementation-custom"


def need(p):
    p = Path(p)
    if not p.is_file() or not p.stat().st_size:
        raise RuntimeError(f"Missing required input: {p}")
    return p


def sha(p):
    h = hashlib.sha256()
    with need(p).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_write(p, content):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != content:
        raise RuntimeError(
            f"Existing file changed: {p}; use a fresh OUT directory"
        )
    if not p.exists():
        p.write_text(content)


def execute(args, log, *, stdout=None, cwd=None,
            env=None, timeout=7200):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        err = stack.enter_context(log.open("wb"))
        out = err if stdout is None else stack.enter_context(
            Path(stdout).open("wb")
        )
        process = subprocess.run(
            [str(x) for x in args],
            stdout=out, stderr=err, cwd=cwd, env=env,
            timeout=timeout, check=False
        )
    if process.returncode:
        last = "\n".join(
            log.read_text(errors="replace").splitlines()[-30:]
        )
        raise RuntimeError(
            f"Command failed: {args[0]}, exit={process.returncode}\n"
            f"{last}\nLog: {log}"
        )


def audit_xml(xml, m):
    n = 2 * m
    root = ET.parse(xml).getroot()
    assert root.tag == "algo"
    assert root.get("coll") == "allgather"
    assert int(root.get("ngpus", "-1")) == n
    gpus = root.findall("gpu")
    assert {
        int(g.get("id")) for g in gpus
    } == set(range(n))

    moves = Counter()
    for g in gpus:
        s = int(g.get("id"))
        for tb in g.findall("tb"):
            d = int(tb.get("send", "-1"))
            if d < 0:
                continue
            assert d != s and 0 <= d < n
            for step in tb.findall("step"):
                if step.get("type") != "s":
                    continue
                off = int(step.get("srcoff"))
                dest = int(step.get("dstoff"))
                cnt = int(step.get("cnt", "1"))
                assert off == dest
                assert cnt > 0 and 0 <= off < n
                assert off + cnt <= n
                for c in range(off, off + cnt):
                    moves[s, d, c] += 1

    total = sum(moves.values())
    inter = sum(
        v for (s, d, _), v in moves.items()
        if s // m != d // m
    )
    pairs = {(d, c) for _, d, c in moves}
    expected = {
        (d, c) for d in range(n)
        for c in range(n) if c != d
    }
    assert total == n * (n - 1)
    assert len(moves) == total
    assert inter == n and pairs == expected
    return total, inter


def runtime(log, fct, n):
    matches = re.findall(
        r"sys\[(\d+)\] finished,\s*(\d+) cycles",
        need(log).read_text(errors="replace")
    )
    results = [(int(i), int(t)) for i, t in matches]
    assert len(results) == n
    assert {i for i, _ in results} == set(range(n))
    count = len([
        l for l in need(fct).read_text().splitlines()
        if l.strip()
    ])
    assert count == n * (n - 1), (
        f"Expected {n*(n-1)} FCT rows, got {count}"
    )
    return max(t for _, t in results), count


def clone_network(path, new_fct):
    lines = need(path).read_text().splitlines()
    output, hits = [], Counter()
    for line in lines:
        words = line.split()
        if words and not line.lstrip().startswith("#"):
            if words[0] == "FCT_OUTPUT_FILE":
                hits["fct"] += 1
                line = f"FCT_OUTPUT_FILE {new_fct}"
            if words[0] == "TOPOLOGY_FILE":
                hits["topo"] += 1
            if (
                words[0] == "ENABLE_TRACE"
                and words[1] != "0"
            ):
                raise RuntimeError(
                    "Old baseline has ENABLE_TRACE != 0"
                )
        output.append(line)
    assert hits == {"fct": 1, "topo": 1}
    return "\n".join(output) + "\n"


def validate_old_topology(net, ns3, n):
    entries = [
        line.split()
        for line in need(net).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    values = [
        item[1] for item in entries
        if item[0] == "TOPOLOGY_FILE" and len(item) == 2
    ]
    assert len(values) == 1
    filename = Path(values[0])
    if not filename.is_absolute():
        filename = (
            ns3 / "build/scratch" / filename
        ).resolve()

    rows = [
        line.split()
        for line in need(filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert list(map(int, rows[0])) == [n+2, 2, n+1]
    assert list(map(int, rows[1])) == [n, n+1]
    assert len(rows[2:]) == n+1

    hosts, inter = set(), 0
    for row in rows[2:]:
        src, dst = map(int, row[:2])
        endpoints = {src, dst}
        if endpoints == {n, n+1}:
            inter += 1
            assert row[2] == "100Gbps"
        else:
            host = next(
                (v for v in (src, dst) if v < n), None
            )
            assert host is not None
            assert endpoints == {
                host, n + host // (n // 2)
            }
            assert row[2] == "400Gbps"
            assert host not in hosts
            hosts.add(host)
    assert inter == 1 and hosts == set(range(n))
    return filename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpus", nargs="+", type=int,
        default=[8, 12, 16]
    )
    ap.add_argument("--run-sim", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()

    if (
        a.timeout <= 0
        or len(set(a.gpus)) != len(a.gpus)
        or any(m not in (8, 12, 16) for m in a.gpus)
    ):
        ap.error(
            "--gpus must list distinct values from 8,12,16"
        )

    astra = Path(os.getenv(
        "ASTRA", "/app/astra-sim/astra-sim"
    ))
    capi = Path(os.getenv(
        "CAPI", "/app/astra-sim/collectiveapi"
    ))
    ns3 = Path(os.getenv(
        "NS3",
        str(astra / "extern/network_backend/ns-3")
    ))
    base = Path(os.getenv(
        "BASE", str(astra / "bai_multinode_remote_core")
    ))
    old = Path(os.getenv(
        "OLD",
        str(astra / "bai_scale_4_8_12_16_hier_20261012")
    ))
    out = Path(os.getenv(
        "OUT",
        str(base / "paper_current_2node_gpu_scale_reuse_hier_16MiB")
    ))

    py = need(os.getenv(
        "PY", "/opt/venv/astra-sim/bin/python3"
    ))
    scheduler = need(
        base / "bai_multinode_remote_core.py"
    )
    converter = need(os.getenv(
        "CONVERTER",
        "/app/astra-sim/collectiveapi-046d961/chakra_converter/et_converter.py"
    ))
    binary = need(os.getenv(
        "BIN",
        str(ns3 / "build/scratch/ns3.42-AstraSimNetwork-default")
    ))
    remote = need(os.getenv(
        "REMOTE",
        str(astra / "examples/remote_memory/analytical/no_memory_expansion.json")
    ))

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(base / "legacy_reference"),
        str(base),
        str(capi / "msccl-tools"),
        env.get("PYTHONPATH", "")
    ])
    out.mkdir(parents=True, exist_ok=True)
    summary = []

    for m in a.gpus:
        n = 2 * m
        spec = f"2x{m}"

        # Read OLD Hierarchical Ring inputs and results.
        hist = old / "runs" / spec / "100Gbps" / "16MiB" / "hier"
        workload = (
            old / "workloads" / spec / "16MiB" / "all_gather"
        )
        logical = need(
            old / "config" / spec / "logical.json"
        )
        assert list(map(
            int,
            json.loads(logical.read_text())["logical-dims"]
        )) == [n]

        old_system = need(hist / "system.json")
        old_network = need(hist / "network.txt")
        old_topology = validate_old_topology(
            old_network, ns3, n
        )
        old_result = json.loads(
            need(hist / "result.json").read_text()
        )
        old_cycles, old_flows = runtime(
            hist / "run.log", hist / "fct.txt", n
        )
        assert old_cycles == int(old_result["cycles"])
        assert old_flows == n * (n-1)
        audit_xml(need(hist / "schedule.xml"), m)

        for r in range(n):
            need(f"{workload}.{r}.et")

        sys_obj = json.loads(old_system.read_text())
        assert isinstance(sys_obj[KEY], list)
        assert len(sys_obj[KEY]) == 1
        old_et = str(sys_obj[KEY][0])

        for r in range(n):
            need(f"{old_et}.{r}.et")

        # Validate historical input fingerprint when present.
        if "fingerprint" in old_result:
            evidence = {
                "dsl": sha(hist / "schedule.py"),
                "xml": sha(hist / "schedule.xml"),
                "topology": sha(old_topology),
                "logical": sha(logical),
                "binary": sha(binary),
                "converter": sha(converter),
                "system": sha(old_system),
                "network": sha(old_network),
                "et": [
                    sha(f"{old_et}.{r}.et")
                    for r in range(n)
                ],
                "workload": [
                    sha(f"{workload}.{r}.et")
                    for r in range(n)
                ]
            }
            old_fp = hashlib.sha256(
                json.dumps(
                    evidence, sort_keys=True
                ).encode()
            ).hexdigest()
            assert old_fp == old_result["fingerprint"], (
                f"{spec} old baseline input fingerprint mismatch"
            )

        print(
            f"HIER_REUSE {spec}: {old_cycles} cycles "
            "(no baseline simulation)",
            flush=True
        )

        # Generate CURRENT MultiNode-RemoteCore.
        case = out / spec
        case.mkdir(parents=True, exist_ok=True)
        plan = case / "schedule.py"
        xml = case / "schedule.xml"
        et = case / "et/custom_allgather"
        et.parent.mkdir(parents=True, exist_ok=True)

        static_inputs = [
            scheduler, converter, binary, remote,
            logical, old_system, old_network,
            old_topology, hist / "result.json",
            hist / "schedule.xml"
        ]
        static_inputs += [
            Path(f"{workload}.{r}.et")
            for r in range(n)
        ]
        legacy = sorted(
            (base / "legacy_reference").glob("*.py")
        )
        static_inputs += legacy
        assert legacy, "Missing legacy_reference sources"

        identifier = hashlib.sha256(
            json.dumps({
                "files": {
                    str(p): sha(p)
                    for p in static_inputs
                },
                "parameters": {
                    "nodes": 2,
                    "gpus_per_node": m,
                    "mode": "core",
                    "max_local_rounds": 1,
                    "size": SIZE
                }
            }, sort_keys=True).encode()
        ).hexdigest()

        stable_write(
            case / "source_fingerprint.txt",
            identifier + "\n"
        )

        if not plan.exists():
            execute([
                py, scheduler,
                "--nodes", "2",
                "--gpus-per-node", str(m),
                "--mode", "core",
                "--max-local-rounds", "1",
                "--freeze-dsl", plan,
                "--report-json", case / "plan.json"
            ], case / "planner.log",
               env=env, timeout=a.timeout)

        need(plan)

        if not xml.exists():
            execute([
                py, plan, "1", "--protocol", "Simple"
            ], case / "xml.log",
               stdout=xml, env=env, timeout=a.timeout)

        total, inter = audit_xml(need(xml), m)
        print(
            f"CURRENT_XML_PASS {spec}: "
            f"total={total}, inter={inter}",
            flush=True
        )

        if not Path(f"{et}.0.et").exists():
            execute([
                py, converter,
                "--input_filename", xml,
                "--output_filename", et,
                "--coll_size", str(SIZE),
                "--collective", "allgather",
                "--log_filename", case / "converter.log"
            ], case / "converter.stdout.log",
               timeout=a.timeout, env=env)

        for r in range(n):
            need(f"{et}.{r}.et")

        # Clone Hierarchical Ring simulation inputs.
        # Only the custom ET path and FCT output are changed.
        clone = dict(sys_obj)
        clone[KEY] = [str(et)]
        assert {
            k for k in clone
            if clone[k] != sys_obj[k]
        } == {KEY}

        stable_write(
            case / "system.json",
            json.dumps(clone, indent=2) + "\n"
        )
        stable_write(
            case / "network.txt",
            clone_network(
                old_network, case / "fct.txt"
            )
        )

        fingerprint = hashlib.sha256(
            json.dumps({
                "input": identifier,
                "xml": sha(xml),
                "system": sha(case / "system.json"),
                "network": sha(case / "network.txt"),
                "et": [
                    sha(Path(f"{et}.{r}.et"))
                    for r in range(n)
                ]
            }, sort_keys=True).encode()
        ).hexdigest()

        stable_write(
            case / "simulation_fingerprint.txt",
            fingerprint + "\n"
        )
        print(f"ET_PASS {spec}: {n} ranks", flush=True)

        if not a.run_sim:
            continue

        # Run ONLY the current MultiNode-RemoteCore.
        result = case / "result.json"

        if result.exists():
            saved = json.loads(result.read_text())
            assert saved["fingerprint"] == fingerprint
            cycles, flows = runtime(
                case / "run.log", case / "fct.txt", n
            )
            assert cycles == saved["cycles"]
            assert flows == saved["fct_flows"]
            print(
                f"REUSE_CURRENT_RESULT {spec}: {cycles} cycles",
                flush=True
            )
        else:
            if (case / "fct.txt").exists():
                (case / "fct.txt").unlink()

            execute([
                binary,
                f"--workload-configuration={workload}",
                f"--system-configuration={case/'system.json'}",
                f"--network-configuration={case/'network.txt'}",
                f"--remote-memory-configuration={remote}",
                f"--logical-topology-configuration={logical}",
                "--comm-group-configuration=empty"
            ], case / "run.log",
               cwd=ns3 / "build/scratch",
               timeout=a.timeout)

            cycles, flows = runtime(
                case / "run.log", case / "fct.txt", n
            )
            result.write_text(
                json.dumps({
                    "ranks": n,
                    "cycles": cycles,
                    "fct_flows": flows,
                    "fingerprint": fingerprint
                }, indent=2) + "\n"
            )
            print(
                f"SIM_PASS {spec}: "
                f"{cycles} cycles, {flows} flows",
                flush=True
            )

        gain = (
            (Decimal(old_cycles) - Decimal(cycles))
            * 100 / Decimal(old_cycles)
        )
        assert abs(
            Decimal(old_cycles) * (1 - gain/100)
            - Decimal(cycles)
        ) < Decimal("0.0000001")

        summary.append(
            (spec, cycles, old_cycles, gain)
        )

    if a.run_sim:
        lines = [
            "topology,current_cycles,hier_ring_cycles,current_time_reduction_pct"
        ]
        print(
            "TOPOLOGY  CURRENT_CYCLES  "
            "HIER_CYCLES  REDUCTION_%"
        )

        for spec, current, hier, gain in summary:
            print(
                f"{spec:8} {current:14,d} "
                f"{hier:11,d} {gain:12.4f}%"
            )
            lines.append(
                f"{spec},{current},{hier},{gain}"
            )

        csv_path = out / "comparison.csv"
        with csv_path.open("w") as f:
            f.write("\n".join(lines) + "\n")
        print("CSV=", csv_path)
    else:
        print(
            "PREFLIGHT_PASS: no simulation performed"
        )

    print("DONE OUT=", out)


if __name__ == "__main__":
    main()
