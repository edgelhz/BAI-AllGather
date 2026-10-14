#!/usr/bin/env python3
"""Strict ET-only Deferred-Inter ablation for MultiNode-RemoteCore.

Source: completed 16MiB / 400:100 ASTRA-sim+ns-3 current run.
Method: add only cross-node, root-ready SEND dependencies on an
inter-independent intra-node SEND/RECV from the same rank.
No MSCCL XML regeneration, changes to transfers, or baseline rerun.

The FCT analysis gives timing/frontier/overlap evidence, NOT an exact
GPU instruction-level critical path.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from chakra.src.third_party.utils.protolib import (
    decodeMessage, encodeMessage, openFileRd,
)
from chakra.schema.protobuf.et_def_pb2 import (
    COMM_RECV_NODE, COMM_SEND_NODE, GlobalMetadata, Node,
)

ET_KEY = "all-gather-implementation-custom"
SIZE = 16777216
OUTPUT_KEYS = {
    "FCT_OUTPUT_FILE", "TRACE_OUTPUT_FILE",
    "PFC_OUTPUT_FILE", "QLEN_MON_FILE"
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def need(path):
    path = Path(path)
    check(
        path.is_file() and path.stat().st_size > 0,
        f"MISSING: {path}"
    )
    return path


def sha(path):
    h = hashlib.sha256()
    with need(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def put_exact(path, data):
    """Create once; never silently rewrite existing artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if path.exists():
        check(
            path.read_bytes() == data,
            f"INPUT_CHANGED: {path}; use a new --out"
        )
    else:
        path.write_bytes(data)


def jtext(value):
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"


def load_json(path):
    return json.loads(need(path).read_text(encoding="utf-8"))


def attr(node, name):
    for a in node.attr:
        if a.name == name:
            field = a.WhichOneof("value")
            return getattr(a, field) if field else None
    return None


def message_key(rank, node, n):
    if node.type == COMM_SEND_NODE:
        dst = attr(node, "comm_dst")
        check(
            dst is not None,
            f"MISSING comm_dst: r{rank} node{node.id}"
        )
        src = rank
    elif node.type == COMM_RECV_NODE:
        src = attr(node, "comm_src")
        check(
            src is not None,
            f"MISSING comm_src: r{rank} node{node.id}"
        )
        dst = rank
    else:
        return None

    tag = attr(node, "comm_tag")
    size = attr(node, "comm_size")
    check(
        tag is not None and size is not None,
        f"MISSING tag/size: r{rank} node{node.id}"
    )
    src, dst, size, tag = (
        int(src), int(dst), int(size), int(tag)
    )
    check(
        0 <= src < n and 0 <= dst < n
        and src != dst and size > 0,
        f"INVALID TRANSFER: r{rank} node{node.id}"
    )
    return (src, dst, tag, size)


def load_et(path):
    fp = openFileRd(str(need(path)))
    try:
        meta = GlobalMetadata()
        check(
            decodeMessage(fp, meta),
            f"INVALID METADATA: {path}"
        )
        nodes = []
        while True:
            node = Node()
            if not decodeMessage(fp, node):
                break
            nodes.append(node)
        check(len(nodes) > 0, f"EMPTY ET: {path}")
        return meta, nodes
    finally:
        fp.close()


def et_bytes(meta, nodes):
    import io
    fp = io.BytesIO()
    encodeMessage(fp, meta)
    for node in nodes:
        encodeMessage(fp, node)
    return fp.getvalue()


def by_id(nodes, rank):
    result = {}
    for node in nodes:
        ident = int(node.id)
        check(
            ident not in result,
            f"DUPLICATE NODE ID: r{rank} node{ident}"
        )
        result[ident] = node

    for node in nodes:
        for p in list(node.data_deps) + list(node.ctrl_deps):
            check(
                int(p) in result,
                f"DANGLING DEP: r{rank} node{node.id} parent{p}"
            )
            check(
                int(p) != int(node.id),
                f"SELF DEP: r{rank} node{node.id}"
            )
    return result


def local_depths(nodes, rank):
    """Dependency-only chain length, not timed critical path."""
    ids = by_id(nodes, rank)
    active = set()
    memo = {}

    def depth(ident):
        if ident in memo:
            return memo[ident]
        check(
            ident not in active,
            f"ET CYCLE: r{rank} node{ident}"
        )
        active.add(ident)
        node = ids[ident]
        parents = list(node.data_deps) + list(node.ctrl_deps)
        ans = 1 + max(
            (depth(int(p)) for p in parents), default=0
        )
        active.remove(ident)
        memo[ident] = ans
        return ans

    for ident in ids:
        depth(ident)
    return memo


def verify_et_pair(
    source, deferred, meta_source, meta_deferred, m, n
):
    check(
        meta_source.SerializeToString()
        == meta_deferred.SerializeToString(),
        "METADATA_CHANGED"
    )
    check(
        set(source) == set(deferred) == set(range(n)),
        "MISSING ET RANKS"
    )

    added = []
    sends_a = collections.Counter()
    sends_b = collections.Counter()
    old_edges = new_edges = 0
    depth_a = depth_b = 0

    for r in sorted(source):
        s, d = source[r], deferred[r]
        check(
            len(s) == len(d),
            f"NODE_COUNT_CHANGED: r{r}"
        )
        check(
            [int(x.id) for x in s]
            == [int(x.id) for x in d],
            f"NODE_ORDER_CHANGED: r{r}"
        )

        depth_a = max(
            depth_a, max(local_depths(s, r).values())
        )
        depth_b = max(
            depth_b, max(local_depths(d, r).values())
        )

        for a, b in zip(s, d):
            x = Node()
            x.CopyFrom(a)
            x.ClearField("data_deps")

            y = Node()
            y.CopyFrom(b)
            y.ClearField("data_deps")

            check(
                x.SerializeToString(deterministic=True)
                == y.SerializeToString(deterministic=True),
                f"NONDEPENDENCY_CHANGED: r{r} node{a.id}"
            )

            pa = list(map(int, a.data_deps))
            pb = list(map(int, b.data_deps))

            check(
                len(pb) == len(set(pb)),
                f"DUPLICATE NEW PARENT: r{r} node{a.id}"
            )
            check(
                set(pa) <= set(pb),
                f"REMOVED DEP: r{r} node{a.id}"
            )

            extra = sorted(set(pb) - set(pa))
            if extra:
                key = message_key(r, a, n)
                check(
                    a.type == COMM_SEND_NODE
                    and key[0] // m != key[1] // m,
                    f"NON-INTER SEND CHANGED: r{r} node{a.id}"
                )
                check(
                    len(pa) == 0 and len(a.ctrl_deps) == 0,
                    f"NONROOT INTER SEND CHANGED: r{r} node{a.id}"
                )

                ids = by_id(d, r)
                for pid in extra:
                    anchor = ids[pid]
                    anchor_key = message_key(r, anchor, n)
                    check(
                        anchor_key is not None
                        and anchor_key[0] // m
                        == anchor_key[1] // m,
                        f"BAD ANCHOR: r{r} node{a.id} <- {pid}"
                    )
                    added.append({
                        "rank": r,
                        "parent_id": pid,
                        "send_id": int(a.id),
                        "sender": key[0],
                        "receiver": key[1],
                        "anchor_kind": (
                            "RECV"
                            if anchor.type == COMM_RECV_NODE
                            else "SEND"
                        )
                    })

            old_edges += len(pa) + len(a.ctrl_deps)
            new_edges += len(pb) + len(b.ctrl_deps)

            for node, dest in (
                (a, sends_a), (b, sends_b)
            ):
                if node.type == COMM_SEND_NODE:
                    dest[message_key(r, node, n)] += 1

    check(
        sends_a == sends_b,
        "TRANSFER MULTISET CHANGED"
    )
    check(
        old_edges + len(added) == new_edges,
        "EDGE ADDITION COUNT INCONSISTENT"
    )
    check(
        len(added) > 0,
        "NO_DEFERRED_EDGES: cannot call this an ablation"
    )

    return {
        "added_count": len(added),
        "added_edges": added,
        "original_dependency_edges": old_edges,
        "deferred_dependency_edges": new_edges,
        "original_max_local_dag_nodes": depth_a,
        "deferred_max_local_dag_nodes": depth_b,
        "transfer_count": sum(sends_a.values()),
        "inter_transfer_count": sum(
            v for k, v in sends_a.items()
            if k[0] // m != k[1] // m
        )
    }


def make_deferred(source, m, n, fallback_send):
    """Construct safe pre-inter local dependency anchors.

    Ambiguous duplicated message keys are handled conservatively:
    a local RECEIVE is pre-inter ready only when every possible
    matching intra SEND is pre-inter ready.
    """
    sends = collections.defaultdict(list)
    recvs = collections.defaultdict(list)

    for r, nodes in source.items():
        for node in nodes:
            k = message_key(r, node, n)
            if k is None:
                continue
            target = (
                sends if node.type == COMM_SEND_NODE
                else recvs
            )
            target[k].append((r, int(node.id)))

    check(
        collections.Counter({
            k: len(v) for k, v in sends.items()
        }) == collections.Counter({
            k: len(v) for k, v in recvs.items()
        }),
        "SOURCE ET SEND/RECV MESSAGE MULTISET MISMATCH"
    )

    ready = set()
    depths = {}

    while True:
        updated = 0
        for r in sorted(source):
            for node in source[r]:
                ident = (r, int(node.id))
                if ident in ready:
                    continue

                k = message_key(r, node, n)
                if (
                    k
                    and k[0] // m != k[1] // m
                ):
                    continue

                parents = [
                    (r, int(p))
                    for p in (
                        list(node.data_deps)
                        + list(node.ctrl_deps)
                    )
                ]
                if not all(p in ready for p in parents):
                    continue

                if node.type == COMM_RECV_NODE:
                    check(
                        k in sends,
                        f"UNMATCHED RECV: r{r} node{node.id}"
                    )
                    parents += sends[k]
                    if not all(
                        p in ready for p in parents
                    ):
                        continue

                ready.add(ident)
                depths[ident] = 1 + max(
                    (depths[p] for p in parents),
                    default=0
                )
                updated += 1

        if not updated:
            break

    deferred = {}
    skipped = []
    added_description = []

    for r, nodes in source.items():
        new_nodes = []

        for old in nodes:
            new = Node()
            new.CopyFrom(old)

            if new.type == COMM_SEND_NODE:
                k = message_key(r, new, n)

                if (
                    k[0] // m != k[1] // m
                    and not old.data_deps
                    and not old.ctrl_deps
                ):
                    recv_candidates = []
                    send_candidates = []

                    for a in nodes:
                        ident = (r, int(a.id))
                        if ident not in ready:
                            continue

                        ak = message_key(r, a, n)
                        if (
                            ak is None
                            or ak[0] // m
                            != ak[1] // m
                        ):
                            continue

                        if a.type == COMM_RECV_NODE:
                            recv_candidates.append(ident)
                        elif a.type == COMM_SEND_NODE:
                            send_candidates.append(ident)

                    choices = recv_candidates or (
                        send_candidates if fallback_send else []
                    )

                    if choices:
                        anchor = min(
                            choices,
                            key=lambda x: (
                                depths[x], x[1]
                            )
                        )
                        new.data_deps.append(anchor[1])
                        added_description.append({
                            "rank": r,
                            "send_id": int(new.id),
                            "parent_id": anchor[1],
                            "anchor_depth_preinter": depths[anchor],
                            "anchor_kind": (
                                "RECV"
                                if recv_candidates else "SEND"
                            )
                        })
                    else:
                        skipped.append({
                            "rank": r,
                            "send_id": int(new.id),
                            "reason": (
                                "no inter-independent intra anchor"
                            )
                        })

            new_nodes.append(new)

        deferred[r] = new_nodes

    check(
        added_description,
        "NO_SAFE_ANCHOR: cannot simulate zero-edge ablation"
    )
    return deferred, added_description, skipped


def read_config(path):
    d = {}
    for line in need(path).read_text().splitlines():
        w = line.split()
        if not w or line.lstrip().startswith("#"):
            continue
        check(
            w[0] not in d,
            f"DUPLICATE NETWORK KEY: {path}: {w[0]}"
        )
        d[w[0]] = w[1:]
    return d


def new_network_text(source, out):
    rows = []
    seen = set()
    replacements = {
        key: str(out / (key.lower() + ".txt"))
        for key in OUTPUT_KEYS
    }
    replacements["FCT_OUTPUT_FILE"] = str(
        out / "fct.txt"
    )

    for line in need(source).read_text().splitlines():
        w = line.split()
        if (
            w
            and not line.lstrip().startswith("#")
            and w[0] in OUTPUT_KEYS
        ):
            check(
                w[0] not in seen,
                f"DUPLICATE {w[0]}"
            )
            seen.add(w[0])
            line = f"{w[0]} {replacements[w[0]]}"
        rows.append(line)

    check(
        "FCT_OUTPUT_FILE" in seen,
        "NETWORK MISSING FCT_OUTPUT_FILE"
    )

    candidate = "\n".join(rows) + "\n"
    before = read_config(source)
    after = {}

    for line in candidate.splitlines():
        w = line.split()
        if w and not line.lstrip().startswith("#"):
            after[w[0]] = w[1:]

    check(
        set(before) == set(after)
        and all(
            before[k] == after[k]
            for k in before
            if k not in OUTPUT_KEYS
        ),
        "NETWORK FAIRNESS FAILED"
    )
    return candidate


def topology_file(network, ns3):
    config = read_config(network)
    name = config.get("TOPOLOGY_FILE")
    check(
        name and len(name) == 1,
        "INVALID TOPOLOGY_FILE"
    )

    p = Path(name[0])
    if not p.is_absolute():
        p = (ns3 / "build/scratch" / p).resolve()
    return need(p)


def verify_run(log, fct, n, expected_flows):
    text = need(log).read_text(errors="replace")
    finished = re.findall(
        r"sys\[(\d+)\] finished,\s*(\d+) cycles",
        text
    )
    check(
        len(finished) == n
        and {int(r) for r, _ in finished} == set(range(n)),
        f"INCOMPLETE SIMULATION: {log}; "
        f"finished={len(finished)}/{n}"
    )

    cycles = max(int(v) for _, v in finished)
    rows = parse_fct(fct, n)

    check(
        len(rows) == expected_flows,
        f"FLOW_COUNT_MISMATCH: {fct}: "
        f"got={len(rows)} want={expected_flows}"
    )
    return cycles, rows


def parse_fct(path, n):
    rows = []

    for line_number, line in enumerate(
        need(path).read_text().splitlines(), 1
    ):
        w = line.split()
        if not w:
            continue

        check(
            len(w) == 8,
            f"INVALID FCT ROW: {path}:{line_number}"
        )

        def rank(ip):
            value = int(ip, 16)
            check(
                value >> 24 == 11
                and ((value >> 16) & 255) == 0
                and (value & 255) == 1,
                f"UNEXPECTED IP LAYOUT {ip} in {path}"
            )
            r = (value >> 8) & 255
            check(
                0 <= r < n,
                f"FCT INVALID RANK {ip} in {path}"
            )
            return r

        src, dst = rank(w[0]), rank(w[1])
        size = int(w[4])
        start = int(w[5])
        duration = int(w[6])

        check(
            src != dst and size > 0
            and start >= 0 and duration > 0,
            f"INVALID FCT VALUES: {path}:{line_number}"
        )

        rows.append({
            "src": src,
            "dst": dst,
            "size": size,
            "start_ns": start,
            "end_ns": start + duration
        })

    return rows


def compare_fct_to_et(rows, source, n):
    expected = collections.Counter()

    for r, nodes in source.items():
        for node in nodes:
            if node.type == COMM_SEND_NODE:
                s, d, tag, size = message_key(r, node, n)
                expected[(s, d, size)] += 1

    observed = collections.Counter(
        (x["src"], x["dst"], x["size"])
        for x in rows
    )

    check(
        expected == observed,
        "FCT/ET SEND MULTISET MISMATCH (endpoint/bytes)"
    )
    return len(expected), sum(expected.values())


def timeline(rows, m, cycles):
    events = collections.defaultdict(lambda: [0, 0])
    last_inter = 0
    last_flow = 0

    for x in rows:
        typ = (
            0 if x["src"] // m != x["dst"] // m
            else 1
        )
        events[x["start_ns"]][typ] += 1
        events[x["end_ns"]][typ] -= 1

        if typ == 0:
            last_inter = max(
                last_inter, x["end_ns"]
            )
        last_flow = max(
            last_flow, x["end_ns"]
        )

    inter = intra = both = 0
    intra_concurrent_inter_ns = 0
    active_i = active_a = 0
    last = None
    max_inter = 0

    for t in sorted(events):
        if last is not None:
            dt = t - last
            if active_i > 0:
                inter += dt
                intra_concurrent_inter_ns += dt * active_a
            if active_a > 0:
                intra += dt
            if active_i > 0 and active_a > 0:
                both += dt

        di, da = events[t]
        active_i += di
        active_a += da

        check(
            active_i >= 0 and active_a >= 0,
            "NEGATIVE FLOW CONCURRENCY"
        )
        max_inter = max(max_inter, active_i)
        last = t

    check(
        active_i == active_a == 0,
        "UNCLOSED FLOW INTERVALS"
    )

    aligned = cycles == last_flow
    return {
        "last_inter_end_ns": last_inter,
        "last_flow_end_ns": last_flow,
        "post_inter_fct_tail_ns": last_flow - last_inter,
        "inter_active_union_ns": inter,
        "intra_active_union_ns": intra,
        "inter_intra_overlap_union_ns": both,
        "inter_intra_overlap_fraction": (
            round(both / inter, 6) if inter else 0
        ),
        "mean_intra_concurrency_during_inter": (
            round(
                intra_concurrent_inter_ns / inter, 4
            ) if inter else 0
        ),
        "max_inter_concurrency": max_inter,
        "cycles_equal_fct_end_ns": aligned,
        "post_inter_collective_tail_if_aligned_ns": (
            cycles - last_inter if aligned else None
        )
    }


def run_sim(
    binary, ns3, workload, system, network,
    remote, logical, job, timeout
):
    (job / "fct.txt").unlink(missing_ok=True)

    cmd = [
        str(binary),
        f"--workload-configuration={workload}",
        f"--system-configuration={system}",
        f"--network-configuration={network}",
        f"--remote-memory-configuration={remote}",
        f"--logical-topology-configuration={logical}",
        "--comm-group-configuration=empty"
    ]

    with (job / "run.log").open("w") as f:
        try:
            p = subprocess.run(
                cmd,
                cwd=ns3 / "build/scratch",
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"SIM_TIMEOUT {job}: {exc}"
            ) from exc

    if p.returncode:
        lines = (
            (job / "run.log")
            .read_text(errors="replace")
            .splitlines()
        )
        raise RuntimeError(
            f"SIM_FAILED {job} rc={p.returncode}\n"
            + "\n".join(lines[-30:])
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--specs", nargs="+", default=["3x4"]
    )
    ap.add_argument(
        "--case",
        help="Explicit completed current case; single spec only"
    )
    ap.add_argument(
        "--out",
        help="Separate destination for ablation artifacts"
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="Run deferred simulation; otherwise preflight only"
    )
    ap.add_argument(
        "--timeout", type=int, default=7200
    )
    ap.add_argument(
        "--recv-only",
        action="store_true",
        help="Never use local SEND fallback anchor"
    )
    args = ap.parse_args()

    check(
        args.timeout > 0
        and len(args.specs) == len(set(args.specs)),
        "BAD CLI ARGUMENTS"
    )
    check(
        not args.case or len(args.specs) == 1,
        "--case REQUIRES SINGLE SPEC"
    )

    astra = Path(os.environ.get(
        "ASTRA", "/app/astra-sim/astra-sim"
    ))
    ns3 = Path(os.environ.get(
        "NS3", str(astra / "extern/network_backend/ns-3")
    ))
    base = Path(os.environ.get(
        "BASE", str(astra / "bai_multinode_remote_core")
    ))
    source_root = Path(os.environ.get(
        "SOURCE_ROOT", str(base / "perf_star_16MiB")
    ))
    out = Path(args.out or (
        base / "deferred_inter_strict_20261014"
    ))

    binary = need(os.environ.get(
        "BIN",
        str(ns3 / "build/scratch/ns3.42-AstraSimNetwork-default")
    ))
    remote = need(os.environ.get(
        "REMOTE",
        str(astra / "examples/remote_memory/analytical/no_memory_expansion.json")
    ))
    need(__file__)

    out.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for spec in args.specs:
        check(
            re.fullmatch(
                r"[2-9][0-9]*x[1-9][0-9]*", spec
            ) is not None,
            f"INVALID SPEC: {spec}"
        )

        k, m = map(int, spec.split("x"))
        n = k * m
        check(
            n <= 254,
            "FCT rank-to-IP mapping requires <255 GPUs"
        )

        case = (
            Path(args.case)
            if args.case
            else source_root / spec / f"{SIZE}B"
        )
        case = case.resolve()
        job = (out / spec).resolve()

        check(
            job != case
            and job not in case.parents
            and case not in job.parents,
            "OUTPUT MUST BE OUTSIDE ORIGINAL CASE"
        )

        need(case / "result.json")
        need(case / "run.log")
        need(case / "fct.txt")

        system_source = need(case / "system.json")
        network_source = need(case / "network.txt")
        workload = case / "workload/all_gather"
        logical = need(case / "logical.json")
        system = load_json(system_source)

        check(
            isinstance(system.get(ET_KEY), list)
            and len(system[ET_KEY]) == 1,
            f"INVALID CUSTOM ET SYSTEM: {system_source}"
        )

        current_prefix = system[ET_KEY][0]
        for r in range(n):
            need(f"{current_prefix}.{r}.et")
            need(f"{workload}.{r}.et")

        topo = topology_file(network_source, ns3)

        check(
            list(map(
                int,
                load_json(logical).get("logical-dims", [])
            )) == [n],
            f"LOGICAL TOPOLOGY NOT MATCHED {spec}"
        )

        meta = {}
        current = {}

        for r in range(n):
            meta[r], current[r] = load_et(
                f"{current_prefix}.{r}.et"
            )

        new, chosen, skipped = make_deferred(
            current, m, n, not args.recv_only
        )

        audit = verify_et_pair(
            current, new, meta[0], meta[0], m, n
        )

        for r in range(n):
            check(
                meta[r].SerializeToString()
                == meta[0].SerializeToString(),
                f"UNEXPECTED PER-RANK METADATA: r{r}"
            )

        check(
            len(chosen) == audit["added_count"],
            "ADDED EDGE REPORT INCONSISTENT"
        )

        current_result = load_json(
            case / "result.json"
        )

        original_cycles, original_flows = verify_run(
            case / "run.log",
            case / "fct.txt",
            n,
            audit["transfer_count"]
        )

        check(
            int(current_result["cycles"]) == original_cycles
            and int(current_result["fct_flows"])
            == len(original_flows)
            and int(current_result["ranks"]) == n,
            "SOURCE RESULT/LOG/FCT MISMATCH"
        )

        compare_fct_to_et(
            original_flows, current, n
        )

        before = timeline(
            original_flows, m, original_cycles
        )

        job.mkdir(
            parents=True, exist_ok=True
        )
        pref = job / "et/custom_allgather"

        for r in range(n):
            put_exact(
                f"{pref}.{r}.et",
                et_bytes(meta[r], new[r])
            )

            parsed_meta, parsed_nodes = load_et(
                f"{pref}.{r}.et"
            )

            check(
                parsed_meta.SerializeToString()
                == meta[r].SerializeToString()
                and [
                    x.SerializeToString()
                    for x in parsed_nodes
                ] == [
                    x.SerializeToString()
                    for x in new[r]
                ],
                f"FAILED ET RELOAD: r{r}"
            )

        new_system = json.loads(
            json.dumps(system)
        )
        new_system[ET_KEY] = [str(pref)]

        put_exact(
            job / "system.json",
            jtext(new_system)
        )

        check(
            {
                key for key in system
                if system[key] != new_system[key]
            } == {ET_KEY},
            "SYSTEM CONFIG UNEXPECTED CHANGE"
        )

        put_exact(
            job / "network.txt",
            new_network_text(
                network_source, job
            )
        )

        source_files = [
            need(__file__),
            binary,
            remote,
            topo,
            logical,
            system_source,
            network_source,
            case / "result.json",
            case / "run.log",
            case / "fct.txt"
        ]

        source_files += [
            need(f"{current_prefix}.{r}.et")
            for r in range(n)
        ]
        source_files += [
            need(f"{workload}.{r}.et")
            for r in range(n)
        ]

        digest_input = {
            str(p): sha(p)
            for p in source_files
        }

        fingerprint = hashlib.sha256(
            jtext(digest_input).encode()
        ).hexdigest()

        put_exact(
            job / "fingerprint.sha256",
            fingerprint + "\n"
        )

        put_exact(
            job / "audit.json",
            jtext({
                "spec": spec,
                "size_bytes": SIZE,
                "source_case": str(case),
                "fingerprint": fingerprint,
                "input_sha256": digest_input,
                "audit": audit,
                "chosen_anchors": chosen,
                "skipped_root_inter_sends": skipped,
                "source_fct_timeline": before
            })
        )

        print(
            f"PREFLIGHT_PASS {spec}: "
            f"current_cycles={original_cycles} "
            f"transfers={audit['transfer_count']} "
            f"inter={audit['inter_transfer_count']} "
            f"new_edges={audit['added_count']} "
            f"skipped={len(skipped)} "
            f"local_dag_nodes="
            f"{audit['original_max_local_dag_nodes']}->"
            f"{audit['deferred_max_local_dag_nodes']}",
            flush=True
        )

        if not args.run:
            continue

        result_path = job / "result.json"

        if result_path.exists():
            saved = load_json(result_path)
            check(
                saved["fingerprint"] == fingerprint,
                "RESULT FINGERPRINT MISMATCH"
            )

            cycles, deferred_flows = verify_run(
                job / "run.log",
                job / "fct.txt",
                n,
                audit["transfer_count"]
            )

            check(
                cycles == saved["cycles"],
                "CACHED RESULT MISMATCH"
            )

            print(
                f"REUSE_RESULT {spec}: {cycles}",
                flush=True
            )
        else:
            print(
                f"RUN_DEFERRED {spec}",
                flush=True
            )

            run_sim(
                binary,
                ns3,
                workload,
                job / "system.json",
                job / "network.txt",
                remote,
                logical,
                job,
                args.timeout
            )

            cycles, deferred_flows = verify_run(
                job / "run.log",
                job / "fct.txt",
                n,
                audit["transfer_count"]
            )

            put_exact(
                result_path,
                jtext({
                    "spec": spec,
                    "ranks": n,
                    "cycles": cycles,
                    "flows": len(deferred_flows),
                    "fingerprint": fingerprint
                })
            )

            print(
                f"SIM_PASS {spec}: "
                f"cycles={cycles} "
                f"flows={len(deferred_flows)}",
                flush=True
            )

        compare_fct_to_et(
            deferred_flows, new, n
        )

        after = timeline(
            deferred_flows, m, cycles
        )

        reduction = (
            100.0
            * (cycles - original_cycles)
            / cycles
        )

        metrics = {
            "spec": spec,
            "current_cycles": original_cycles,
            "deferred_cycles": cycles,
            "current_time_reduction_pct": reduction,
            "current_last_inter_end_ns":
                before["last_inter_end_ns"],
            "deferred_last_inter_end_ns":
                after["last_inter_end_ns"],
            "last_inter_advance_ns":
                after["last_inter_end_ns"]
                - before["last_inter_end_ns"],
            "current_overlap_fraction":
                before["inter_intra_overlap_fraction"],
            "deferred_overlap_fraction":
                after["inter_intra_overlap_fraction"],
            "current_post_inter_fct_tail_ns":
                before["post_inter_fct_tail_ns"],
            "deferred_post_inter_fct_tail_ns":
                after["post_inter_fct_tail_ns"],
            "new_dependencies":
                audit["added_count"],
            "source_timeline": before,
            "deferred_timeline": after
        }

        put_exact(
            job / "mechanism.json",
            jtext(metrics)
        )

        report_rows.append({
            key: val
            for key, val in metrics.items()
            if key not in (
                "source_timeline",
                "deferred_timeline"
            )
        })

        print(
            f"COMPARE {spec}: "
            f"current={original_cycles} "
            f"deferred={cycles} "
            f"current_reduction={reduction:.4f}% "
            f"last_inter_advance_ns="
            f"{metrics['last_inter_advance_ns']} "
            f"overlap="
            f"{before['inter_intra_overlap_fraction']:.4f}/"
            f"{after['inter_intra_overlap_fraction']:.4f}",
            flush=True
        )

    if args.run and report_rows:
        summary = (
            out / "deferred_inter_comparison.csv"
        )
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=list(report_rows[0])
        )
        writer.writeheader()
        writer.writerows(report_rows)
        summary.write_text(
            buf.getvalue(), encoding="utf-8"
        )
        print(
            f"CSV={summary}",
            flush=True
        )

    if not args.run:
        print(
            "PREFLIGHT_ONLY: no simulation started",
            flush=True
        )

    print(
        f"DONE OUT={out}",
        flush=True
    )


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError, OSError, ValueError, KeyError
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
            flush=True
        )
        sys.exit(2)
