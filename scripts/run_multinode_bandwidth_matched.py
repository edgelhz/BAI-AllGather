#!/usr/bin/env python3
"""Bandwidth-only ns-3 sweep using existing 16 MiB schedules."""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from decimal import Decimal
from pathlib import Path

KEY = "all-gather-implementation-custom"
SIZE = 16777216
OUTPUT_KEYS = {
    "FCT_OUTPUT_FILE", "TRACE_OUTPUT_FILE",
    "PFC_OUTPUT_FILE", "QLEN_MON_FILE"
}
ECN_KEYS = ("KMIN_MAP", "KMAX_MAP", "PMAX_MAP")


def require(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"MISSING_FILE: {path}")
    return path


def sha(path):
    h = hashlib.sha256()
    with require(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def put_unchanged(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != contents:
            raise RuntimeError(
                f"STALE_CACHE_INPUT: {path}; choose a fresh OUT"
            )
    else:
        path.write_text(contents)


def jsonfile(path):
    return json.loads(require(path).read_text())


def lookup_system(case):
    matches = []
    for p in case.rglob("*.json"):
        if len(p.relative_to(case).parts) > 3:
            continue
        try:
            j = jsonfile(p)
            if (
                isinstance(j, dict)
                and isinstance(j.get(KEY), list)
                and len(j[KEY]) == 1
            ):
                matches.append(p)
        except (ValueError, OSError):
            pass

    if not matches:
        raise RuntimeError(f"SYSTEM_NOT_FOUND: {case}")

    preferred = [
        p for p in matches if p.name == "system.json"
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(f"AMBIGUOUS_SYSTEM: {matches}")


def lookup_network(case):
    matches = []
    for p in case.rglob("*.txt"):
        if len(p.relative_to(case).parts) > 3:
            continue
        try:
            text = p.read_text()
            if all(
                x in text for x in (
                    "TOPOLOGY_FILE",
                    "FCT_OUTPUT_FILE",
                    "KMIN_MAP"
                )
            ):
                matches.append(p)
        except (UnicodeError, OSError):
            pass

    if not matches:
        raise RuntimeError(f"NETWORK_NOT_FOUND: {case}")

    preferred = [
        p for p in matches if p.name == "network.txt"
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(f"AMBIGUOUS_NETWORK: {matches}")


def config_lines(path):
    rows = {}
    for line in require(path).read_text().splitlines():
        words = line.split()
        if words and not line.lstrip().startswith("#"):
            key = words[0]
            if key in rows:
                raise RuntimeError(
                    f"DUPLICATE_NETWORK_KEY: {key}"
                )
            rows[key] = words[1:]
    return rows


def resolve_topology(netfile, ns3):
    fields = config_lines(netfile).get("TOPOLOGY_FILE")
    if not fields or len(fields) != 1:
        raise RuntimeError(
            f"INVALID_TOPOLOGY_FILE: {netfile}"
        )

    t = Path(fields[0])
    if not t.is_absolute():
        t = (ns3 / "build/scratch" / t).resolve()

    return require(t)


def audit_star(topo, n, k, m, target=None):
    lines = topo.read_text().splitlines()
    records = [
        (i, line.split())
        for i, line in enumerate(lines)
        if line.strip()
        and not line.lstrip().startswith("#")
    ]

    assert len(records) == n + k + 2
    assert list(map(int, records[0][1])) == [
        n+k+1, k+1, n+k
    ]
    assert list(map(int, records[1][1])) == list(
        range(n, n+k+1)
    )

    hosts = set()
    uplinks = set()
    output = list(lines)

    for pos, w in records[2:]:
        assert len(w) >= 5

        a, b = int(w[0]), int(w[1])

        if a < n or b < n:
            g = a if a < n else b
            local = b if a < n else a

            assert 0 <= g < n
            assert local == n + g // m
            assert g not in hosts
            assert w[2] == "400Gbps"

            hosts.add(g)

        else:
            assert n+k in (a, b)
            local = b if a == n+k else a

            assert n <= local < n+k
            assert local not in uplinks
            assert w[2] == "100Gbps"

            uplinks.add(local)

            if target is not None:
                w[2] = f"{target}Gbps"
                output[pos] = " ".join(w)

    assert hosts == set(range(n))
    assert uplinks == set(range(n, n+k))

    return "\n".join(output) + "\n"


def parse_map(fields, key):
    assert fields and len(fields) >= 2

    count = int(fields[0])
    raw = fields[1:]

    assert len(raw) == 2 * count

    mapping = {
        int(raw[i]): raw[i+1]
        for i in range(0, len(raw), 2)
    }

    assert len(mapping) == count
    return mapping


def load_policy(rate, paper):
    cfg = (
        paper
        / f"intra400_inter{rate}"
        / "current/network.txt"
    )

    net = config_lines(require(cfg))
    target = rate * 1000000000
    policy = {}

    for key in ECN_KEYS:
        values = parse_map(net[key], key)

        if target not in values:
            raise RuntimeError(
                f"ECN_{rate}_NOT_FOUND: {key} in {cfg}"
            )

        policy[key] = values[target]

    return policy, cfg


def matched_configs(
    bai_sys, hier_sys, bai_net, hier_net, ns3
):
    b = jsonfile(bai_sys)
    h = jsonfile(hier_sys)

    assert set(b) == set(h)
    assert KEY in b

    assert {
        x for x in b if b[x] != h[x]
    } == {KEY}, "SYSTEM_CONFIG_NOT_MATCHED"

    n1 = config_lines(bai_net)
    n2 = config_lines(hier_net)

    assert set(n1) == set(n2)

    ignored = OUTPUT_KEYS | {"TOPOLOGY_FILE"}

    assert {
        key for key in n1
        if n1[key] != n2[key]
    } <= ignored, "NETWORK_CONFIG_NOT_MATCHED"

    assert sha(
        resolve_topology(bai_net, ns3)
    ) == sha(
        resolve_topology(hier_net, ns3)
    ), "PHYSICAL_TOPOLOGY_NOT_MATCHED"

    return b, h


def make_network(source, topology, fct, rate, policy):
    rows = config_lines(source)
    target = rate * 1000000000

    assert "TOPOLOGY_FILE" in rows
    assert "FCT_OUTPUT_FILE" in rows

    lines = []
    seen = set()

    for line in require(source).read_text().splitlines():
        w = line.split()

        if not w or line.lstrip().startswith("#"):
            lines.append(line)
            continue

        key = w[0]

        if key == "TOPOLOGY_FILE":
            line = f"TOPOLOGY_FILE {topology}"

        elif key == "FCT_OUTPUT_FILE":
            line = f"FCT_OUTPUT_FILE {fct}"

        elif key in OUTPUT_KEYS - {"FCT_OUTPUT_FILE"}:
            line = (
                f"{key} "
                f"{fct.parent / (key.lower()+'.txt')}"
            )

        elif key in ECN_KEYS:
            mapping = parse_map(rows[key], key)

            assert 100000000000 in mapping
            assert 400000000000 in mapping

            if target not in mapping:
                mapping[target] = policy[key]

            tokens = [key, str(len(mapping))]

            for bps, value in sorted(mapping.items()):
                tokens.extend([str(bps), value])

            line = " ".join(tokens)

        if key in (
            "TOPOLOGY_FILE",
            "FCT_OUTPUT_FILE"
        ) + ECN_KEYS:
            seen.add(key)

        lines.append(line)

    assert seen == set(
        ("TOPOLOGY_FILE", "FCT_OUTPUT_FILE")
        + ECN_KEYS
    )

    return "\n".join(lines) + "\n"


def choose_logical(
    spec, n, bai_case, hier_case,
    base, bairoot, hierroot
):
    env = (
        os.getenv("LOGICAL_"+spec.replace("x", "X"))
        or os.getenv("LOGICAL_"+spec)
    )

    if env:
        j = jsonfile(env)
        assert list(map(
            int, j["logical-dims"]
        )) == [n]
        return Path(env)

    roots = [
        bai_case, hier_case,
        bai_case.parent, hier_case.parent,
        bairoot, hierroot, base
    ]

    seen = set()
    matches = []

    for parent in roots:
        if not parent.is_dir():
            continue

        for path in parent.rglob("*.json"):
            if path in seen:
                continue
            seen.add(path)

            if "logical" not in path.name.lower():
                continue

            try:
                j = jsonfile(path)
                dims = list(map(
                    int, j.get("logical-dims", [])
                ))

                if dims == [n]:
                    matches.append(path)

            except (
                ValueError, KeyError,
                TypeError, OSError
            ):
                continue

        if matches:
            hashes = {sha(p) for p in matches}

            if len(hashes) == 1:
                return matches[0]

            raise RuntimeError(
                f"AMBIGUOUS_LOGICAL_{spec}: {matches[:12]}"
            )

    raise RuntimeError(
        f"LOGICAL_NOT_FOUND_{spec}"
    )


def choose_workload(
    spec, n, bai_case, hier_case,
    base, bairoot, hierroot
):
    env = (
        os.getenv("WORKLOAD_"+spec.replace("x", "X"))
        or os.getenv("WORKLOAD_"+spec)
    )

    if env:
        for r in range(n):
            require(f"{env}.{r}.et")
        return Path(env)

    roots = [
        bai_case, hier_case,
        bai_case.parent, hier_case.parent,
        bairoot, hierroot, base
    ]

    seen = set()
    matches = []

    for parent in roots:
        if not parent.is_dir():
            continue

        for file in parent.rglob("all_gather.0.et"):
            prefix = Path(
                str(file)[:-len(".0.et")]
            )

            if prefix in seen:
                continue

            seen.add(prefix)

            complete = all(
                Path(f"{prefix}.{r}.et").is_file()
                for r in range(n)
            )

            if (
                complete
                and not Path(f"{prefix}.{n}.et").exists()
            ):
                matches.append(prefix)

        if matches:
            signatures = {
                tuple(
                    sha(f"{p}.{i}.et")
                    for i in range(n)
                )
                for p in matches
            }

            if len(signatures) == 1:
                return matches[0]

            raise RuntimeError(
                f"AMBIGUOUS_WORKLOAD_{spec}"
            )

    raise RuntimeError(
        f"WORKLOAD_NOT_FOUND_{spec}"
    )


def verify_runtime(log, fct, n):
    text = require(log).read_text(errors="replace")

    rows = re.findall(
        r"sys\[(\d+)\] finished,\s*(\d+) cycles",
        text
    )

    assert len(rows) == n
    assert {
        int(r) for r, _ in rows
    } == set(range(n))

    cycles = max(
        int(t) for _, t in rows
    )

    flows = sum(
        bool(x.strip())
        for x in require(fct).read_text().splitlines()
    )

    assert cycles > 0
    assert flows == n * (n-1)

    return cycles, flows


def verify_source(case, n):
    data = jsonfile(case / "result.json")

    assert int(data["ranks"]) == n
    assert int(data["fct_flows"]) == n*(n-1)

    cycles, flows = verify_runtime(
        case / "run.log",
        case / "fct.txt",
        n
    )

    assert cycles == int(data["cycles"])
    assert flows == n*(n-1)

    return cycles


def simulate(
    binary, ns3, workload,
    system, network, remote,
    logical, case, timeout
):
    case.mkdir(parents=True, exist_ok=True)

    (case / "fct.txt").unlink(missing_ok=True)

    cmd = [
        binary,
        f"--workload-configuration={workload}",
        f"--system-configuration={system}",
        f"--network-configuration={network}",
        f"--remote-memory-configuration={remote}",
        f"--logical-topology-configuration={logical}",
        "--comm-group-configuration=empty"
    ]

    with (case / "run.log").open("w") as fd:
        try:
            cp = subprocess.run(
                list(map(str, cmd)),
                cwd=ns3 / "build/scratch",
                stdout=fd,
                stderr=subprocess.STDOUT,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TIMEOUT: {case}: {exc}"
            )

    if cp.returncode != 0:
        lines = (
            case / "run.log"
        ).read_text(errors="replace").splitlines()

        raise RuntimeError(
            f"SIM_FAILED: {case} "
            f"code={cp.returncode}\n"
            + "\n".join(lines[-35:])
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--run", action="store_true"
    )
    ap.add_argument(
        "--specs", nargs="+",
        default=["3x4", "4x4", "8x4"]
    )
    ap.add_argument(
        "--rates", nargs="+", type=int,
        default=[40, 200]
    )
    ap.add_argument(
        "--timeout", type=int,
        default=7200
    )

    args = ap.parse_args()

    assert args.timeout > 0
    assert len(set(args.specs)) == len(args.specs)
    assert len(set(args.rates)) == len(args.rates)
    assert set(args.rates) <= {40, 200}

    for spec in args.specs:
        assert re.fullmatch(
            r"[3-9]\dx[1-9]\d*|[3-9]x[1-9]\d*",
            spec
        )

    astra = Path(os.getenv(
        "ASTRA", "/app/astra-sim/astra-sim"
    ))
    ns3 = Path(os.getenv(
        "NS3",
        str(astra / "extern/network_backend/ns-3")
    ))
    base = Path(os.getenv(
        "BASE",
        str(astra / "bai_multinode_remote_core")
    ))

    bairoot = Path(os.getenv(
        "BAIROOT",
        str(base / "perf_star_16MiB")
    ))
    hier34 = Path(os.getenv(
        "HIER34",
        str(base / "hier_ring_matched_stUubJh9")
    ))
    hier8 = Path(os.getenv(
        "HIER8",
        str(base / "nodecount_yC3RLZ08/hier")
    ))
    paper = Path(os.getenv(
        "PAPER",
        str(base / "bandwidth_fig4_current_20261013")
    ))
    out = Path(os.getenv(
        "OUT",
        str(base / "bandwidth_star_multinode_20261013")
    ))

    binary = require(os.getenv(
        "BIN",
        str(
            ns3 / "build/scratch"
            / "ns3.42-AstraSimNetwork-default"
        )
    ))
    remote = require(os.getenv(
        "REMOTE",
        str(
            astra
            / "examples/remote_memory/analytical"
            / "no_memory_expansion.json"
        )
    ))

    policies = {
        rate: load_policy(rate, paper)
        for rate in args.rates
    }

    out.mkdir(parents=True, exist_ok=True)
    results = []

    for spec in args.specs:
        k, m = map(int, spec.split("x"))
        n = k*m

        bcase = (
            bairoot / spec / f"{SIZE}B"
        )
        hroot = (
            hier34
            if spec in ("3x4", "4x4")
            else hier8
        )
        hcase = (
            hroot / spec / f"{SIZE}B"
        )

        base_cycles = {
            "current": verify_source(bcase, n),
            "hier": verify_source(hcase, n)
        }

        sources = {
            "current": (
                lookup_system(bcase),
                lookup_network(bcase),
                bcase
            ),
            "hier": (
                lookup_system(hcase),
                lookup_network(hcase),
                hcase
            )
        }

        cur_sys, cur_net, _ = sources["current"]
        hier_sys, hier_net, _ = sources["hier"]

        matched_configs(
            cur_sys, hier_sys,
            cur_net, hier_net, ns3
        )

        raw_topo = resolve_topology(cur_net, ns3)
        audit_star(raw_topo, n, k, m)

        logical = choose_logical(
            spec, n, bcase, hcase,
            base, bairoot, hroot
        )
        workload = choose_workload(
            spec, n, bcase, hcase,
            base, bairoot, hroot
        )

        for role, (sy, ne, _) in sources.items():
            for r in range(n):
                prefix = jsonfile(sy)[KEY][0]
                require(f"{prefix}.{r}.et")

        for r in range(n):
            require(f"{workload}.{r}.et")

        print(
            f"INPUT_PASS {spec}: ranks={n} "
            f"16MiB current={base_cycles['current']} "
            f"hier={base_cycles['hier']}",
            flush=True
        )

        for rate in args.rates:
            policy, policyfile = policies[rate]

            top = (
                ns3 / "scratch/topology"
                / f"{out.name}_{spec}_400_{rate}.txt"
            )

            put_unchanged(
                top,
                audit_star(
                    raw_topo, n, k, m,
                    target=rate
                )
            )

            values = {}

            for role, (sy, net, _) in sources.items():
                job = (
                    out / spec / f"inter{rate}" / role
                )
                job.mkdir(
                    parents=True, exist_ok=True
                )

                system_text = json.dumps(
                    jsonfile(sy),
                    indent=2,
                    sort_keys=True
                ) + "\n"

                network_text = make_network(
                    net, top,
                    job / "fct.txt",
                    rate, policy
                )

                put_unchanged(
                    job / "system.json",
                    system_text
                )
                put_unchanged(
                    job / "network.txt",
                    network_text
                )

                files = [
                    sy, net, raw_topo, top,
                    job / "system.json",
                    job / "network.txt",
                    logical, remote, binary,
                    policyfile
                ]

                prefix = jsonfile(sy)[KEY][0]
                files += [
                    Path(f"{prefix}.{r}.et")
                    for r in range(n)
                ]
                files += [
                    Path(f"{workload}.{r}.et")
                    for r in range(n)
                ]

                fingerprint = hashlib.sha256(
                    json.dumps({
                        str(p): sha(p)
                        for p in files
                    }, sort_keys=True).encode()
                ).hexdigest()

                put_unchanged(
                    job / "fingerprint.sha256",
                    fingerprint + "\n"
                )

                saved = job / "result.json"

                if saved.exists():
                    data = jsonfile(saved)

                    assert (
                        data["fingerprint"]
                        == fingerprint
                    )

                    t, flows = verify_runtime(
                        job / "run.log",
                        job / "fct.txt",
                        n
                    )

                    assert t == int(data["cycles"])
                    assert flows == int(
                        data["fct_flows"]
                    )

                    values[role] = t

                    print(
                        f"REUSE_RESULT {spec} "
                        f"400/{rate} {role}: {t}",
                        flush=True
                    )

                elif not args.run:
                    print(
                        f"PRECHECK_PASS {spec} "
                        f"400/{rate} {role}",
                        flush=True
                    )

                else:
                    print(
                        f"RUN {spec} "
                        f"400/{rate} {role}",
                        flush=True
                    )

                    simulate(
                        binary, ns3, workload,
                        job / "system.json",
                        job / "network.txt",
                        remote, logical,
                        job, args.timeout
                    )

                    t, flows = verify_runtime(
                        job / "run.log",
                        job / "fct.txt",
                        n
                    )

                    saved.write_text(
                        json.dumps({
                            "ranks": n,
                            "cycles": t,
                            "fct_flows": flows,
                            "fingerprint": fingerprint
                        }, indent=2) + "\n"
                    )

                    values[role] = t

                    print(
                        f"SIM_PASS {spec} "
                        f"400/{rate} {role}: "
                        f"cycles={t} flows={flows}",
                        flush=True
                    )

            if len(values) == 2:
                gain = (
                    Decimal(
                        values["hier"]
                    ) - Decimal(
                        values["current"]
                    )
                ) * 100 / Decimal(
                    values["hier"]
                )

                results.append([
                    spec, 400, rate, "16MiB",
                    values["current"],
                    values["hier"],
                    str(gain), "new"
                ])

                print(
                    f"COMPARE {spec} 400/{rate}: "
                    f"reduction={gain:.4f}%",
                    flush=True
                )

        gain = (
            Decimal(
                base_cycles["hier"]
            ) - Decimal(
                base_cycles["current"]
            )
        ) * 100 / Decimal(
            base_cycles["hier"]
        )

        results.append([
            spec, 400, 100, "16MiB",
            base_cycles["current"],
            base_cycles["hier"],
            str(gain), "reused"
        ])

    if args.run:
        csvfile = (
            out / "bandwidth_multinode_comparison.csv"
        )

        with csvfile.open(
            "w", newline=""
        ) as fp:
            writer = csv.writer(fp)

            writer.writerow([
                "topology",
                "intra_gbps",
                "inter_gbps",
                "size",
                "bai_cycles",
                "hier_cycles",
                "bai_time_reduction_pct",
                "source"
            ])

            writer.writerows(
                sorted(
                    results,
                    key=lambda r: (
                        int(r[0].split("x")[0]),
                        r[2]
                    )
                )
            )

        print(
            f"CSV={csvfile}",
            flush=True
        )

    else:
        print(
            "PREFLIGHT_PASS: no simulation executed",
            flush=True
        )

    print(
        f"DONE OUT={out}",
        flush=True
    )


if __name__ == "__main__":
    main()
