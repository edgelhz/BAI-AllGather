#!/usr/bin/env python3
"""Greedy BAI-like AllGather scheduler for a dual-node GPU topology."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# -----------------------------------------------------------------------------
# Topology and alpha-beta model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    bandwidth_gbps: float
    alpha_ns: float = 0.0

    def one_way_ns(self, chunks: int, chunk_bytes: int) -> float:
        if chunks <= 0:
            return 0.0
        if self.bandwidth_gbps <= 0 or self.alpha_ns < 0:
            raise ValueError("invalid link model")
        # 1 Gbit/s == 1 bit/ns. Each chunk is emitted by one copy(), so alpha
        # is charged once per chunk.
        return (
            chunks * self.alpha_ns
            + chunks * chunk_bytes * 8.0 / self.bandwidth_gbps
        )

    def exchange_ns(self, a_to_b: int, b_to_a: int, chunk_bytes: int) -> float:
        # Full-duplex approximation for the two directions of the node cut.
        return max(
            self.one_way_ns(a_to_b, chunk_bytes),
            self.one_way_ns(b_to_a, chunk_bytes),
        )


@dataclass(frozen=True)
class Topology:
    nodes: Tuple[Tuple[int, ...], Tuple[int, ...]]
    intra: Link
    inter: Link
    chunk_bytes: int

    def __post_init__(self) -> None:
        if len(self.nodes) != 2 or len(self.nodes[0]) != len(self.nodes[1]):
            raise ValueError("need exactly two equal-size GPU nodes")
        ranks = [r for node in self.nodes for r in node]
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError("ranks must be exactly 0..N-1")
        if self.chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be > 0")

    @property
    def nranks(self) -> int:
        return sum(len(x) for x in self.nodes)

    @property
    def gpus_per_node(self) -> int:
        return len(self.nodes[0])

    def node_of(self, rank: int) -> int:
        if rank in self.nodes[0]:
            return 0
        if rank in self.nodes[1]:
            return 1
        raise KeyError(rank)

    @classmethod
    def builtin(
        cls,
        gpus_per_node: int = 4,
        chunk_bytes: int = 16 * 1024 * 1024,
        intra_gbps: float = 400.0,
        inter_gbps: float = 100.0,
        intra_alpha_ns: float = 0.0,
        inter_alpha_ns: float = 0.0,
    ) -> "Topology":
        if gpus_per_node < 1:
            raise ValueError("gpus_per_node must be >= 1")
        a = tuple(range(gpus_per_node))
        b = tuple(range(gpus_per_node, 2 * gpus_per_node))
        return cls(
            (a, b),
            Link(intra_gbps, intra_alpha_ns),
            Link(inter_gbps, inter_alpha_ns),
            chunk_bytes,
        )


_RATE_RE = re.compile(
    r"^([0-9]+(?:\.[0-9]+)?)(Tbps|Gbps|Mbps|Kbps)$",
    re.IGNORECASE,
)


def _rate_gbps(token: str) -> float:
    m = _RATE_RE.match(token)
    if not m:
        raise ValueError(f"bad rate token: {token}")
    scale = {"tbps": 1000.0, "gbps": 1.0, "mbps": 1e-3, "kbps": 1e-6}
    return float(m.group(1)) * scale[m.group(2).lower()]


def read_ns3_topology(
    path: str | Path,
    chunk_bytes: int,
    intra_alpha_ns: float = 0.0,
    inter_alpha_ns: float = 0.0,
) -> Topology:
    """Read the project's 2-switch ASTRA/ns-3 physical topology file."""
    lines = [
        x.strip()
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    if len(lines) < 3:
        raise ValueError("topology file is too short")

    node_num, switch_num, link_num = map(int, lines[0].split()[:3])
    if switch_num != 2:
        raise ValueError(f"expected 2 switches, got {switch_num}")
    switches = tuple(map(int, lines[1].split()[:switch_num]))
    switch_set = set(switches)
    nranks = node_num - switch_num
    ranks = set(range(nranks))

    host_switch: Dict[int, int] = {}
    intra_rates: List[float] = []
    inter_rates: List[float] = []
    parsed_edges = 0

    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        rates = [p for p in parts if _RATE_RE.match(p)]
        if len(rates) != 1:
            raise ValueError(f"expected one bandwidth token: {line!r}")
        bw = _rate_gbps(rates[0])
        parsed_edges += 1

        if u in switch_set and v in switch_set:
            inter_rates.append(bw)
        elif u in ranks and v in switch_set:
            host_switch[u] = v
            intra_rates.append(bw)
        elif v in ranks and u in switch_set:
            host_switch[v] = u
            intra_rates.append(bw)
        else:
            raise ValueError(f"unexpected edge: {line!r}")

    if parsed_edges != link_num:
        raise ValueError(f"expected {link_num} links, parsed {parsed_edges}")
    if set(host_switch) != ranks:
        raise ValueError(f"unattached ranks: {sorted(ranks - set(host_switch))}")
    if len(inter_rates) != 1:
        raise ValueError(f"expected one switch-switch link, got {len(inter_rates)}")

    groups = tuple(
        tuple(sorted(r for r, sw in host_switch.items() if sw == switch))
        for switch in switches
    )
    if len(groups[0]) != len(groups[1]):
        raise ValueError(f"unequal node sizes: {groups}")

    return Topology(
        (groups[0], groups[1]),
        Link(min(intra_rates), intra_alpha_ns),
        Link(inter_rates[0], inter_alpha_ns),
        chunk_bytes,
    )


# -----------------------------------------------------------------------------
# Schedule objects
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Transfer:
    src: int
    dst: int
    chunks: Tuple[int, ...]
    kind: str


@dataclass
class Group:
    transfers: List[Transfer] = field(default_factory=list)
    inter_budget_ns: float = 0.0
    intra_time_ns: float = 0.0
    note: str = ""


@dataclass
class InterJob:
    a: int
    b: int
    a_chunks: Tuple[int, ...]
    b_chunks: Tuple[int, ...]
    done: bool = False

    @property
    def weight(self) -> int:
        return max(len(self.a_chunks), len(self.b_chunks))


# -----------------------------------------------------------------------------
# Greedy scheduler
# -----------------------------------------------------------------------------


class BAIScheduler:
    def __init__(
        self,
        topo: Topology,
        reserve_local_gpus: int = 1,
        max_intra_rounds_per_inter: int = 1,
    ) -> None:
        self.topo = topo
        m = topo.gpus_per_node
        if not 0 <= reserve_local_gpus < m:
            raise ValueError("reserve_local_gpus must satisfy 0 <= reserve < GPUs/node")
        if max_intra_rounds_per_inter < 0:
            raise ValueError("max_intra_rounds_per_inter must be >= 0")

        self.reserve = reserve_local_gpus
        self.max_overlap_rounds = max_intra_rounds_per_inter
        self.owned: Dict[int, Set[int]] = {r: {r} for r in range(topo.nranks)}
        self.groups: List[Group] = []
        self.jobs = self._make_inter_jobs()

        # node.find_inter(): chunks that have not yet crossed into each node.
        self.cut_missing = {
            0: set(topo.nodes[1]),
            1: set(topo.nodes[0]),
        }
        self.gateway_need: Dict[int, Set[int]] = {}
        for j in self.jobs:
            self.gateway_need[j.a] = set(j.a_chunks)
            self.gateway_need[j.b] = set(j.b_chunks)

    def _make_inter_jobs(self) -> List[InterJob]:
        a, b = self.topo.nodes
        r = self.reserve
        reserved_a, gateways_a = list(a[:r]), list(a[r:])
        reserved_b, gateways_b = list(b[:r]), list(b[r:])

        assign_a = {g: [g] for g in gateways_a}
        assign_b = {g: [g] for g in gateways_b}
        k = len(gateways_a)

        # Reserved-origin chunks are pre-positioned onto gateways. For the
        # target 4-GPU node with reserve=1 this produces bundle sizes 1,1,2.
        for i, c in enumerate(reserved_a):
            assign_a[gateways_a[-1 - i % k]].append(c)
        for i, c in enumerate(reserved_b):
            assign_b[gateways_b[-1 - i % k]].append(c)

        return [
            InterJob(ga, gb, tuple(assign_a[ga]), tuple(assign_b[gb]))
            for ga, gb in zip(gateways_a, gateways_b)
        ]

    def _job_ready(self, j: InterJob) -> bool:
        return set(j.a_chunks) <= self.owned[j.a] and set(j.b_chunks) <= self.owned[j.b]

    def _choose_inter(self) -> Optional[InterJob]:
        # nodeset.get_top(): in a symmetric two-node AllGather this is usually a
        # tie, but keep the policy explicit for intermediate states.
        target = max((0, 1), key=lambda n: (len(self.cut_missing[n]), -n))
        ready = [j for j in self.jobs if not j.done and self._job_ready(j)]
        if not ready:
            return None

        def useful(j: InterJob) -> int:
            if target == 0:
                return len(set(j.b_chunks) & self.cut_missing[0])
            return len(set(j.a_chunks) & self.cut_missing[1])

        # Small/root-ready inter bundles first, aggregate bundle later.
        return min(ready, key=lambda j: (j.weight, -useful(j), j.a, j.b))

    @staticmethod
    def _pair_transfers(a: int, b: int, state: Dict[int, Set[int]]) -> List[Transfer]:
        a_to_b = tuple(sorted(state[a] - state[b]))
        b_to_a = tuple(sorted(state[b] - state[a]))
        out: List[Transfer] = []
        if a_to_b:
            out.append(Transfer(a, b, a_to_b, "intra"))
        if b_to_a:
            out.append(Transfer(b, a, b_to_a, "intra"))
        return out

    def _pair_score(
        self,
        a: int,
        b: int,
        state: Dict[int, Set[int]],
    ) -> Tuple[int, int, int]:
        a_to_b = state[a] - state[b]
        b_to_a = state[b] - state[a]
        gain = len(a_to_b) + len(b_to_a)
        prep = 0
        if b in self.gateway_need:
            prep += len(a_to_b & (self.gateway_need[b] - state[b]))
        if a in self.gateway_need:
            prep += len(b_to_a & (self.gateway_need[a] - state[a]))
        balance = -abs(len(a_to_b) - len(b_to_a))
        return prep, gain, balance

    def _best_matching(
        self,
        ranks: Sequence[int],
        state: Dict[int, Set[int]],
        budget_ns: Optional[float],
    ) -> Tuple[List[Transfer], float]:
        ranks = tuple(sorted(ranks))
        pair_info = {}

        for i, a in enumerate(ranks):
            for b in ranks[i + 1 :]:
                ts = self._pair_transfers(a, b, state)
                if not ts:
                    continue
                a_to_b = sum(len(t.chunks) for t in ts if t.src == a)
                b_to_a = sum(len(t.chunks) for t in ts if t.src == b)
                duration = self.topo.intra.exchange_ns(
                    a_to_b, b_to_a, self.topo.chunk_bytes
                )
                if budget_ns is not None and duration > budget_ns + 1e-9:
                    continue
                pair_info[(a, b)] = (self._pair_score(a, b, state), ts, duration)

        best_key = None
        best_ts: List[Transfer] = []
        best_duration = 0.0

        def rec(remain: Tuple[int, ...], chosen: List[Tuple[int, int]]) -> None:
            nonlocal best_key, best_ts, best_duration
            if not remain:
                prep = gain = balance = 0
                duration = 0.0
                ts: List[Transfer] = []
                for pair in chosen:
                    score, pts, d = pair_info[pair]
                    prep += score[0]
                    gain += score[1]
                    balance += score[2]
                    duration = max(duration, d)
                    ts.extend(pts)
                key = (prep, gain, balance, len(chosen), tuple(chosen))
                if best_key is None or key > best_key:
                    best_key, best_ts, best_duration = key, ts, duration
                return

            first = remain[0]
            rec(remain[1:], chosen)
            for i in range(1, len(remain)):
                second = remain[i]
                pair = (min(first, second), max(first, second))
                if pair not in pair_info:
                    continue
                rest = remain[1:i] + remain[i + 1 :]
                rec(rest, chosen + [pair])

        rec(ranks, [])
        return best_ts, best_duration

    @staticmethod
    def _apply_round(state: Dict[int, Set[int]], transfers: Sequence[Transfer]) -> None:
        before = {r: set(v) for r, v in state.items()}
        add = {r: set() for r in state}
        seen = set()

        for t in transfers:
            cs = set(t.chunks)
            if not cs <= before[t.src]:
                raise RuntimeError(f"{t.src}->{t.dst}: source missing {sorted(cs-before[t.src])}")
            if cs & before[t.dst]:
                raise RuntimeError(f"{t.src}->{t.dst}: redundant {sorted(cs & before[t.dst])}")
            for c in cs:
                if (t.dst, c) in seen:
                    raise RuntimeError(f"duplicate round delivery: rank={t.dst}, chunk={c}")
                seen.add((t.dst, c))
            add[t.dst].update(cs)

        for r in state:
            state[r].update(add[r])

    def _prepare_gateway(self) -> None:
        before = {r: set(v) for r, v in self.owned.items()}
        transfers: List[Transfer] = []
        duration = 0.0
        for node in self.topo.nodes:
            ts, d = self._best_matching(node, before, None)
            transfers.extend(ts)
            duration = max(duration, d)
        if not transfers:
            raise RuntimeError("scheduler stalled while preparing an inter gateway")
        self._apply_round(self.owned, transfers)
        self.groups.append(Group(transfers, 0.0, duration, "gateway-preparation"))

    def _schedule_inter(self, j: InterJob) -> None:
        inter = [
            Transfer(j.a, j.b, j.a_chunks, "inter"),
            Transfer(j.b, j.a, j.b_chunks, "inter"),
        ]
        budget = self.topo.inter.exchange_ns(
            len(j.a_chunks), len(j.b_chunks), self.topo.chunk_bytes
        )

        # Local rounds are planned without adding the current inter arrivals to
        # temp, so their data dependencies are independent of this inter job.
        temp = {r: set(v) for r, v in self.owned.items()}
        blocked = {j.a, j.b}
        local_rounds: List[Tuple[List[Transfer], float]] = []
        elapsed = 0.0

        while elapsed + 1e-9 < budget:
            if self.max_overlap_rounds and len(local_rounds) >= self.max_overlap_rounds:
                break
            before = {r: set(v) for r, v in temp.items()}
            remaining = budget - elapsed
            round_ts: List[Transfer] = []
            round_time = 0.0

            for node in self.topo.nodes:
                idle = [r for r in node if r not in blocked]
                ts, d = self._best_matching(idle, before, remaining)
                round_ts.extend(ts)
                round_time = max(round_time, d)

            if not round_ts or round_time <= 0:
                break
            self._apply_round(temp, round_ts)
            local_rounds.append((round_ts, round_time))
            elapsed += round_time

        first_local = local_rounds[0][0] if local_rounds else []
        first_time = local_rounds[0][1] if local_rounds else 0.0
        self._apply_round(self.owned, inter + first_local)
        self.groups.append(
            Group(inter + first_local, budget, first_time, f"inter {j.a}<->{j.b}")
        )

        # Extra overlap rounds are construction groups only; emitted DSL is flat.
        for idx, (ts, d) in enumerate(local_rounds[1:], start=1):
            self._apply_round(self.owned, ts)
            self.groups.append(Group(ts, budget, d, f"same inter window, local round {idx}"))

        j.done = True
        self.cut_missing[1].difference_update(j.a_chunks)
        self.cut_missing[0].difference_update(j.b_chunks)

    def _finish_intra(self) -> None:
        target = set(range(self.topo.nranks))
        while any(self.owned[r] != target for r in range(self.topo.nranks)):
            before = {r: set(v) for r, v in self.owned.items()}
            transfers: List[Transfer] = []
            duration = 0.0
            for node in self.topo.nodes:
                ts, d = self._best_matching(node, before, None)
                transfers.extend(ts)
                duration = max(duration, d)
            if not transfers:
                raise RuntimeError("intra completion stalled")
            self._apply_round(self.owned, transfers)
            self.groups.append(Group(transfers, 0.0, duration, "intra-finish"))

    def build(self) -> List[Group]:
        while not all(j.done for j in self.jobs):
            j = self._choose_inter()
            if j is None:
                self._prepare_gateway()
            else:
                self._schedule_inter(j)
        if self.cut_missing[0] or self.cut_missing[1]:
            raise RuntimeError(f"cut dissemination incomplete: {self.cut_missing}")
        self._finish_intra()
        self.verify()
        return self.groups

    def flat(self) -> List[Transfer]:
        return [t for g in self.groups for t in g.transfers]

    def verify(self) -> Dict[str, int]:
        state = {r: {r} for r in range(self.topo.nranks)}
        total = inter = 0
        for step, t in enumerate(self.flat()):
            cs = set(t.chunks)
            if not cs <= state[t.src]:
                raise AssertionError((step, t, "source unavailable"))
            if cs & state[t.dst]:
                raise AssertionError((step, t, "duplicate delivery"))
            state[t.dst].update(cs)
            total += len(cs)
            if self.topo.node_of(t.src) != self.topo.node_of(t.dst):
                inter += len(cs)

        expected = set(range(self.topo.nranks))
        if any(state[r] != expected for r in state):
            raise AssertionError("AllGather incomplete")

        n = self.topo.nranks
        if total != n * (n - 1):
            raise AssertionError(("total", total, n * (n - 1)))
        if inter != n:
            raise AssertionError(("inter", inter, n))
        return {
            "groups": len(self.groups),
            "total": total,
            "inter": inter,
            "intra": total - inter,
        }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def show_schedule(groups: Sequence[Group]) -> None:
    for i, g in enumerate(groups):
        intra = sum(len(t.chunks) for t in g.transfers if t.kind == "intra")
        inter = sum(len(t.chunks) for t in g.transfers if t.kind == "inter")
        print(
            f"group {i}: intra={intra} inter={inter} "
            f"inter_budget_ns={g.inter_budget_ns:.3f} "
            f"intra_time_ns={g.intra_time_ns:.3f}  # {g.note}",
            file=sys.stderr,
        )
        for t in g.transfers:
            print(
                f"  T({t.src}, {t.dst}, {list(t.chunks)})  # {t.kind}",
                file=sys.stderr,
            )


def render_dsl(topo: Topology, schedule: Sequence[Transfer], name: str) -> str:
    rows = "\n".join(f"    T({t.src}, {t.dst}, {list(t.chunks)})," for t in schedule)
    return f'''#!/usr/bin/env python3
import argparse
import sys
from msccl.language import *
from msccl.language.collectives import AllGather
from msccl.topologies import fully_connected

NUM_RANKS = {topo.nranks}
NODE0 = {topo.nodes[0]!r}
NODE1 = {topo.nodes[1]!r}


def T(src, dst, chunks):
    return src, dst, list(chunks)


SCHEDULE = [
{rows}
]


def node_of(rank):
    return 0 if rank in NODE0 else 1


def verify_schedule(show=False):
    state = {{r: {{r}} for r in range(NUM_RANKS)}}
    total = inter = 0
    for step, (src, dst, chunks) in enumerate(SCHEDULE):
        cs = set(chunks)
        if not cs <= state[src]:
            raise RuntimeError(f"step {{step}}: source {{src}} lacks {{sorted(cs-state[src])}}")
        if cs & state[dst]:
            raise RuntimeError(f"step {{step}}: redundant {{sorted(cs & state[dst])}}")
        state[dst].update(cs)
        total += len(chunks)
        is_inter = node_of(src) != node_of(dst)
        inter += len(chunks) if is_inter else 0
        if show:
            kind = "INTER" if is_inter else "LOCAL"
            print(f"step {{step:02d}}: {{src}} -> {{dst}} {{chunks}} {{kind}}", file=sys.stderr)

    expected = set(range(NUM_RANKS))
    for r in range(NUM_RANKS):
        if state[r] != expected:
            raise RuntimeError(f"rank {{r}} incomplete: {{sorted(state[r])}}")
    if total != NUM_RANKS * (NUM_RANKS - 1) or inter != NUM_RANKS:
        raise RuntimeError(f"bad counts: total={{total}}, inter={{inter}}")
    if show:
        print(f"total={{total}} inter={{inter}} local={{total-inter}} PASS", file=sys.stderr)


def emit(src, dst, c):
    ref = chunk(src, Buffer.output, c, 1)
    if ref is None:
        raise RuntimeError(f"cannot find chunk {{c}} on rank {{src}}")
    ref.copy(dst, Buffer.output, c, 1, ch=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("instances", type=int)
    p.add_argument("--protocol", default="Simple", choices=["Simple", "LL", "LL128"])
    p.add_argument("--show-schedule", action="store_true")
    args = p.parse_args()

    verify_schedule(args.show_schedule)
    topology = fully_connected(NUM_RANKS)
    collective = AllGather(NUM_RANKS, 1, True)
    program = MSCCLProgram(
        {name!r}, topology, collective, args.instances,
        protocol=args.protocol,
        dependence_nop=True,
        instr_fusion=False,
    )
    with program:
        for src, dst, chunks in SCHEDULE:
            for c in chunks:
                emit(src, dst, c)
        Check()
        XML()


if __name__ == "__main__":
    main()
'''


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ns3-topology")
    p.add_argument("--gpus-per-node", type=int, default=4)
    p.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    p.add_argument("--intra-gbps", type=float, default=400.0)
    p.add_argument("--inter-gbps", type=float, default=100.0)
    p.add_argument("--intra-alpha-ns", type=float, default=0.0)
    p.add_argument("--inter-alpha-ns", type=float, default=0.0)
    p.add_argument("--reserve-local-gpus", type=int, default=1)
    p.add_argument(
        "--max-intra-rounds-per-inter",
        type=int,
        default=1,
        help="0 = no round-count cap; alpha-beta budget still limits packing",
    )
    p.add_argument("--show-schedule", action="store_true")
    p.add_argument("--freeze-dsl", type=Path)
    p.add_argument("--program-name", default="bai_greedy_generated")
    args = p.parse_args()

    if args.ns3_topology:
        topo = read_ns3_topology(
            args.ns3_topology,
            args.chunk_bytes,
            args.intra_alpha_ns,
            args.inter_alpha_ns,
        )
    else:
        topo = Topology.builtin(
            args.gpus_per_node,
            args.chunk_bytes,
            args.intra_gbps,
            args.inter_gbps,
            args.intra_alpha_ns,
            args.inter_alpha_ns,
        )

    scheduler = BAIScheduler(
        topo,
        args.reserve_local_gpus,
        args.max_intra_rounds_per_inter,
    )
    groups = scheduler.build()
    stats = scheduler.verify()
    shape = [
        (
            sum(len(t.chunks) for t in g.transfers if t.kind == "intra"),
            sum(len(t.chunks) for t in g.transfers if t.kind == "inter"),
        )
        for g in groups
    ]

    print(
        f"nodes={topo.nodes} chunk_bytes={topo.chunk_bytes} "
        f"intra={topo.intra.bandwidth_gbps}Gbps "
        f"inter={topo.inter.bandwidth_gbps}Gbps",
        file=sys.stderr,
    )
    print(f"stats={stats}", file=sys.stderr)
    print(f"group_shape(intra,inter)={shape}", file=sys.stderr)

    if args.show_schedule:
        show_schedule(groups)
    if args.freeze_dsl:
        args.freeze_dsl.write_text(
            render_dsl(topo, scheduler.flat(), args.program_name),
            encoding="utf-8",
        )
        print(f"frozen_dsl={args.freeze_dsl}", file=sys.stderr)


if __name__ == "__main__":
    main()
