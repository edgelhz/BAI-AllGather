#!/usr/bin/env python3
"""Multi-node extension of BAI V2-RemoteCore.

Compatibility rule
------------------
For exactly two physical nodes this program calls the UNMODIFIED legacy
bai_v2_remote_core.build_and_report() and bai_scheduler_v1.render_dsl().
Thus the generated ordered transfers, their MSCCLang dependency construction,
and the DSL text are the same as those of V2-RemoteCore for matching inputs.

For K >= 3 it schedules exactly one logical copy of every origin chunk into
each other node, interleaves one inter-node GPU-pair exchange with a local
RemoteCore matching window, and completes distribution with missing-only RD.
Physical link utilization/overlap remains a simulator-measured question.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import sys
from typing import Any

import bai_scheduler_v1 as base
from bai_inter_first_rd_scheduler_v2 import InterFirstRD
from bai_v2_remote_core import RemoteCoreV2, build_and_report


def digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MultiTopology:
    """Equal-size nodes, with an ns-3 switch graph or nominal single inter BW."""
    nodes: tuple[tuple[int, ...], ...]
    intra: base.Link
    inter: base.Link
    chunk_bytes: int
    pair_bandwidths: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        k = len(self.nodes)
        if k < 3:
            raise ValueError("MultiTopology is only for >= 3 nodes")
        sizes = {len(group) for group in self.nodes}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError("all nodes must have an equal positive number of GPUs")
        n = sum(map(len, self.nodes))
        if sorted(r for group in self.nodes for r in group) != list(range(n)):
            raise ValueError("ranks must be exactly 0..N-1")
        if self.chunk_bytes <= 0:
            raise ValueError("chunk-bytes must be positive")
        if self.intra.bandwidth_gbps <= 0 or self.inter.bandwidth_gbps <= 0:
            raise ValueError("all bandwidths must be positive")
        if len(self.pair_bandwidths) != k or any(len(row) != k for row in self.pair_bandwidths):
            raise ValueError("pair-bandwidth matrix has wrong dimensions")
        for i in range(k):
            for j in range(k):
                if i != j and (self.pair_bandwidths[i][j] <= 0 or
                               self.pair_bandwidths[i][j] != self.pair_bandwidths[j][i]):
                    raise ValueError("pair bandwidth must be positive and symmetric")
        rank_to_node = {}
        for i, group in enumerate(self.nodes):
            for r in group:
                rank_to_node[r] = i
        object.__setattr__(self, "_rank_to_node", rank_to_node)

    @property
    def nranks(self) -> int:
        return sum(len(g) for g in self.nodes)

    @property
    def gpus_per_node(self) -> int:
        return len(self.nodes[0])

    def node_of(self, rank: int) -> int:
        return self._rank_to_node[rank]

    def inter_exchange_ns(self, i: int, j: int, ca: int, cb: int) -> float:
        path_link = base.Link(self.pair_bandwidths[i][j], self.inter.alpha_ns)
        return path_link.exchange_ns(ca, cb, self.chunk_bytes)

    @classmethod
    def builtin(cls, k: int, m: int, chunk_bytes: int,
                intra_gbps: float, inter_gbps: float,
                intra_alpha_ns: float, inter_alpha_ns: float) -> "MultiTopology":
        if k < 3 or m < 1:
            raise ValueError("need >= 3 nodes and >= 1 GPU per node")
        if any(v <= 0 for v in (chunk_bytes, intra_gbps, inter_gbps)):
            raise ValueError("chunk size and link rates must be positive")
        if min(intra_alpha_ns, inter_alpha_ns) < 0:
            raise ValueError("alpha must be nonnegative")
        groups = tuple(tuple(range(i*m, (i+1)*m)) for i in range(k))
        rates = tuple(tuple(inter_gbps if i != j else 0. for j in range(k))
                      for i in range(k))
        return cls(groups, base.Link(intra_gbps, intra_alpha_ns),
                   base.Link(inter_gbps, inter_alpha_ns), chunk_bytes, rates)


def read_multinode_ns3(path: Path, chunk_bytes: int,
                       intra_alpha_ns: float, inter_alpha_ns: float) -> MultiTopology:
    """Parse one-switch-per-node ns-3 physical topology, all connected graphs.

    Pair capacity estimate is the widest-path bottleneck rate, not a measured
    multi-flow share of the physical links. ns-3 handles actual routing.
    """
    lines = [x.strip() for x in path.read_text().splitlines()
             if x.strip() and not x.lstrip().startswith("#")]
    if len(lines) < 3:
        raise ValueError("ns-3 topology is empty or too short")
    head = lines[0].split()
    if len(head) != 3:
        raise ValueError("expected: total_devices switch_count edge_count")
    total, k, count = map(int, head)
    if not (total > k >= 3):
        raise ValueError("need >=3 physical switches and attached GPUs")
    switches = tuple(map(int, lines[1].split()))
    if len(switches) != k or len(set(switches)) != k:
        raise ValueError("switch declaration does not match switch count")
    sw_set = set(switches)
    n = total - k
    rank_set = set(range(n))
    if rank_set & sw_set:
        raise ValueError("switch IDs collide with GPU rank IDs")
    host_sw: dict[int, int] = {}
    host_rates = []
    graph: dict[int, dict[int, float]] = {sw: {} for sw in switches}
    edges = 0
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"malformed link: {line!r}")
        u, v = int(parts[0]), int(parts[1])
        rates = [p for p in parts if base._RATE_RE.fullmatch(p)]
        if len(rates) != 1:
            raise ValueError(f"link must have one bandwidth token: {line!r}")
        bw = base._rate_gbps(rates[0])
        if bw <= 0:
            raise ValueError("link bandwidth must be positive")
        edges += 1
        if u in sw_set and v in sw_set:
            if u == v or v in graph[u]:
                raise ValueError("duplicate or self-connected switch edge")
            graph[u][v] = bw
            graph[v][u] = bw
        elif u in rank_set and v in sw_set or v in rank_set and u in sw_set:
            rank, sw = (u, v) if u in rank_set else (v, u)
            if rank in host_sw:
                raise ValueError("each rank must attach to exactly one switch")
            host_sw[rank] = sw
            host_rates.append(bw)
        else:
            raise ValueError(f"unknown physical edge: {line!r}")
    if edges != count or set(host_sw) != rank_set:
        raise ValueError("edge count or GPU attachment count mismatch")
    groups = tuple(tuple(sorted(r for r, sw in host_sw.items() if sw == s))
                   for s in switches)
    if len({len(g) for g in groups}) != 1 or not groups[0]:
        raise ValueError("GPU counts per physical switch must be equal and nonzero")

    def widest(source: int) -> dict[int, float]:
        known = {source: float("inf")}
        heap = [(-float("inf"), source)]
        while heap:
            minus_rate, u = heapq.heappop(heap)
            cur = -minus_rate
            if cur < known[u]:
                continue
            for v, rate in graph[u].items():
                candidate = min(cur, rate)
                if candidate > known.get(v, -1.):
                    known[v] = candidate
                    heapq.heappush(heap, (-candidate, v))
        return known

    matrix = []
    for sw in switches:
        best = widest(sw)
        if set(best) != sw_set:
            raise ValueError("physical switch topology is disconnected")
        matrix.append(tuple(0. if sw == peer else best[peer]
                            for peer in switches))
    return MultiTopology(
        groups,
        base.Link(min(host_rates), intra_alpha_ns),
        base.Link(min(edge for row in graph.values() for edge in row.values()), inter_alpha_ns),
        chunk_bytes,
        tuple(matrix),
    )


@dataclass(frozen=True)
class MultiJob:
    node_a: int
    node_b: int
    a: int
    b: int
    ca: tuple[int, ...]
    cb: tuple[int, ...]
    ordinal: int

    @property
    def weight(self) -> int:
        return max(len(self.ca), len(self.cb))


def round_robin_pairs(k: int) -> list[tuple[int, int]]:
    """Round-robin node-pair order (nodes in one round are disjoint)."""
    players = list(range(k))
    if k % 2:
        players.append(None)
    out = []
    seen = set()
    for _ in range(len(players) - 1):
        for i in range(len(players)//2):
            a, b = players[i], players[-i-1]
            if a is None or b is None:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                raise AssertionError("duplicate node pairing")
            seen.add(pair)
            out.append(pair)
        players = [players[0], players[-1], *players[1:-1]]
    if len(seen) != k*(k-1)//2:
        raise AssertionError("incomplete round-robin node pair schedule")
    return out


class MultiNodeRemoteCore:
    """Preserve V2's per-node matching rule for a generalized K-node system."""

    # These methods are the EXACT original V2 / RemoteCore methods. The code is
    # shared, not paraphrased or reimplemented, so scoring/tie breaks persist.
    duplicate = staticmethod(InterFirstRD.duplicate)
    apply = InterFirstRD.apply
    pair = staticmethod(InterFirstRD.pair)
    prep_value = InterFirstRD.prep_value
    local_round = RemoteCoreV2.local_round
    finish_rd = InterFirstRD.finish_rd

    def __init__(self, topo: MultiTopology, max_rounds: int = 1):
        if max_rounds < 0:
            raise ValueError("max-local-rounds must be nonnegative")
        self.topo = topo
        self.max_rounds = max_rounds
        self.state = {r: {r} for r in range(topo.nranks)}
        self.groups: list[base.Group] = []
        self.hot: dict[int, set[int]] = {}
        self.remote_chunks = {
            rank: frozenset(set(range(topo.nranks)) - set(topo.nodes[topo.node_of(rank)]))
            for rank in range(topo.nranks)
        }
        self.local_matching_rounds = 0
        self.selected_foreign_chunks = 0
        self.core_selections = []

        jobs = []
        for a, b in round_robin_pairs(len(topo.nodes)):
            ga, gb = topo.nodes[a], topo.nodes[b]
            m = topo.gpus_per_node
            if m < 3:
                specs = [(i, (ga[i],), (gb[i],)) for i in range(m)]
            else:
                specs = [(i, (ga[i],), (gb[i],)) for i in range(1, m-1)]
                specs.append((m-1, (ga[m-1], ga[0]), (gb[m-1], gb[0])))
            for offset, ac, bc in specs:
                jobs.append(MultiJob(a, b, ga[offset], gb[offset], ac, bc, len(jobs)))
        self.jobs = tuple(jobs)
        self.pending = list(jobs)

    @staticmethod
    def _key(job: MultiJob) -> tuple[int, int]:
        return job.weight, job.ordinal

    def inter_ready(self, job: MultiJob) -> bool:
        return set(job.ca) <= self.state[job.a] and set(job.cb) <= self.state[job.b]

    def prepare_fallback(self) -> None:
        """Prepare only one pending gateway pair, avoiding endpoint conflicts."""
        job = min(self.pending, key=self._key)
        transfers = []
        for dst, needed in ((job.a, job.ca), (job.b, job.cb)):
            missing = sorted(set(needed) - self.state[dst])
            for chunk in missing:
                src = next((r for r in self.topo.nodes[self.topo.node_of(dst)]
                            if r != dst and chunk in self.state[r]), None)
                if src is None:
                    raise RuntimeError(f"cannot prepare chunk {chunk} for gateway {dst}")
                transfers.append(base.Transfer(src, dst, (chunk,), 'intra'))
        if not transfers:
            raise RuntimeError("stalled without a missing gateway chunk")
        self.apply(self.state, transfers)
        self.groups.append(base.Group(transfers, 0., 0., "multinode gateway preparation"))

    def inter_window(self, job: MultiJob) -> None:
        cross = [base.Transfer(job.a, job.b, job.ca, 'inter'),
                 base.Transfer(job.b, job.a, job.cb, 'inter')]
        budget = self.topo.inter_exchange_ns(job.node_a, job.node_b,
                                             len(job.ca), len(job.cb))
        future = sorted((x for x in self.pending if x != job), key=self._key)
        next_busy = {future[0].a, future[0].b} if future else set()
        tmp = self.duplicate(self.state)
        blocked = {job.a, job.b}
        rounds: list[tuple[list[base.Transfer], float]] = []
        elapsed = 0.
        while elapsed + 1e-6 < budget and (
            self.max_rounds == 0 or len(rounds) < self.max_rounds
        ):
            intra, duration = self.local_round(tmp, blocked, budget - elapsed, next_busy)
            if not intra or duration <= 0:
                break
            self.apply(tmp, intra)
            rounds.append((intra, duration))
            elapsed += duration
        first, first_ns = rounds[0] if rounds else ([], 0.)
        self.apply(self.state, cross + first)
        self.groups.append(base.Group(
            cross + first, budget, first_ns,
            f"multinode inter nodes {job.node_a}<->{job.node_b} gateways {job.a}<->{job.b}",
        ))
        for round_no, (local, ns) in enumerate(rounds[1:], 1):
            self.apply(self.state, local)
            self.groups.append(base.Group(local, budget, ns,
                                          f"multinode inter-window extra-local {round_no}"))
        self.pending.remove(job)
        self.hot = {job.a: set(job.cb), job.b: set(job.ca)}

    def build(self) -> list[base.Group]:
        while self.pending:
            ready = [job for job in self.pending if self.inter_ready(job)]
            if ready:
                self.inter_window(min(ready, key=self._key))
            else:
                self.prepare_fallback()
        self.finish_rd()
        self.verify()
        return self.groups

    def verify(self) -> dict[str, Any]:
        state = {r: {r} for r in range(self.topo.nranks)}
        inter = intra = 0
        cross_counts: Counter[tuple[int, int]] = Counter()
        by_pair: Counter[tuple[int, int]] = Counter()
        for g in self.groups:
            self.apply(state, g.transfers)
            for tr in g.transfers:
                if tr.kind not in ('inter', 'intra'):
                    raise AssertionError("invalid transfer kind")
                real_kind = 'inter' if self.topo.node_of(tr.src) != self.topo.node_of(tr.dst) else 'intra'
                if real_kind != tr.kind:
                    raise AssertionError("incorrect transfer type")
                if real_kind == 'inter':
                    inter += len(tr.chunks)
                    for c in tr.chunks:
                        cross_counts[(c, self.topo.node_of(tr.dst))] += 1
                        by_pair[(self.topo.node_of(tr.src), self.topo.node_of(tr.dst))] += 1
                else:
                    intra += len(tr.chunks)
        if state != self.state:
            raise AssertionError("recorded final state differs from replay")
        all_chunks = set(range(self.topo.nranks))
        if any(owned != all_chunks for owned in state.values()):
            raise AssertionError("AllGather incomplete")
        k, n = len(self.topo.nodes), self.topo.nranks
        expected = n*(k-1)
        if inter != expected or intra + inter != n*(n-1):
            raise AssertionError(f"wrong counts: inter={inter}, intra={intra}")
        if any(cross_counts[(c, node)] != 1
               for c in range(n) for node in range(k) if node != self.topo.node_of(c)):
            raise AssertionError("a chunk crossed into a node zero or multiple times")
        if len(cross_counts) != expected:
            raise AssertionError("unexpected cross-node chunk delivery")
        return {
            'total': intra + inter, 'inter': inter, 'intra': intra,
            'inter_windows': len(self.jobs), 'groups': len(self.groups),
            'unique_cross_destinations': len(cross_counts),
            'inter_by_node_pair': {f'{a}->{b}': cnt for (a,b),cnt in sorted(by_pair.items())},
        }


def render_multi_dsl(topo: MultiTopology, groups: list[base.Group], name: str) -> str:
    rows = '\n'.join(
        f'    ({tr.src}, {tr.dst}, {list(tr.chunks)}),'
        for g in groups for tr in g.transfers
    )
    ranks = [topo.node_of(r) for r in range(topo.nranks)]
    # A single schedule without artificial group barriers: dependency comes
    # from MSCCLang's chunk-copy read/write readiness, as in legacy V2.
    return f'''#!/usr/bin/env python3
import argparse
import sys
from msccl.language import *
from msccl.language.collectives import AllGather
from msccl.topologies import fully_connected

NUM_RANKS = {topo.nranks}
NUM_NODES = {len(topo.nodes)}
NODE_OF_RANK = {ranks!r}
SCHEDULE = [
{rows}
]

def verify_schedule(show=False):
    state = {{r: {{r}} for r in range(NUM_RANKS)}}
    inter = total = 0
    cross = set()
    for index, (src, dst, chunks) in enumerate(SCHEDULE):
        cs = set(chunks)
        if not cs or len(cs) != len(chunks) or not cs <= state[src]:
            raise RuntimeError(f"step {{index}} source missing chunk: {{src}}, {{chunks}}")
        if cs & state[dst]:
            raise RuntimeError(f"step {{index}} duplicated delivery at {{dst}}")
        state[dst].update(cs)
        total += len(chunks)
        if NODE_OF_RANK[src] != NODE_OF_RANK[dst]:
            inter += len(chunks)
            for c in chunks:
                key = (c, NODE_OF_RANK[dst])
                if key in cross:
                    raise RuntimeError(f"duplicate arrival into node: {{key}}")
                cross.add(key)
        if show:
            print(f"step {{index}}: {{src}} -> {{dst}} {{chunks}}", file=sys.stderr)
    if any(len(state[r]) != NUM_RANKS for r in range(NUM_RANKS)):
        raise RuntimeError("incomplete AllGather")
    if total != NUM_RANKS*(NUM_RANKS-1) or inter != NUM_RANKS*(NUM_NODES-1):
        raise RuntimeError(f"count mismatch total={{total}} inter={{inter}}")
    if len(cross) != inter:
        raise RuntimeError("bad cross-destination coverage")
    return {{'total':total, 'inter':inter, 'intra':total-inter}}


def emit(src, dst, c):
    ref = chunk(src, Buffer.output, c, 1)
    if ref is None:
        raise RuntimeError(f"source {{src}} lacks chunk {{c}}")
    ref.copy(dst, Buffer.output, c, 1, ch=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('instances', type=int)
    p.add_argument('--protocol', default='Simple', choices=['Simple','LL','LL128'])
    p.add_argument('--show-schedule', action='store_true')
    args = p.parse_args()
    verify_schedule(args.show_schedule)
    program = MSCCLProgram(
        {name!r}, fully_connected(NUM_RANKS),
        AllGather(NUM_RANKS, 1, True), args.instances,
        protocol=args.protocol, dependence_nop=True, instr_fusion=False,
    )
    with program:
        for src, dst, chunks in SCHEDULE:
            for c in chunks:
                emit(src, dst, c)
        Check()
        XML()

if __name__ == '__main__':
    main()
'''


def write_json(path: Path | None, content: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + '\n')


def cli(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nodes', type=int, default=2, help='number of physical GPU nodes >=2')
    ap.add_argument('--gpus-per-node', type=int, default=4)
    ap.add_argument('--ns3-topology', type=Path)
    ap.add_argument('--chunk-bytes', type=int, default=16777216)
    ap.add_argument('--intra-gbps', type=float, default=400.)
    ap.add_argument('--inter-gbps', type=float, default=100.)
    ap.add_argument('--intra-alpha-ns', type=float, default=0.)
    ap.add_argument('--inter-alpha-ns', type=float, default=0.)
    ap.add_argument('--max-local-rounds', type=int, default=1)
    ap.add_argument('--mode', choices=('core', 'baseline'), default='core')
    ap.add_argument('--freeze-dsl', type=Path, required=True)
    ap.add_argument('--report-json', type=Path)
    ap.add_argument('--reference-dsl', type=Path,
                    help='for 2 nodes, require byte-for-byte match with an original V2-RemoteCore DSL')
    ap.add_argument('--plan-json', type=Path)
    ns = ap.parse_args(argv)
    if ns.nodes < 2 or ns.gpus_per_node < 1 or ns.chunk_bytes <= 0:
        ap.error('nodes >=2, gpus-per-node >=1, chunk-bytes >0 required')
    if ns.max_local_rounds < 0:
        ap.error('max-local-rounds must be >=0')
    if ns.nodes > 2 and ns.mode != 'core':
        ap.error('baseline mode is only defined for the original two-node V2')
    if ns.nodes > 2 and ns.reference_dsl is not None:
        ap.error('--reference-dsl can only be used for exactly two nodes')
    if ns.ns3_topology:
        if ns.nodes == 2:
            topo = base.read_ns3_topology(ns.ns3_topology, ns.chunk_bytes,
                                          ns.intra_alpha_ns, ns.inter_alpha_ns)
        else:
            topo = read_multinode_ns3(ns.ns3_topology, ns.chunk_bytes,
                                      ns.intra_alpha_ns, ns.inter_alpha_ns)
        if len(topo.nodes) != ns.nodes or topo.gpus_per_node != ns.gpus_per_node:
            ap.error('requested nodes/GPUs do not match physical topology')
    elif ns.nodes == 2:
        topo = base.Topology.builtin(ns.gpus_per_node, ns.chunk_bytes,
                                     ns.intra_gbps, ns.inter_gbps,
                                     ns.intra_alpha_ns, ns.inter_alpha_ns)
    else:
        topo = MultiTopology.builtin(ns.nodes, ns.gpus_per_node,
                                     ns.chunk_bytes, ns.intra_gbps,
                                     ns.inter_gbps, ns.intra_alpha_ns, ns.inter_alpha_ns)

    if ns.nodes == 2:
        chosen, original, legacy = build_and_report(topo, ns.max_local_rounds, ns.mode)
        plan = chosen.groups
        dsl = base.render_dsl(topo, [t for g in plan for t in g.transfers],
                              'bai_inter_first_rd_v2')
        # Independent legacy construction: checks ordered transfers + group
        # boundaries and DSL equality, not just volume equivalence.
        legacy_again = (RemoteCoreV2(topo, ns.max_local_rounds)
                        if ns.mode == 'core' else InterFirstRD(topo, ns.max_local_rounds))
        legacy_again.build()
        old_signature = [
            (g.note, [(t.src, t.dst, t.chunks, t.kind) for t in g.transfers])
            for g in legacy_again.groups
        ]
        current_signature = [
            (g.note, [(t.src, t.dst, t.chunks, t.kind) for t in g.transfers])
            for g in plan
        ]
        if current_signature != old_signature:
            raise AssertionError('failed exact legacy schedule comparison')
        old_dsl = base.render_dsl(
            topo, [t for g in legacy_again.groups for t in g.transfers],
            'bai_inter_first_rd_v2'
        )
        if dsl != old_dsl:
            raise AssertionError('failed byte-identical legacy DSL comparison')
        report = dict(legacy)
        report['legacy_exact_equivalence'] = True
    else:
        planner = MultiNodeRemoteCore(topo, ns.max_local_rounds)
        plan = planner.build()
        audit = planner.verify()
        dsl = render_multi_dsl(topo, plan, 'bai_multinode_remote_core')
        report = dict(
            mode='core', nodes=ns.nodes, gpus_per_node=ns.gpus_per_node,
            nranks=topo.nranks, chunk_bytes=topo.chunk_bytes,
            max_local_rounds=ns.max_local_rounds,
            local_matching_rounds=planner.local_matching_rounds,
            new_foreign_chunks_during_local_rounds=planner.selected_foreign_chunks,
            legacy_exact_equivalence=None,
            **audit,
        )
    if ns.reference_dsl is not None and ns.reference_dsl.read_bytes() != dsl.encode('utf-8'):
        raise AssertionError('generated DSL differs from frozen legacy reference')
    compile(dsl, str(ns.freeze_dsl), 'exec')
    ns.freeze_dsl.parent.mkdir(parents=True, exist_ok=True)
    ns.freeze_dsl.write_text(dsl, encoding='utf-8')
    report.update(
        dsl_sha256=digest(dsl),
        group_count=len(plan),
        transfer_count=sum(len(g.transfers) for g in plan),
        group_shape=[[
            sum(len(t.chunks) for t in g.transfers if t.kind == 'intra'),
            sum(len(t.chunks) for t in g.transfers if t.kind == 'inter'),
        ] for g in plan],
    )
    write_json(ns.report_json, report)
    if ns.plan_json:
        write_json(ns.plan_json, {
            'nodes': len(topo.nodes), 'gpus_per_node': topo.gpus_per_node,
            'groups': [{
                'note': g.note,
                'inter_budget_ns': g.inter_budget_ns,
                'intra_time_ns': g.intra_time_ns,
                'transfers': [{'src': t.src, 'dst': t.dst,
                               'chunks': list(t.chunks), 'kind': t.kind}
                              for t in g.transfers],
            } for g in plan]
        })
    print('PLAN_AUDIT=PASS ' + json.dumps({
        'nodes': len(topo.nodes), 'gpus_per_node': topo.gpus_per_node,
        'total': report['total'], 'inter': report['inter'],
        'intra': report['intra'], 'groups': len(plan),
        'legacy_exact': report['legacy_exact_equivalence'],
        'dsl_sha256': report['dsl_sha256'],
    }, sort_keys=True), file=sys.stderr)
    return report


if __name__ == '__main__':
    cli()
