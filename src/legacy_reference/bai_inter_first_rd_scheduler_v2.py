
#!/usr/bin/env python3
"""Inter-first overlap -> missing-only recursive-doubling AllGather.

Depends on bai_scheduler_v1 only for the existing topology/alpha-beta model,
transfer dataclass, and proven-compatible MSCCLang DSL printer. No V1
scheduling or priority logic is reused.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import bai_scheduler_v1 as base


@dataclass(frozen=True)
class Inter:
    a: int
    b: int
    ca: tuple[int, ...]
    cb: tuple[int, ...]

    @property
    def weight(self):
        return max(len(self.ca), len(self.cb))


class InterFirstRD:
    def __init__(self, topo, max_rounds=1):
        if max_rounds < 0:
            raise ValueError('max_rounds must be nonnegative')
        self.topo, self.max_rounds = topo, max_rounds
        self.state = {r: {r} for r in range(topo.nranks)}
        self.groups = []
        self.hot = {}
        a, b = topo.nodes

        if topo.gpus_per_node < 3:
            self.jobs = [
                Inter(x, y, (x,), (y,))
                for x, y in zip(a, b)
            ]
        else:
            self.jobs = [
                Inter(x, y, (x,), (y,))
                for x, y in zip(a[1:], b[1:])
            ]
            last = self.jobs[-1]
            self.jobs[-1] = Inter(
                last.a, last.b,
                (last.a, a[0]),
                (last.b, b[0])
            )

        self.pending = list(self.jobs)

    @staticmethod
    def duplicate(s):
        return {r: set(v) for r, v in s.items()}

    def apply(self, state, transfers):
        old = self.duplicate(state)
        new = {r: set() for r in state}
        peers = {}

        for tr in transfers:
            cs = set(tr.chunks)

            if (
                not cs
                or len(cs) != len(tr.chunks)
                or not cs <= old[tr.src]
                or cs & (old[tr.dst] | new[tr.dst])
            ):
                raise AssertionError(f'bad transfer: {tr}')

            peers.setdefault(tr.src, set()).add(tr.dst)
            peers.setdefault(tr.dst, set()).add(tr.src)
            new[tr.dst].update(cs)

        if any(len(s) > 1 for s in peers.values()):
            raise AssertionError(
                'a GPU uses more than one peer in one round'
            )

        for r, cs in new.items():
            state[r].update(cs)

    @staticmethod
    def pair(a, b, state):
        xy = tuple(sorted(state[a] - state[b]))
        yx = tuple(sorted(state[b] - state[a]))
        result = []

        if xy:
            result.append(
                base.Transfer(a, b, xy, 'intra')
            )
        if yx:
            result.append(
                base.Transfer(b, a, yx, 'intra')
            )

        return result

    def prep_value(self, transfers, state):
        value = 0

        for tr in transfers:
            for j in self.pending:
                if tr.dst == j.a:
                    value += len(
                        set(tr.chunks)
                        & (set(j.ca) - state[j.a])
                    )
                if tr.dst == j.b:
                    value += len(
                        set(tr.chunks)
                        & (set(j.cb) - state[j.b])
                    )

        return value

    def local_round(self, state, busy, budget, next_busy):
        result, costs = [], []

        for node in self.topo.nodes:
            idle = [r for r in node if r not in busy]
            candidates = []

            for a, b in combinations(idle, 2):
                ts = self.pair(a, b, state)

                if not ts:
                    continue

                ab = sum(
                    len(x.chunks)
                    for x in ts if x.src == a
                )
                ba = sum(
                    len(x.chunks)
                    for x in ts if x.src == b
                )

                cost = self.topo.intra.exchange_ns(
                    ab, ba, self.topo.chunk_bytes
                )

                if cost > budget + 1e-6:
                    continue

                # Newly received inter-node chunks take priority.
                fresh = sum(
                    len(
                        set(x.chunks)
                        & self.hot.get(x.src, set())
                    )
                    for x in ts
                )

                # Priorities:
                # 1. Prepare data required by future inter jobs.
                # 2. Disseminate just-received inter chunks.
                # 3. Avoid occupying the next inter gateway.
                # 4. Maximize useful missing-chunk delivery.
                score = (
                    self.prep_value(ts, state),
                    fresh,
                    -(
                        int(a in next_busy)
                        + int(b in next_busy)
                    ),
                    ab + ba,
                    -cost,
                    -a,
                    -b
                )

                candidates.append(
                    (score, a, b, ts, cost)
                )

            used = set()

            for _, a, b, ts, cost in sorted(
                candidates, reverse=True
            ):
                if a not in used and b not in used:
                    used.update((a, b))
                    result.extend(ts)
                    costs.append(cost)

        return result, max(costs, default=0.0)

    def inter_ready(self, j):
        return (
            set(j.ca) <= self.state[j.a]
            and set(j.cb) <= self.state[j.b]
        )

    def prepare_fallback(self):
        # Used only when preparation cannot fit
        # inside an earlier inter-node window.
        t = []

        for j in self.pending:
            for dest, needed in (
                (j.a, j.ca),
                (j.b, j.cb)
            ):
                for chunk in needed:
                    if chunk in self.state[dest]:
                        continue

                    node = self.topo.nodes[
                        self.topo.node_of(dest)
                    ]
                    src = next(
                        (
                            r for r in node
                            if r != dest
                            and chunk in self.state[r]
                        ),
                        None
                    )

                    if src is None:
                        raise RuntimeError(
                            'no source for deferred gateway preparation'
                        )

                    t.append(
                        base.Transfer(
                            src, dest, (chunk,), 'intra'
                        )
                    )

        if not t:
            raise RuntimeError('inter scheduler stalled')

        self.apply(self.state, t)
        self.groups.append(
            base.Group(
                t, 0, 0,
                'stage1 prep fallback'
            )
        )

    def inter_window(self, j):
        across = [
            base.Transfer(
                j.a, j.b, j.ca, 'inter'
            ),
            base.Transfer(
                j.b, j.a, j.cb, 'inter'
            )
        ]

        budget = self.topo.inter.exchange_ns(
            len(j.ca),
            len(j.cb),
            self.topo.chunk_bytes
        )

        future = sorted(
            (x for x in self.pending if x != j),
            key=lambda x: (x.weight, x.a, x.b)
        )

        next_busy = (
            {future[0].a, future[0].b}
            if future else set()
        )

        blocked = {j.a, j.b}
        tmp = self.duplicate(self.state)

        rounds, elapsed = [], 0.0

        while (
            elapsed + 1e-6 < budget
            and (
                self.max_rounds == 0
                or len(rounds) < self.max_rounds
            )
        ):
            intra, duration = self.local_round(
                tmp,
                blocked,
                budget - elapsed,
                next_busy
            )

            if not intra or duration <= 0:
                break

            self.apply(tmp, intra)
            rounds.append((intra, duration))
            elapsed += duration

        first, first_ns = (
            rounds[0] if rounds else ([], 0.0)
        )

        # Inter and the first local round are independent.
        self.apply(self.state, across + first)

        self.groups.append(
            base.Group(
                across + first,
                budget,
                first_ns,
                f'stage1 inter {j.a}<->{j.b}'
            )
        )

        for k, (local, ns) in enumerate(
            rounds[1:], 1
        ):
            self.apply(self.state, local)
            self.groups.append(
                base.Group(
                    local,
                    budget,
                    ns,
                    f'stage1 inter-window extra-local {k}'
                )
            )

        self.pending.remove(j)

        # The endpoints of this inter job become
        # high-priority local disseminators next time.
        self.hot = {
            j.a: set(j.cb),
            j.b: set(j.ca)
        }

    def finish_rd(self):
        m = self.topo.gpus_per_node

        # Stage 2: local recursive doubling.
        # Only copy chunks missing at the receiver.
        for bit in range((m - 1).bit_length()):
            ts = []

            for node in self.topo.nodes:
                for i in range(m):
                    k = i ^ (1 << bit)

                    if i < k < m:
                        ts.extend(
                            self.pair(
                                node[i],
                                node[k],
                                self.state
                            )
                        )

            if ts:
                self.apply(self.state, ts)
                self.groups.append(
                    base.Group(
                        ts, 0, 0,
                        f'stage2 RD XOR {1 << bit}'
                    )
                )

        # Non-power-of-two node counts may require
        # additional deficit-only pairing rounds.
        all_chunks = set(range(self.topo.nranks))

        for index in range(2 * m * m):
            if all(
                x == all_chunks
                for x in self.state.values()
            ):
                return

            ts = []

            for node in self.topo.nodes:
                candidates = []

                for a, b in combinations(node, 2):
                    part = self.pair(a, b, self.state)
                    candidates.append(
                        (
                            sum(
                                len(t.chunks)
                                for t in part
                            ),
                            a, b, part
                        )
                    )

                used = set()

                for gain, a, b, part in sorted(
                    candidates, reverse=True
                ):
                    if (
                        gain
                        and a not in used
                        and b not in used
                    ):
                        used.update((a, b))
                        ts.extend(part)

            if not ts:
                raise RuntimeError(
                    'RD deficit repair stalled'
                )

            self.apply(self.state, ts)
            self.groups.append(
                base.Group(
                    ts, 0, 0,
                    f'stage2 RD repair {index}'
                )
            )

        raise RuntimeError('RD did not converge')

    def build(self):
        while self.pending:
            ready = [
                j for j in self.pending
                if self.inter_ready(j)
            ]

            if not ready:
                if len(self.pending) == len(self.jobs):
                    raise RuntimeError(
                        'cannot start with inter transfer'
                    )
                self.prepare_fallback()
                continue

            self.inter_window(
                min(
                    ready,
                    key=lambda j: (
                        j.weight,
                        j.a,
                        j.b
                    )
                )
            )

        self.finish_rd()
        self.verify()
        return self.groups

    def verify(self):
        state = {
            r: {r}
            for r in range(self.topo.nranks)
        }
        total = inter = 0

        for g in self.groups:
            self.apply(state, g.transfers)

            for t in g.transfers:
                total += len(t.chunks)

                if (
                    self.topo.node_of(t.src)
                    != self.topo.node_of(t.dst)
                ):
                    inter += len(t.chunks)

        n = self.topo.nranks

        if (
            state != self.state
            or any(
                len(x) != n
                for x in state.values()
            )
        ):
            raise AssertionError(
                'AllGather final state invalid'
            )

        if (total, inter) != (
            n * (n - 1), n
        ):
            raise AssertionError(
                f'bad cut/delivery counts: {total}/{inter}'
            )

        if not any(
            t.kind == 'inter'
            for t in self.groups[0].transfers
        ):
            raise AssertionError(
                'inter not scheduled first'
            )

        return {
            'total': total,
            'inter': inter,
            'intra': total - inter,
            'inter_windows': len(self.jobs),
            'groups': len(self.groups)
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ns3-topology', type=Path)
    p.add_argument('--gpus-per-node', type=int, default=4)
    p.add_argument('--chunk-bytes', type=int, default=16777216)
    p.add_argument('--intra-gbps', type=float, default=400)
    p.add_argument('--inter-gbps', type=float, default=100)
    p.add_argument('--intra-alpha-ns', type=float, default=0)
    p.add_argument('--inter-alpha-ns', type=float, default=0)
    p.add_argument('--max-local-rounds', type=int, default=1)
    p.add_argument('--freeze-dsl', type=Path, required=True)
    p.add_argument('--show-schedule', action='store_true')
    args = p.parse_args()

    if args.ns3_topology:
        topo = base.read_ns3_topology(
            args.ns3_topology,
            args.chunk_bytes,
            args.intra_alpha_ns,
            args.inter_alpha_ns
        )
    else:
        topo = base.Topology.builtin(
            args.gpus_per_node,
            args.chunk_bytes,
            args.intra_gbps,
            args.inter_gbps,
            args.intra_alpha_ns,
            args.inter_alpha_ns
        )

    solver = InterFirstRD(topo, args.max_local_rounds)
    solver.build()

    print(
        f'STATS={solver.verify()}',
        file=sys.stderr
    )

    shape = []

    for k, g in enumerate(solver.groups):
        loc = sum(
            len(t.chunks)
            for t in g.transfers
            if t.kind == 'intra'
        )
        rem = sum(
            len(t.chunks)
            for t in g.transfers
            if t.kind == 'inter'
        )
        shape.append((loc, rem))

        print(
            f'GROUP[{k}] ({loc},{rem}) '
            f'{g.note} '
            f'budget={g.inter_budget_ns:.2f}ns',
            file=sys.stderr
        )

        if args.show_schedule:
            for t in g.transfers:
                print(
                    f'  T({t.src}, {t.dst}, '
                    f'{list(t.chunks)}) {t.kind}',
                    file=sys.stderr
                )

    dsl = base.render_dsl(
        topo,
        [
            t
            for g in solver.groups
            for t in g.transfers
        ],
        'bai_inter_first_rd_v2'
    )

    compile(dsl, str(args.freeze_dsl), 'exec')

    args.freeze_dsl.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.freeze_dsl.write_text(
        dsl, encoding='utf8'
    )

    print(
        f'GROUP_SHAPE={shape}',
        file=sys.stderr
    )
    print(
        f'DSL={args.freeze_dsl}',
        file=sys.stderr
    )


if __name__ == '__main__':
    main()
