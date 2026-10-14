#!/usr/bin/env python3
"""BAI V2 remote-chunk-core matching during original inter-node windows.

Only local_round's candidate ranking changes. Inter jobs, their chunk payloads,
readiness logic, temporal grouping, V2's missing-only RD, and the V1 renderer
remain untouched.  Does NOT insert barriers or claim actual overlap from a
static alpha-beta budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import bai_scheduler_v1 as base
from bai_inter_first_rd_scheduler_v2 import InterFirstRD


class RemoteCoreV2(InterFirstRD):
    """Use accumulated remote-chunk ownership as a forwarding-core signal."""

    def __init__(self, topo, max_rounds=1):
        super().__init__(topo, max_rounds)
        self.remote_chunks = {
            r: frozenset(self.topo.nodes[1 - topo.node_of(r)])
            for r in range(self.topo.nranks)
        }
        self.local_matching_rounds = 0
        self.selected_foreign_chunks = 0
        self.core_selections = []

    def local_round(self, state, busy, budget, next_busy):
        """Match all idle endpoints to maximize useful remote dissemination.

        DP is exact over per-node disjoint matchings (not over all network
        schedules). Each pair still uses the V2 full missing-only exchange
        and must fit the original window's per-pair budget estimate.
        """
        from functools import lru_cache
        if budget < 0:
            raise AssertionError('negative local backfill budget')
        selected, costs = [], []
        selected_records = []
        for node in self.topo.nodes:
            idle = [r for r in node if r not in busy]
            opts = {}
            for i, a in enumerate(idle):
                for j in range(i+1, len(idle)):
                    b = idle[j]
                    transfers = self.pair(a, b, state)
                    if not transfers:
                        continue
                    ab = sum(len(t.chunks) for t in transfers if t.src == a)
                    ba = sum(len(t.chunks) for t in transfers if t.src == b)
                    cost = self.topo.intra.exchange_ns(ab, ba, self.topo.chunk_bytes)
                    if cost > budget + 1e-6:
                        continue
                    stock_a = len(state[a] & self.remote_chunks[a])
                    stock_b = len(state[b] & self.remote_chunks[b])
                    gained_a = sum(len(set(t.chunks) & self.remote_chunks[a])
                                   for t in transfers if t.src == a)
                    gained_b = sum(len(set(t.chunks) & self.remote_chunks[b])
                                   for t in transfers if t.src == b)
                    prep = self.prep_value(transfers, state)
                    fresh = sum(len(set(t.chunks) & self.hot.get(t.src,set()))
                                for t in transfers)
                    # Maximize node's *aggregate* remote deliveries first;
                    # then give priority to senders rich in foreign chunks.
                    metric = (prep, gained_a+gained_b,
                              stock_a*gained_a+stock_b*gained_b,
                              fresh, ab+ba,
                              -(int(a in next_busy)+int(b in next_busy)),
                              -cost)
                    opts[(i,j)] = (metric, transfers, cost, a,b,
                                   stock_a,stock_b,gained_a,gained_b)

            zeros = (0,0,0,0,0,0,0.)
            @lru_cache(maxsize=None)
            def solve(mask):
                if mask == 0:
                    return zeros, ()
                bit = mask & -mask
                i = bit.bit_length()-1
                best_score, best_pairs = solve(mask ^ bit)  # leave rank idle
                mask2 = mask ^ bit
                while mask2:
                    bit2 = mask2 & -mask2
                    j = bit2.bit_length()-1
                    mask2 ^= bit2
                    option = opts.get((i,j))
                    if option is None:
                        continue
                    score_rest, pairs_rest = solve(mask ^ bit ^ bit2)
                    score = tuple(x+y for x,y in zip(option[0], score_rest))
                    pair_seq = ((i,j),) + pairs_rest
                    if score > best_score or (score == best_score and pair_seq < best_pairs):
                        best_score, best_pairs = score, pair_seq
                return best_score, best_pairs
            _, matching = solve((1<<len(idle))-1)
            for ij in matching:
                metric, transfers, cost, a,b, stock_a,stock_b,gained_a,gained_b = opts[ij]
                selected.extend(transfers)
                costs.append(cost)
                selected_records.append(dict(
                    pair=[a,b], stock=[stock_a,stock_b],
                    new_foreign=[gained_a,gained_b], prep=metric[0],
                    predicted_ns=cost,
                ))
                self.selected_foreign_chunks += gained_a+gained_b

        if selected:
            self.local_matching_rounds += 1
            self.core_selections.append(selected_records)
        return selected, max(costs,default=0.0)


def flattened(groups):
    return [(t.src, t.dst, int(c), t.kind)
            for g in groups for t in g.transfers for c in t.chunks]


def inter_signature(groups):
    return [(t.src, t.dst, tuple(int(c) for c in t.chunks))
            for g in groups for t in g.transfers if t.kind == 'inter']


def foreign_replay(groups, topo):
    """Audit/useful remote-chunk deliveries per group, using pre-group holders."""
    state = {r: {r} for r in range(topo.nranks)}
    group_stats = []
    for index, group in enumerate(groups):
        old = {r: set(s) for r, s in state.items()}
        inter = total = useful_foreign = 0
        remote_by_src = Counter()
        new = {r: set() for r in state}
        for tr in group.transfers:
            cs = set(tr.chunks)
            assert cs and cs <= old[tr.src] and not cs & (old[tr.dst] | new[tr.dst])
            new[tr.dst].update(cs)
            total += len(cs)
            if topo.node_of(tr.src) != topo.node_of(tr.dst):
                inter += len(cs)
            else:
                opposite = set(topo.nodes[1 - topo.node_of(tr.src)])
                remote = cs & opposite
                useful_foreign += len(remote)
                if remote:
                    remote_by_src[tr.src] += len(remote)
        for r, cs in new.items():
            state[r].update(cs)
        group_stats.append(dict(group=index, note=group.note,
                                total=total, inter=inter,
                                remote_intra=useful_foreign,
                                remote_sources=dict(sorted(remote_by_src.items()))))
    assert all(len(chunks) == topo.nranks for chunks in state.values())
    return group_stats


def build_and_report(topo, max_rounds, mode):
    if mode not in ('baseline', 'core'):
        raise ValueError('invalid mode')
    original = InterFirstRD(topo, max_rounds)
    original.build()
    chosen = original if mode == 'baseline' else RemoteCoreV2(topo, max_rounds)
    if chosen is not original:
        chosen.build()
    original_stats, chosen_stats = original.verify(), chosen.verify()
    original_inter, chosen_inter = inter_signature(original.groups), inter_signature(chosen.groups)
    assert original_inter == chosen_inter, 'ERROR: original V2 inter order or payload changed'
    n = topo.nranks
    expected = n * (n - 1)
    old_ops, new_ops = flattened(original.groups), flattened(chosen.groups)
    assert len(old_ops) == len(new_ops) == expected
    assert chosen_stats['inter'] == original_stats['inter'] == n
    assert chosen_stats['total'] == original_stats['total'] == expected
    assert [(s, d, c) for s, d, c, k in new_ops if k == 'inter'] == [
           (s, d, c) for s, d, c, k in old_ops if k == 'inter']
    group_stats = foreign_replay(chosen.groups, topo)
    assert sum(g['remote_intra'] for g in group_stats) == (
        topo.gpus_per_node * topo.gpus_per_node * 2 - topo.nranks
    ), 'remote chunk propagation count inconsistent'
    # Stage-1 extra-local rounds belong to their inter window despite
    # appearing as separate groups without an inter-node transfer.
    window_remote = []
    for g in group_stats:
        if g['note'].startswith('stage1 inter '):
            window_remote.append(g['remote_intra'])
        elif g['note'].startswith('stage1 inter-window extra-local'):
            if not window_remote:
                raise AssertionError('extra-local round before first inter')
            window_remote[-1] += g['remote_intra']
    stage1 = [g for g in group_stats if g['note'].startswith('stage1 inter')]
    stage2 = [g for g in group_stats if g['note'].startswith('stage2')]
    report = dict(
        mode=mode, nranks=n, gpus_per_node=topo.gpus_per_node,
        chunk_bytes=topo.chunk_bytes, max_local_rounds=max_rounds,
        total=chosen_stats['total'], inter=chosen_stats['inter'],
        intra=chosen_stats['intra'], gateway_pair_count=len(chosen.jobs),
        inter_exact_sequence_preserved=True,
        original_v2_group_shape=[[
            sum(len(t.chunks) for t in g.transfers if t.kind == 'intra'),
            sum(len(t.chunks) for t in g.transfers if t.kind == 'inter')
        ] for g in original.groups],
        final_group_shape=[[
            sum(len(t.chunks) for t in g.transfers if t.kind == 'intra'),
            sum(len(t.chunks) for t in g.transfers if t.kind == 'inter')
        ] for g in chosen.groups],
        original_intra_signature_sha256=hashlib.sha256(str(
            [x for x in old_ops if x[3]=='intra']).encode()).hexdigest(),
        chosen_intra_signature_sha256=hashlib.sha256(str(
            [x for x in new_ops if x[3]=='intra']).encode()).hexdigest(),
        inter_window_remote_per_group=window_remote,
        stage1_local_chunks=sum(g['total']-g['inter'] for g in stage1),
        stage1_remote_intra=sum(g['remote_intra'] for g in stage1),
        stage2_remote_intra=sum(g['remote_intra'] for g in stage2),
        predictive_only=True,
        runtime_overlap_verified=False,
    )
    if isinstance(chosen, RemoteCoreV2):
        report['local_matching_rounds']=chosen.local_matching_rounds
        report['chosen_foreign_chunks_in_local_rounds']=chosen.selected_foreign_chunks
        report['core_selection_rounds']=chosen.core_selections
    return chosen, original, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ns3-topology', type=Path)
    parser.add_argument('--gpus-per-node', type=int, default=8)
    parser.add_argument('--chunk-bytes', type=int, default=16777216)
    parser.add_argument('--intra-gbps', type=float, default=400.)
    parser.add_argument('--inter-gbps', type=float, default=100.)
    parser.add_argument('--intra-alpha-ns', type=float, default=0.)
    parser.add_argument('--inter-alpha-ns', type=float, default=0.)
    parser.add_argument('--max-local-rounds', type=int, default=1)
    parser.add_argument('--mode', choices=('baseline','core'), default='core')
    parser.add_argument('--freeze-dsl', type=Path, required=True)
    parser.add_argument('--report-json', type=Path)
    args = parser.parse_args()
    if args.ns3_topology:
        topo = base.read_ns3_topology(
            args.ns3_topology, args.chunk_bytes,
            args.intra_alpha_ns, args.inter_alpha_ns,
        )
    else:
        topo = base.Topology.builtin(
            args.gpus_per_node, args.chunk_bytes, args.intra_gbps,
            args.inter_gbps, args.intra_alpha_ns, args.inter_alpha_ns)
    solver, original, report = build_and_report(topo, args.max_local_rounds, args.mode)
    dsl = base.render_dsl(topo, [t for g in solver.groups for t in g.transfers],
                          'bai_inter_first_rd_v2')
    if args.mode == 'baseline':
        orig_dsl = base.render_dsl(topo,
            [t for g in original.groups for t in g.transfers], 'bai_inter_first_rd_v2')
        assert dsl == orig_dsl, 'baseline is not byte-identical to V2'
    compile(dsl, str(args.freeze_dsl), 'exec')
    args.freeze_dsl.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_dsl.write_text(dsl, encoding='utf-8')
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    short = {k:v for k,v in report.items() if k not in ('core_selection_rounds', 'original_v2_group_shape','final_group_shape','all_group_remote_intra')}
    print('V2_REMOTE_CORE_AUDIT=PASS '+json.dumps(short, sort_keys=True), file=sys.stderr)

if __name__ == '__main__':
    main()
