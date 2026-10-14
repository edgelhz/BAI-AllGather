#!/usr/bin/env python3
"""Current MultiNode-RemoteCore vs Hierarchical Ring: 2x4 bandwidth sweep."""
import argparse, csv, hashlib, json, os, re, subprocess
from pathlib import Path
from collections import Counter
from decimal import Decimal
from xml.etree import ElementTree as ET

PAIRS = [
    (400,25), (400,40), (200,25), (200,40),
    (100,25), (400,100), (100,40), (200,100),
    (400,200), (100,100), (200,200), (400,400)
]
KEY = 'all-gather-implementation-custom'

def need(p):
    p = Path(p)
    if not p.is_file() or not p.stat().st_size:
        raise RuntimeError(f'Missing: {p}')
    return p

def sha(p):
    h = hashlib.sha256()
    with need(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def immutable(p, s):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_text() != s:
            raise RuntimeError(f'Changed cached input: {p}')
    else:
        p.write_text(s)

def audit_xml(p):
    root = ET.parse(need(p)).getroot()
    assert root.tag == 'algo'
    assert root.get('coll') == 'allgather'
    assert int(root.get('ngpus', '-1')) == 8
    gpus = root.findall('gpu')
    assert len(gpus) == 8
    assert {int(x.get('id')) for x in gpus} == set(range(8))
    copies = Counter()
    for g in gpus:
        src = int(g.get('id'))
        for tb in g.findall('tb'):
            dst = int(tb.get('send', '-1'))
            if dst < 0:
                continue
            assert 0 <= dst < 8 and src != dst
            for st in tb.findall('step'):
                if st.get('type') != 's':
                    continue
                off = int(st.get('srcoff'))
                assert off == int(st.get('dstoff'))
                n = int(st.get('cnt', '1'))
                assert n >= 1 and 0 <= off and off + n <= 8
                for c in range(off, off+n):
                    copies[src,dst,c] += 1
    assert sum(copies.values()) == len(copies) == 56
    assert {(d,c) for s,d,c in copies} == {
        (d,c) for d in range(8) for c in range(8) if c != d
    }
    assert sum(
        v for (s,d,c),v in copies.items() if s//4 != d//4
    ) == 8

def topology_text(p, intra, inter):
    lines = need(p).read_text().splitlines()
    ids = [
        (i,l.split()) for i,l in enumerate(lines)
        if l.strip() and not l.lstrip().startswith('#')
    ]
    assert list(map(int, ids[0][1][:3])) == [10,2,9]
    assert list(map(int, ids[1][1])) == [8,9]
    seen, cross = set(), 0
    for pos,r in ids[2:]:
        u,v = map(int, r[:2])
        assert len(r) >= 5
        if {u,v} == {8,9}:
            assert r[2] == '100Gbps'
            cross += 1
            r[2] = f'{inter}Gbps'
        else:
            host = u if u < 8 else v
            sw = v if u < 8 else u
            assert host in range(8)
            assert sw == 8 + host//4
            assert r[2] == '400Gbps'
            assert host not in seen
            seen.add(host)
            r[2] = f'{intra}Gbps'
        lines[pos] = ' '.join(r)
    assert seen == set(range(8)) and cross == 1
    return '\n'.join(lines) + '\n'

def network_text(p, topo_name, fct, intra, inter, copy100):
    lines, hits = [], Counter()
    for line in need(p).read_text().splitlines():
        w = line.split()
        if not w or line.lstrip().startswith('#'):
            lines.append(line)
            continue
        k = w[0]
        if k in ('KMIN_MAP','KMAX_MAP','PMAX_MAP'):
            hits[k] += 1
            assert len(w) == 2 + 2*int(w[1])
            rates = {
                int(w[i]): w[i+1]
                for i in range(2,len(w),2)
            }
            assert len(rates) == int(w[1])
            for speed in (intra,inter):
                bps = speed*1000000000
                if bps not in rates:
                    if not copy100 or 100000000000 not in rates:
                        raise RuntimeError(
                            f'{k} has no {speed}Gbps mapping'
                        )
                    rates[bps] = rates[100000000000]
            if copy100:
                w = [k,str(len(rates))]
                for speed,v in sorted(rates.items()):
                    w.extend([str(speed),v])
                line = ' '.join(w)
        elif k == 'TOPOLOGY_FILE':
            hits[k] += 1
            line = (
                'TOPOLOGY_FILE ../../scratch/topology/'
                + topo_name
            )
        elif k == 'FCT_OUTPUT_FILE':
            hits[k] += 1
            line = f'FCT_OUTPUT_FILE {fct}'
        elif k == 'ENABLE_TRACE':
            hits[k] += 1
            line = 'ENABLE_TRACE 0'
        elif k in (
            'TRACE_OUTPUT_FILE',
            'PFC_OUTPUT_FILE',
            'QLEN_MON_FILE'
        ):
            hits[k] += 1
            line = f'{k} {fct.parent}/{k.lower()}.txt'
        lines.append(line)
    assert hits['TOPOLOGY_FILE'] == 1
    assert hits['FCT_OUTPUT_FILE'] == 1
    assert all(
        hits[k] == 1
        for k in ('KMIN_MAP','KMAX_MAP','PMAX_MAP')
    )
    assert all(v <= 1 for v in hits.values())
    if not hits['ENABLE_TRACE']:
        lines.append('ENABLE_TRACE 0')
    return '\n'.join(lines) + '\n'

def verified_result(d):
    j = json.loads(need(d/'result.json').read_text())
    log = need(d/'run.log').read_text(errors='replace')
    pairs = re.findall(
        r'sys\[(\d+)\] finished,\s*(\d+) cycles', log
    )
    assert len(pairs) == 8
    assert {int(x) for x,y in pairs} == set(range(8))
    t = max(int(y) for x,y in pairs)
    flows = sum(
        bool(x.strip())
        for x in need(d/'fct.txt').read_text().splitlines()
    )
    assert flows == 56 and t == int(j['cycles'])
    return t

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='store_true')
    p.add_argument('--timeout', type=int, default=7200)
    p.add_argument(
        '--ecn-copy-from-100', action='store_true',
        help='Explicit fallback for missing ECN maps'
    )
    a = p.parse_args()
    assert a.timeout > 0

    astra = Path(os.getenv(
        'ASTRA', '/app/astra-sim/astra-sim'
    ))
    capi = Path(os.getenv(
        'CAPI', '/app/astra-sim/collectiveapi'
    ))
    ns3 = Path(os.getenv(
        'NS3', str(astra/'extern/network_backend/ns-3')
    ))
    base = Path(os.getenv(
        'BASE', str(astra/'bai_multinode_remote_core')
    ))
    paper = base/'paper_current_2x4_400_100'
    old = capi/'runs/fullmesh_ring_priority_v2_matched_20260919_083537'
    ref = old/'16MiB'
    out = Path(os.getenv(
        'OUT', str(base/'bandwidth_fig4_current_20261013')
    ))

    binary = need(os.getenv(
        'BIN', str(ns3/'build/scratch/ns3.42-AstraSimNetwork-default')
    ))
    remote = need(
        astra/'examples/remote_memory/analytical/no_memory_expansion.json'
    )
    logical = need(
        astra/'examples/network/ns3/sample_8nodes_1D.json'
    )
    topo = need(
        ns3/'scratch/topology/8_nodes_2switch_2x4_1lane_slow_inter_topology.txt'
    )
    net = need(
        ns3/'scratch/config/config_2node_4x4_1lane_slow_inter.txt'
    )
    workload = ref/'workload/all_gather'

    assert sha(topo) == (
        'f63348c7f20702606e4b7e4ec2daa25a95112d00f6b5f51958a3c84e2f7afbd3'
    )
    assert sha(net) == (
        '8fd7b80ebb8ddcb59d2cf8d6edc57b827f6cebc7f42f0355be280dbbc1a67fc7'
    )

    cur = paper/'16MiB/current'
    hier = paper/'16MiB/hier'
    sys_source = {
        'current': need(cur/'system.json'),
        'hier': need(ref/'hierarchical_ring.json')
    }
    source = {
        k: json.loads(v.read_text())
        for k,v in sys_source.items()
    }
    assert set(source['current']) == set(source['hier'])
    assert {
        k for k in source['current']
        if source['current'][k] != source['hier'][k]
    } == {KEY}

    xml = {
        'current': paper/'current_multinode_remote_core_2x4.xml',
        'hier': old/'hierarchical_ring.xml'
    }
    for role in xml:
        audit_xml(xml[role])
        assert len(source[role][KEY]) == 1
        for rank in range(8):
            need(str(source[role][KEY][0])+f'.{rank}.et')
    for rank in range(8):
        need(f'{workload}.{rank}.et')

    print(
        'INPUT_AUDIT_PASS current XML + Hier XML: total=56 inter=8',
        flush=True
    )

    results = []
    for intra,inter in PAIRS:
        if (intra,inter) == (400,100):
            c,h = verified_result(cur),verified_result(hier)
            kind = 'reused'
            print(f'REUSE 400/100 CURRENT={c} HIER={h}',flush=True)
        else:
            top_new = (
                ns3/'scratch/topology'
                /f'bai_bw_{out.name}_2x4_{intra}_{inter}.txt'
            )
            immutable(
                top_new, topology_text(topo,intra,inter)
            )
            values = {}
            kind = 'new'

            for role in ('current','hier'):
                job = out/f'intra{intra}_inter{inter}'/role
                fct = job/'fct.txt'
                system = job/'system.json'
                network = job/'network.txt'

                immutable(
                    system,
                    json.dumps(
                        source[role],indent=2,sort_keys=True
                    )+'\n'
                )
                immutable(
                    network,
                    network_text(
                        net,top_new.name,fct,intra,inter,
                        a.ecn_copy_from_100
                    )
                )
                files = [
                    top_new,system,network,binary,remote,logical,
                    xml[role]
                ]
                files += [
                    Path(f'{workload}.{r}.et')
                    for r in range(8)
                ]
                files += [
                    Path(str(source[role][KEY][0])+f'.{r}.et')
                    for r in range(8)
                ]
                fingerprint = hashlib.sha256(
                    json.dumps(
                        {str(q):sha(q) for q in files},
                        sort_keys=True
                    ).encode()
                ).hexdigest()
                immutable(
                    job/'fingerprint.sha256',
                    fingerprint+'\n'
                )

                if (job/'result.json').exists():
                    saved = json.loads(
                        need(job/'result.json').read_text()
                    )
                    assert saved['fingerprint'] == fingerprint
                    values[role] = verified_result(job)
                    print(
                        f'REUSE {intra}/{inter} {role}={values[role]}',
                        flush=True
                    )
                elif not a.run:
                    print(
                        f'PRECHECK {intra}/{inter} {role}',
                        flush=True
                    )
                else:
                    if fct.exists():
                        fct.unlink()
                    cmd = [
                        binary,
                        f'--workload-configuration={workload}',
                        f'--system-configuration={system}',
                        f'--network-configuration={network}',
                        f'--remote-memory-configuration={remote}',
                        f'--logical-topology-configuration={logical}',
                        '--comm-group-configuration=empty'
                    ]
                    with (job/'run.log').open('w') as f:
                        done = subprocess.run(
                            [str(x) for x in cmd],
                            cwd=ns3/'build/scratch',
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            timeout=a.timeout
                        )
                    if done.returncode:
                        raise RuntimeError(
                            f'Simulation failed {intra}/{inter} {role}: '
                            + '\n'.join(
                                (job/'run.log').read_text().splitlines()[-25:]
                            )
                        )
                    txt = (job/'run.log').read_text(errors='replace')
                    matches = re.findall(
                        r'sys\[(\d+)\] finished,\s*(\d+) cycles',
                        txt
                    )
                    assert len(matches) == 8
                    assert {
                        int(r) for r,t in matches
                    } == set(range(8))
                    cycles = max(int(t) for r,t in matches)
                    flows = sum(
                        bool(x.strip())
                        for x in need(fct).read_text().splitlines()
                    )
                    assert flows == 56
                    (job/'result.json').write_text(
                        json.dumps({
                            'cycles':cycles,
                            'fct_flows':flows,
                            'fingerprint':fingerprint
                        },indent=2)+'\n'
                    )
                    values[role] = verified_result(job)
                    print(
                        f'SIM_PASS {intra}/{inter} '
                        f'{role}={values[role]}',
                        flush=True
                    )
            if len(values) != 2:
                continue
            c,h = values['current'],values['hier']

        gain = (Decimal(h)-Decimal(c))*100/Decimal(h)
        assert abs(
            Decimal(h)*(1-gain/100)-Decimal(c)
        ) < Decimal('0.000001')
        results.append([
            intra,inter,
            str(Decimal(inter)/Decimal(intra)),
            c,h,str(gain),kind
        ])
        print(
            f'COMPARE {intra}/{inter}: '
            f'BAI={c} HIER={h} reduction={gain:.4f}%',
            flush=True
        )

    if a.run:
        out.mkdir(parents=True,exist_ok=True)
        csv_path = out/'fig4_current_matrix.csv'
        with csv_path.open('w',newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'intra_gbps','inter_gbps','rho',
                'bai_cycles','hier_cycles',
                'reduction_pct','source'
            ])
            w.writerows(results)
        print('CSV='+str(csv_path))
    else:
        print('PREFLIGHT_PASS (no simulation)')
    print('DONE')

if __name__ == '__main__':
    main()
