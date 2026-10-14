from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
import bai_scheduler_v1 as base
from bai_v2_remote_core import RemoteCoreV2
from bai_multinode_remote_core import MultiTopology, MultiNodeRemoteCore, round_robin_pairs, read_multinode_ns3, cli

cases=0
for k in (3,4,5,6,8):
    pairs=round_robin_pairs(k)
    assert len(pairs)==k*(k-1)//2
    assert len(set(pairs))==len(pairs)
    for m in (1,2,3,4,5,6,8):
        for rate in (50.,100.):
            for rounds in (0,1,2):
                t=MultiTopology.builtin(k,m, 4096 if rate==50 else 16777216,400.,rate,0,0)
                a=MultiNodeRemoteCore(t,rounds)
                a.build()
                v=a.verify()
                n=k*m
                assert v['total']==n*(n-1)
                assert v['inter']==n*(k-1)
                assert v['intra']==n*(n-k)
                assert len(a.groups)>=len(a.jobs)
                cases+=1
                if cases%50==0: print('PASSED',cases, f'{k}x{m} rate={rate} rounds={rounds}',flush=True)
for k,m in ((3,12),(4,8),(4,12),(4,16),(8,8)):
    t=MultiTopology.builtin(k,m,16777216,400,100,0,0)
    a=MultiNodeRemoteCore(t,1)
    a.build()
    n=k*m
    assert a.verify()['total']==n*(n-1)
    print('LARGE PASSED',k,m,flush=True)
    cases+=1
print('TOTAL VALID=',cases)

# Synthetic 4-switch mesh and line, test widest-path tie and connectivity.
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'topology.txt';k=4;m=3;n=k*m
    links=[]
    for i in range(k):
        for r in range(i*m,(i+1)*m):
            links.append(f'{r} {n+i} 400Gbps 0.0001ms')
    links.extend([
        f'{n} {n+1} 100Gbps 0.0001ms',
        f'{n+1} {n+2} 50Gbps 0.0001ms',
        f'{n+2} {n+3} 80Gbps 0.0001ms',
        f'{n} {n+3} 20Gbps 0.0001ms',
    ])
    p.write_text(f'{n+k} {k} {len(links)}\n'+ ' '.join(map(str,range(n,n+k)))+'\n'+'\n'.join(links)+'\n')
    t=read_multinode_ns3(p,4096,0,0)
    assert t.gpus_per_node==m
    assert t.pair_bandwidths[0][3]==50.
    a=MultiNodeRemoteCore(t,1)
    a.build()
    assert a.verify()['inter']==n*(k-1)
    bad=p.read_text()
    # disconnected (4th switch no attachments to graph)
    p.write_text(bad.replace(f'{n+2} {n+3} 80Gbps 0.0001ms', f'{n+2} {n+2} 80Gbps 0.0001ms').replace(f'{n} {n+3} 20Gbps 0.0001ms', f'{n} {n+2} 20Gbps 0.0001ms'))
    try: read_multinode_ns3(p,4096,0,0)
    except ValueError: pass
    else: raise AssertionError('bad topology accepted')

for bad in ((2,3), (3,0), (0,4)):
    try: MultiTopology.builtin(bad[0],bad[1],4096,400,100,0,0)
    except ValueError: pass
    else: raise AssertionError(('bad accepted',bad))
for bad_rate in (0,-1):
    try: MultiTopology.builtin(3,4,4096,400,bad_rate,0,0)
    except ValueError: pass
    else: raise AssertionError('bad inter accepted')
print('PHYSICAL+NEGATIVE PASSED')
