#!/usr/bin/env python3
"""Create portable working copies of frozen 2x4 inputs."""
import argparse
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--et-prefix', required=True)
p.add_argument('--output-dir', required=True)
a = p.parse_args()

prefix = Path(a.et_prefix).expanduser().resolve()
for rank in range(8):
    et = Path(f'{prefix}.{rank}.et')
    if not et.is_file() or not et.stat().st_size:
        p.error(f'Missing rank trace: {et}')

base = repo / 'configs'
net = base / 'observed/r1_2x4_network.txt'
topo = base / 'topology/r1_2x4_topology.txt'
sys = base / 'observed/r1_2x4_system.json'

for src in (net, topo, sys):
    if not src.is_file():
        p.error(f'Missing archived input: {src}')

out = Path(a.output_dir).expanduser().resolve()
out.mkdir(parents=True, exist_ok=True)
logs = out / 'logs'
logs.mkdir(exist_ok=True)

lines = []
for line in net.read_text().splitlines():
    fields = line.split(maxsplit=1)
    if len(fields) == 2 and not line.lstrip().startswith('#'):
        key = fields[0]
        if key == 'TOPOLOGY_FILE':
            line = f'{key} {topo.resolve()}'
        elif key.endswith('_FILE'):
            line = f'{key} {logs / key.lower()}'
    lines.append(line)

(out / 'network.txt').write_text('\n'.join(lines) + '\n')

data = json.loads(sys.read_text())
if 'all-gather-implementation-custom' not in data:
    p.error('Custom AllGather trace setting is missing')

data['all-gather-implementation-custom'] = [str(prefix)]
(out / 'system.json').write_text(
    json.dumps(data, indent=2) + '\n'
)

print('READY', out / 'network.txt', out / 'system.json')
