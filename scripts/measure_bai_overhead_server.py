#!/usr/bin/env python3
"""Measure BAI offline preparation overhead."""
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

PY = os.getenv(
    'BAI_PY', '/opt/venv/astra-sim/bin/python3')
HOME = Path(os.getenv(
    'BAI_HOME',
    '/app/astra-sim/astra-sim/bai_multinode_remote_core'))
GEN = Path(os.getenv(
    'BAI_GENERATOR',
    str(HOME / 'bai_multinode_remote_core.py')))
CONV = Path(os.getenv(
    'CHAKRA_CONVERTER',
    '/app/astra-sim/collectiveapi-046d961/'
    'chakra_converter/et_converter.py'))

CASES = os.getenv(
    'BAI_CASES', '2x4,2x16,4x4,8x4').split(',')
REPS = int(os.getenv('BAI_REPS', '5'))
WARM = int(os.getenv('BAI_WARMUPS', '1'))
SIZE = int(os.getenv('BAI_CHUNK_BYTES', '16777216'))
LIMIT = int(os.getenv('BAI_TIMEOUT', '1800'))
OUT = Path(os.getenv(
    'BAI_MEASURE_OUT',
    'bai_overhead_'
    + datetime.now().strftime('%Y%m%d_%H%M%S')
    + '_' + str(os.getpid()))).resolve()


def need(p):
    if not p.is_file() or not p.stat().st_size:
        raise RuntimeError('Missing/empty file: ' + str(p))


def cmd(args, directory, name, xml=False):
    stdout = directory / (
        'schedule.xml' if xml else name + '.stdout.log')
    stderr = directory / (name + '.stderr.log')
    t = time.perf_counter_ns()
    with stdout.open('wb') as a, stderr.open('wb') as b:
        result = subprocess.run(
            args, cwd=directory, env=ENV,
            stdin=subprocess.DEVNULL,
            stdout=a, stderr=b, timeout=LIMIT)
    ms = (time.perf_counter_ns() - t) / 1e6
    if result.returncode:
        raise RuntimeError(
            f'{name} exited {result.returncode}; see {stderr}')
    return round(ms, 6)


def check_xml(path, k, m):
    need(path)
    root = ET.parse(path).getroot()
    n = k * m
    if (root.tag != 'algo'
            or root.get('coll') != 'allgather'
            or int(root.get('ngpus', '-1')) != n):
        raise RuntimeError('Wrong XML header: ' + str(path))
    if len(root.findall('gpu')) != n:
        raise RuntimeError('Incorrect GPU element count')

    total = inter = 0
    for gpu in root.findall('gpu'):
        src = int(gpu.get('id'))
        for tb in gpu.findall('tb'):
            dst = int(tb.get('send', '-1'))
            for step in tb.findall('step'):
                if step.get('type') != 's':
                    continue
                count = int(step.get('cnt', '0'))
                if (count < 1
                        or not (0 <= src < n and 0 <= dst < n)
                        or src == dst):
                    raise RuntimeError('Invalid send in XML')
                total += count
                inter += count * (src // m != dst // m)

    expected = (n * (n - 1), n * (k - 1))
    if (total, inter) != expected:
        raise RuntimeError(
            f'XML deliveries {total}/{inter}, '
            f'expected {expected[0]}/{expected[1]}')


def save(path, data, cols):
    with path.open('w', newline='', encoding='utf-8') as fd:
        w = csv.DictWriter(fd, fieldnames=cols)
        w.writeheader()
        w.writerows(data)


try:
    if REPS < 1 or WARM < 0 or SIZE < 1 or LIMIT < 1:
        raise ValueError('Invalid measurement parameters')

    for f in (Path(PY), GEN, CONV):
        need(f)

    topo = []
    for spec in CASES:
        if not re.fullmatch(r'[1-9]\d*x[1-9]\d*', spec):
            raise ValueError('Invalid topology: ' + spec)
        k, m = map(int, spec.split('x'))
        if k < 2:
            raise ValueError('Requires at least two nodes')
        topo.append((spec, k, m))

    ENV = os.environ.copy()
    paths = [
        str(HOME / 'legacy_reference'),
        str(HOME),
        '/app/astra-sim/collectiveapi/msccl-tools'
    ]
    ENV['PYTHONPATH'] = os.pathsep.join(
        paths + ([ENV['PYTHONPATH']]
                 if ENV.get('PYTHONPATH') else []))

    OUT.mkdir(parents=True, exist_ok=False)

    digest = hashlib.sha256(GEN.read_bytes()).hexdigest()
    manifest = dict(
        generator=str(GEN),
        sha256=digest,
        converter=str(CONV),
        cases=CASES,
        repetitions=REPS,
        warmups=WARM,
        chunk_bytes=SIZE,
        mode='core',
        max_local_rounds=1
    )
    (OUT / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')

    rows, summary = [], []

    for spec, k, m in topo:
        n = k * m

        for rep in range(WARM + REPS):
            run = OUT / spec / f'run_{rep:02d}'
            run.mkdir(parents=True)

            plan = run / 'report.json'
            dsl = run / 'schedule.py'

            a = cmd([
                PY, str(GEN),
                '--nodes', str(k),
                '--gpus-per-node', str(m),
                '--mode', 'core',
                '--max-local-rounds', '1',
                '--chunk-bytes', str(SIZE),
                '--intra-gbps', '400',
                '--inter-gbps', '100',
                '--freeze-dsl', str(dsl),
                '--report-json', str(plan)
            ], run, 'planning')

            need(dsl)
            need(plan)
            data = json.loads(plan.read_text())

            expected = [
                ('nranks', n),
                ('gpus_per_node', m),
                ('total', n * (n - 1)),
                ('inter', n * (k - 1))
            ]
            for key, value in expected:
                if data.get(key) != value:
                    raise RuntimeError(
                        f'Planning audit failed {key}: '
                        f'{data.get(key)} != {value}')

            b = cmd([
                PY, str(dsl), '1',
                '--protocol', 'Simple'
            ], run, 'xml_compilation', xml=True)

            check_xml(run / 'schedule.xml', k, m)

            prefix = run / 'custom_allgather'
            c = cmd([
                PY, str(CONV),
                '--input_filename', str(run / 'schedule.xml'),
                '--output_filename', str(prefix),
                '--coll_size', str(SIZE),
                '--collective', 'allgather',
                '--log_filename', str(run / 'converter.log')
            ], run, 'chakra_conversion')

            for rank in range(n):
                need(run / f'custom_allgather.{rank}.et')

            row = dict(
                case=spec,
                sample=rep,
                warmup=int(rep < WARM),
                planning_ms=a,
                xml_ms=b,
                chakra_ms=c,
                total_ms=round(a+b+c, 6)
            )
            rows.append(row)
            save(
                OUT / 'raw_measurements.csv',
                rows, list(row))

            print(
                f'PASS {spec} sample={rep} '
                f'warmup={row["warmup"]} '
                f'planning={a:.3f}ms '
                f'xml={b:.3f}ms '
                f'chakra={c:.3f}ms',
                flush=True)

        measured = [
            r for r in rows
            if r['case'] == spec and not r['warmup']
        ]

        for key in (
            'planning_ms', 'xml_ms',
            'chakra_ms', 'total_ms'
        ):
            values = [r[key] for r in measured]
            summary.append(dict(
                case=spec,
                stage=key,
                n=len(values),
                median_ms=round(
                    statistics.median(values), 6),
                min_ms=min(values),
                max_ms=max(values)
            ))

        save(
            OUT / 'summary.csv',
            summary, list(summary[0]))

    report = (
        '| Case | Stage | Median (ms) |\n'
        '|---|---|---:|\n'
        + '\n'.join(
            f'| {r["case"]} | {r["stage"]} | '
            f'{r["median_ms"]:.3f} |'
            for r in summary)
        + '\n'
    )
    (OUT / 'report.md').write_text(report)

    print('\nRESULTS:', OUT)
    print(report)

except Exception as exc:
    print('ERROR:', exc, file=sys.stderr)
    sys.exit(1)
