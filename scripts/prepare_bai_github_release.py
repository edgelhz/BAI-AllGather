#!/usr/bin/env python3
"""Inspect and safely stage BAI-AllGather publication artifacts; never push."""
import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_BYTES = 8 * 1024 * 1024

SCRIPTS = (
    'run_multinode_rc_only.sh',
    'run_multinode_hier_matched.sh',
    'run_nodesweep_extension.sh',
    'run_paper_current_2x4.sh',
    'run_current_only_reuse_hier.py',
    'run_multinode_message_sizes.sh',
    'scan_bw_2x4_current.py',
    'run_multinode_bandwidth_matched.py',
    'run_multinode_remote_core_deferred_inter.py',
    'check_astra_variation.py',
    'prepare_bai_github_release.py',
)

ESSENTIAL_SCRIPTS = (
    'run_multinode_rc_only.sh',
    'run_multinode_hier_matched.sh',
    'run_multinode_bandwidth_matched.py',
    'run_multinode_remote_core_deferred_inter.py',
)

CSV_ITEMS = (
    ('paper_current_2x4_400_100/tableIV_current_vs_hier.csv',
     'two_node_message_sizes.csv', 7),
    ('bandwidth_fig4_current_20261013/fig4_current_matrix.csv',
     'two_node_bandwidth.csv', 12),
    ('paper_current_2node_gpu_scale_reuse_hier_16MiB/comparison.csv',
     'two_node_gpu_scale.csv', 3),
    ('nodecount_yC3RLZ08/nodecount_comparison.csv',
     'multi_node_scalability.csv', 5),
    ('message_sweep_multinode_20261013/message_size_comparison_full21.csv',
     'multi_node_message_sizes.csv', 21),
    ('bandwidth_star_multinode_20261013/bandwidth_multinode_comparison.csv',
     'multi_node_bandwidth.csv', 9),
    ('deferred_inter_strict_20261014/deferred_inter_comparison.csv',
     'deferred_inter.csv', 3),
)

SECRET_RE = [
    ('private-key', re.compile(
        r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('github-token', re.compile(
        r'\b(?:ghp_|gho_|ghu_|ghs_|github_pat_)[A-Za-z0-9_]{12,}')),
    ('cloud-access-key', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('api-token', re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b')),
    ('credential-in-url', re.compile(
        r'https?://[^\s/@:]+:[^\s/@]+@')),
    ('sensitive-assignment', re.compile(
        r'''(?im)^\s*(?:password|secret_key|access_token|api_key)\s*[:=]\s*["']?[^\s"']{8,}''')),
]

ABS_RE = re.compile(r'/app/astra-sim/|/opt/venv/|/tmp/')


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def run_capture(*args, cwd=None):
    try:
        p = subprocess.run(
            args, cwd=cwd, text=True,
            capture_output=True, timeout=15
        )
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def repo_identity(path):
    path = Path(path)
    commit = run_capture(
        'git', '-C', str(path), 'rev-parse', '--verify', 'HEAD'
    )
    changes = run_capture(
        'git', '-C', str(path),
        'status', '--porcelain', '--untracked-files=no'
    )
    return {
        'path': str(path),
        'commit': commit,
        'tracked_changes': (
            changes if changes is not None else 'UNKNOWN'
        ),
    }


def imported_modules(path):
    tree = ast.parse(
        Path(path).read_text(encoding='utf-8'),
        filename=str(path)
    )
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(
                alias.name.split('.')[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split('.')[0])
    return names


def locate_topology(network, ns3):
    rows = [
        line.strip().split()
        for line in network.read_text(errors='replace').splitlines()
    ]
    names = [
        parts[1] for parts in rows
        if len(parts) == 2 and parts[0] == 'TOPOLOGY_FILE'
    ]
    if len(names) != 1:
        raise ValueError(
            f'expected one TOPOLOGY_FILE in {network}'
        )
    name = Path(names[0])
    if not name.is_absolute():
        name = ns3 / 'build/scratch' / name
    path = name.resolve()
    root = (ns3 / 'scratch/topology').resolve()
    if path != root and root not in path.parents:
        raise ValueError(
            f'topology outside trusted ns-3 topology directory: {path}'
        )
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--base',
        default='/app/astra-sim/astra-sim/bai_multinode_remote_core'
    )
    ap.add_argument('--scripts-root', default='/tmp')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    base = Path(args.base).resolve()
    astra = base.parent
    ns3 = astra / 'extern/network_backend/ns-3'
    capi = astra.parent / 'collectiveapi'
    scripts_root = Path(args.scripts_root).resolve()
    target = (
        Path(args.out).resolve()
        if args.out else base / 'BAI-AllGather-release'
    )

    if not base.is_dir():
        ap.error(f'BASE directory absent: {base}')
    if target == base or target in (astra, ns3):
        ap.error('Unsafe destination path')

    selected = {}
    warnings = []
    blockers = []
    result_counts = {}

    def select(path, rel, mandatory=False):
        path = Path(path)
        if path.is_symlink():
            blockers.append(
                f'SYMLINK_SKIPPED {rel}: {path}'
            )
            return False
        if not path.is_file():
            (blockers if mandatory else warnings).append(
                f'MISSING {rel}: {path}'
            )
            return False
        if (
            path.stat().st_size == 0
            or path.stat().st_size > MAX_BYTES
        ):
            blockers.append(
                f'BAD_SIZE {rel}: {path} size={path.stat().st_size}'
            )
            return False
        if rel in selected:
            blockers.append(f'DUPLICATE_DESTINATION {rel}')
            return False
        selected[rel] = path
        return True

    core = base / 'bai_multinode_remote_core.py'
    select(
        core, 'src/bai_multinode_remote_core.py',
        mandatory=True
    )

    testfile = base / 'test_bai_multinode_remote_core.py'
    if testfile.is_file():
        select(
            testfile,
            'tests/test_bai_multinode_remote_core.py'
        )
    else:
        warnings.append(
            'OPTIONAL_TEST_MISSING test_bai_multinode_remote_core.py'
        )

    local_py = {
        p.stem: p
        for p in (
            list(base.glob('*.py'))
            + list((base / 'legacy_reference').glob('*.py'))
        )
    }

    todo = (
        [core, testfile] if testfile.is_file()
        else [core]
    )
    seen = set()

    while todo:
        current = todo.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)

        try:
            deps = imported_modules(current)
        except (SyntaxError, UnicodeError) as exc:
            blockers.append(
                f'BAD_PYTHON_SYNTAX {current}: {exc}'
            )
            continue

        source_text = current.read_text(errors='replace')
        if (
            'legacy_reference' in source_text
            or 'bai_v2_remote_core' in source_text
        ):
            deps.add('bai_v2_remote_core')

        for name in sorted(deps):
            candidate = local_py.get(name)
            if (
                candidate is None
                or candidate == core
                or candidate == testfile
            ):
                continue

            if candidate.parent.name == 'legacy_reference':
                rel = f'src/legacy_reference/{candidate.name}'
            else:
                rel = f'src/{candidate.name}'

            if rel not in selected and select(candidate, rel):
                todo.append(candidate)

    if (
        (base / 'legacy_reference/bai_v2_remote_core.py').is_file()
        and not any(
            'bai_v2_remote_core.py' in s
            for s in selected
        )
    ):
        warnings.append(
            'CHECK_LOCAL_DEPENDENCY '
            'legacy_reference/bai_v2_remote_core.py '
            'was not imported statically'
        )

    for name in SCRIPTS:
        source = (
            base / name
            if (base / name).is_file()
            else scripts_root / name
        )
        select(
            source, f'scripts/{name}',
            mandatory=name in ESSENTIAL_SCRIPTS
        )

    for original, published, expected in CSV_ITEMS:
        source = base / original
        rel = f'results/{published}'
        if not select(source, rel, mandatory=True):
            continue

        try:
            with source.open(
                newline='', encoding='utf-8-sig'
            ) as f:
                rows = list(csv.reader(f))
            rows = [
                row for row in rows
                if row and any(cell.strip() for cell in row)
            ]
            count = len(rows) - 1
            if count < 0:
                raise ValueError('empty csv')

            result_counts[published] = {
                'rows': count,
                'expected': expected,
            }
            if count != expected:
                blockers.append(
                    f'INCOMPLETE_CSV {published}: '
                    f'rows={count}, expected={expected}; '
                    'verify original scan batches'
                )
        except (
            UnicodeError, OSError, csv.Error, ValueError
        ) as exc:
            blockers.append(
                f'INVALID_CSV {published}: {exc}'
            )

    for spec in ('3x4', '4x4', '8x4'):
        case = (
            base / 'perf_star_16MiB'
            / spec / '16777216B'
        )
        for item in (
            'network.txt', 'system.json', 'logical.json'
        ):
            select(
                case / item,
                f'configs/observed/{spec}/{item}',
                mandatory=True
            )

        network = case / 'network.txt'
        if network.is_file():
            try:
                select(
                    locate_topology(network, ns3),
                    f'configs/topology/{spec}.txt',
                    mandatory=True
                )
            except ValueError as exc:
                blockers.append(str(exc))

    reference = ns3 / 'scratch'
    select(
        reference / 'config/config_2node_4x4_1lane_slow_inter.txt',
        'configs/two_node_reference/network.txt'
    )
    select(
        reference / 'topology/8_nodes_2switch_2x4_1lane_slow_inter_topology.txt',
        'configs/two_node_reference/topology.txt'
    )

    current_2x4 = (
        base / 'paper_current_2x4_400_100'
        / '16MiB/current'
    )
    for item in ('network.txt', 'system.json'):
        if (current_2x4 / item).is_file():
            select(
                current_2x4 / item,
                f'configs/observed/2x4/{item}'
            )

    if (base / 'LICENSE').is_file():
        select(base / 'LICENSE', 'LICENSE')
    else:
        blockers.append(
            'LICENSE_MISSING: choose and review '
            'a license before a public push'
        )

    for rel, path in selected.items():
        try:
            content = (
                path.read_text(errors='replace')
                if path.suffix in (
                    '.py', '.sh', '.md',
                    '.csv', '.txt', '.json'
                )
                else ''
            )

            for kind, pattern in SECRET_RE:
                if pattern.search(content):
                    blockers.append(
                        f'SENSITIVE_CONTENT {kind}: {rel}; '
                        'review locally before publishing'
                    )

            if (
                rel.startswith(('src/', 'scripts/'))
                and ABS_RE.search(content)
            ):
                warnings.append(
                    f'ABSOLUTE_RUNTIME_PATHS {rel}: '
                    'review and document replacement '
                    'before public release'
                )

            if rel.startswith(('src/', 'scripts/')):
                if path.suffix == '.py':
                    ast.parse(content, filename=str(path))
                if path.suffix == '.sh':
                    test = subprocess.run(
                        ['bash', '-n', str(path)],
                        capture_output=True, text=True
                    )
                    if test.returncode:
                        blockers.append(
                            f'BAD_SHELL_SYNTAX {rel}: '
                            f'{test.stderr.strip()[:200]}'
                        )
        except (SyntaxError, UnicodeError) as exc:
            blockers.append(
                f'INVALID_SOURCE {rel}: {exc}'
            )

    versions = {
        'astra': repo_identity(astra),
        'ns3': repo_identity(ns3),
        'msccl': repo_identity(capi / 'msccl-tools'),
        'sha256': {},
    }

    for label, path in (
        ('scheduler', core),
        (
            'simulator_binary',
            ns3 / 'build/scratch/ns3.42-AstraSimNetwork-default'
        ),
        (
            'chakra_converter',
            astra.parent
            / 'collectiveapi-046d961/chakra_converter/et_converter.py'
        ),
    ):
        versions['sha256'][label] = (
            checksum(path) if path.is_file() else None
        )
        if versions['sha256'][label] is None:
            warnings.append(
                f'ENV_COMPONENT_MISSING {label}: {path}'
            )

    for label in ('astra', 'ns3', 'msccl'):
        value = versions[label]
        if (
            value['commit'] is None
            or value['tracked_changes'] != ''
        ):
            warnings.append(
                f'CHECK_SIMULATOR_WORKTREE {label}: '
                f'commit={value["commit"] or "UNKNOWN"}, '
                f'tracked_changes={value["tracked_changes"][:140]}'
            )

    print(f'CHECKED_FILES={len(selected)}')
    for rel in sorted(selected):
        print(f'FOUND {rel} <- {selected[rel]}')
    for note in warnings:
        print(f'WARNING {note}')
    for note in blockers:
        print(f'BLOCKER {note}')

    print(
        'CHECK_STATUS='
        + ('REVIEW_REQUIRED' if blockers or warnings else 'PASS')
    )

    if args.check_only:
        print('CHECK_ONLY: no publication folder created')
        return

    if 'src/bai_multinode_remote_core.py' not in selected:
        raise RuntimeError(
            'PRIMARY_ALGORITHM_MISSING; refusing incomplete staging'
        )

    if target.exists():
        raise RuntimeError(
            f'DESTINATION_EXISTS {target}; '
            'review/remove or choose another --out; no overwrite'
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix='.BAI-release-build-',
        dir=target.parent
    ))

    try:
        for rel, source in selected.items():
            destination = staging / rel
            destination.parent.mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy2(source, destination)
            if checksum(source) != checksum(destination):
                raise RuntimeError(f'COPY_MISMATCH {rel}')

        (staging / '.gitignore').write_text(
            '__pycache__/\n'
            '*.pyc\n*.log\n*.et\n*.tr\n*.pid\n'
            '.env\n.venv/\n.DS_Store\nraw_runs/\n'
        )

        (staging / 'requirements.txt').write_text(
            '# Third-party simulator and MSCCLang dependencies are external.\n'
            '# Do not infer a pip-only environment from this file.\n'
            '# See reproducibility/environment.md\n'
        )

        (staging / 'README.md').write_text('''# BAI-AllGather: reproduction artifacts

This repository stages the MultiNode-RemoteCore collective schedule generator,
experiment runners, example simulator input snapshots, and result summaries.

## Layout
- `src/`: generator and its local implementation dependencies
- `tests/`: available regression tests
- `scripts/`: experiment construction and comparison scripts
- `configs/`: snapshots of configurations used in completed experiments
- `results/`: copies of published CSV summaries (see validation report)
- `reproducibility/`: external revisions, hashes, checks, and review report

## Generate a schedule (with MSCCL-tools installed externally)
```bash
REPO=$(pwd)
export PYTHONPATH="$REPO/src:$REPO/src/legacy_reference:/path/to/msccl-tools:${PYTHONPATH:-}"
python3 "$REPO/src/bai_multinode_remote_core.py" \\
  --nodes 3 --gpus-per-node 4 --max-local-rounds 1 \\
  --freeze-dsl "$REPO/schedule.py" --report-json "$REPO/report.json"
```

## Reproduction caveats
The historical runner scripts and configuration snapshots may contain absolute
`/app/astra-sim` paths and must be adapted to the checkout on a new machine.
Install compatible ASTRA-sim, ns-3, Chakra, and MSCCL-tools revisions first.
See `reproducibility/environment.md` for captured versions and exact hashes.
`configs/two_node_reference` is a frozen reference, not automatically proof of
the 2x4 Current evaluation input.

Before public release: read `reproducibility/release_audit.md`, verify CSV
completeness, review licensing, secrets, local source dependencies, and
portability. No environment binaries or raw packet traces are included.
''')

        rdir = staging / 'reproducibility'
        rdir.mkdir(parents=True, exist_ok=True)

        (rdir / 'versions.json').write_text(
            json.dumps(
                versions, ensure_ascii=False, indent=2
            ) + '\n'
        )

        imports = {}
        for rel, source in sorted(selected.items()):
            if rel.endswith('.py'):
                try:
                    imports[rel] = sorted(
                        imported_modules(source)
                    )
                except (SyntaxError, UnicodeError):
                    imports[rel] = ['SYNTAX_CHECK_FAILED']

        (rdir / 'imports.json').write_text(
            json.dumps(
                imports, ensure_ascii=False, indent=2
            ) + '\n'
        )

        (rdir / 'environment.md').write_text('''# Experiment environment

See `versions.json` for the actual local git commits, dirty tracked-file
status, and SHA256 of the scheduler, simulation executable, and ET converter.
For the default evaluated configurations: 400 Gbps GPU access links at
0.00025 ms and 100 Gbps inter-node links/uplinks at 0.0015 ms; the network
configurations use PACKET_PAYLOAD_SIZE 1000 and CC_MODE 12. The precise
meaning of CC_MODE 12, endpoint model, and fixed startup costs must be
verified against the pinned simulator source before claiming them.
The third-party simulator and tooling are NOT vendored in this repository.
The supplied system/network JSON and TXT files are original snapshots, not
portable templates; update paths using the included runners and inspect them.
''')

        (rdir / 'release_audit.md').write_text(
            '# Pre-push inspection\n\n'
            f'Checked source items: {len(selected)}\n\n'
            '## Blockers\n'
            + (
                '\n'.join('- ' + x for x in blockers)
                if blockers else '- None detected'
            )
            + '\n\n## Warnings\n'
            + (
                '\n'.join('- ' + x for x in warnings)
                if warnings else '- None detected'
            )
            + '\n\n## CSV rows (data rows exclude header)\n'
            + '\n'.join(
                f'- {name}: {item["rows"]} rows, '
                f'expected {item["expected"]}'
                for name, item in sorted(result_counts.items())
            )
            + '\n\n## Review reminder\n'
            'Absolute paths in config snapshots and runner scripts must\n'
            'be adapted or documented for new systems. Inspect for tokens, private\n'
            'data, third-party licenses, and uncommitted simulator patches.\n'
        )

        manifest = []
        for file in sorted(staging.rglob('*')):
            if not file.is_file():
                continue

            rel = file.relative_to(staging).as_posix()
            if rel in (
                'reproducibility/manifest.json',
                'reproducibility/checksums.sha256'
            ):
                continue

            manifest.append({
                'path': rel,
                'bytes': file.stat().st_size,
                'sha256': checksum(file),
            })

        (rdir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n'
        )
        (rdir / 'checksums.sha256').write_text(
            ''.join(
                f'{item["sha256"]}  {item["path"]}\n'
                for item in manifest
            )
        )

        staging.rename(target)

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f'STAGED={target}')
    print(f'FILES={len(selected)}')
    print(
        'RELEASE_STATUS='
        + (
            'REVIEW_REQUIRED'
            if blockers or warnings
            else 'PREPARED'
        )
    )
    print(
        f'REVIEW={target / "reproducibility/release_audit.md"}'
    )
    print(
        'NOT_PUSHED: Review before running '
        'git init/add/commit/push'
    )


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
