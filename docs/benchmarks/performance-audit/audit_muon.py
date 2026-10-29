import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np

root = Path('/Users/tu15/alex_cpp/CPP_Completed')
out = root / 'artifacts/performance-audit/muon'
out.mkdir(exist_ok=False)
base = [str(root / 'build/bin/mini_mla_bench'), '--kind', 'train', '--config', str(root / 'config/mla36m-v1.json'), '--backend', 'mps', '--precision', 'fp16', '--attention', 'reference', '--sequence-length', '128', '--microbatch', '8', '--grad-accum', '1', '--submission-window', '4', '--time-zero-grad', '--optimizer', 'muon', '--muon-layout', 'batched']
commands, results = {}, {}
def run(name, order, parity=False):
    command = base + ['--muon-ns-order', order, '--warmup', '0' if parity else '3', '--repetitions', '1' if parity else '30']
    if parity:
        command += ['--comparison-data', str(out / (name + '-tensors'))]
    commands[name] = command
    with (out / (name + '.log')).open('w') as log:
        subprocess.run(command, check=True, timeout=240, stdout=log, stderr=subprocess.STDOUT)
    results[name] = json.loads((out / (name + '.log')).read_text())
    (out / (name + '.json')).write_text(json.dumps(results[name], indent=2) + '\n')
    print(name, round(results[name]['median_ms'], 3), 'ms', flush=True)

run('chain-parity', 'chain', True)
run('gram-parity', 'gram', True)

def tensors(name):
    path = out / (name + '-tensors')
    meta = json.loads((path / 'comparison.json').read_text())
    values = np.memmap(path / 'tensors.bin', dtype=np.float32, mode='r')
    return {t['name']: values[t['offset']//4:(t['offset']+t['nbytes'])//4] for t in meta['tensors']}
a, b = tensors('chain-parity'), tensors('gram-parity')
checks, rows = 0, []
for name in a:
    if not name.endswith('.final'):
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
        checks += 1
for name in a:
    if not name.endswith('.weight'):
        continue
    prefix = name[:-7]
    final = prefix + '.final'
    delta_a = a[final].astype(np.float64) - a[name]
    delta_b = b[final].astype(np.float64) - b[name]
    difference = delta_b - delta_a
    norm_a, norm_b = np.linalg.norm(delta_a), np.linalg.norm(delta_b)
    relative = float(np.linalg.norm(difference) / max(norm_a, 1e-20))
    cosine = float(np.dot(delta_a, delta_b) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
    maximum = float(np.max(np.abs(difference)))
    if relative > 0.001 or cosine < 0.99999 or maximum > 2e-6:
        raise AssertionError((prefix, relative, cosine, maximum))
    rows.append({'parameter': prefix, 'update_relative_l2': relative, 'update_cosine': cosine, 'max_absolute_weight_difference': maximum})
    checks += 1
parity = {'status': 'passed', 'checks': checks, 'parameters': len(rows), 'max_update_relative_l2': max(r['update_relative_l2'] for r in rows), 'min_update_cosine': min(r['update_cosine'] for r in rows), 'max_absolute_weight_difference': max(r['max_absolute_weight_difference'] for r in rows), 'gates': {'initial_weights_and_gradients': 'bitwise equality', 'every_update_relative_l2': 0.001, 'every_update_cosine': 0.99999, 'absolute_weight_difference': 2e-6}, 'results': rows}
(out / 'parity.json').write_text(json.dumps(parity, indent=2) + '\n')
print('full-model parity', parity['max_update_relative_l2'], parity['min_update_cosine'], flush=True)
for name, order in [('chain-first','chain'),('gram-first','gram'),('gram-repeat','gram'),('chain-repeat','chain')]:
    run(name, order)
record = {'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'platform': platform.platform(), 'commands': commands, 'results': results, 'parity': parity, 'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ['src/backends/mps_backend.mm', 'src/train/optimizer.cpp', 'include/mini_mla/execution.hpp', 'apps/bench_main.cpp', 'config/mla36m-v1.json']}, 'binary_sha256': hashlib.sha256((root/'build/bin/mini_mla_bench').read_bytes()).hexdigest()}
(out / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')
