#!/usr/bin/env python3
"""Verify saved capacities, decomposition columns and prediction metrics.

Python 3 standard library only. Does not train or write original result files.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'B5': ('NASA', 60, 1.4), '38': ('CALCE', 200, .88)}
MODELS = ['BiTCN', 'BiTCN-AM', 'VMD-BiTCN', 'VMD-BiTCN-AM', 'ALA-VMD-BiTCN-AM']

def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return [[float(x) for x in r] for r in csv.reader(f) if r]

def crossing(values, indices, threshold):
    return next((i for i, y in zip(indices, values) if y < threshold), None)

def evaluate():
    provenance = json.loads((ROOT / 'source_manifest.json').read_text())
    for name, record in provenance.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record['sha256'], name
    report = {'verified_source_files': len(provenance), 'datasets': {}}
    for key, (dataset, start, threshold) in CONFIG.items():
        capacity_file = 'B5.csv' if key == 'B5' else 'CS2_38.csv'
        capacity = [r[0] for r in rows(ROOT / 'data/capacity' / capacity_file)]
        decomposition_files = ['ALAVMDB5.csv', 'VMDB5.csv'] if key == 'B5' else ['ALAVMDCS238.csv', 'VMDCS238.csv']
        components = {}
        for name in decomposition_files:
            signals = rows(ROOT / 'data/decompositions' / name)
            assert len(signals) == len(capacity), name
            assert all(abs(r[-1] - y) < 1e-8 for r, y in zip(signals, capacity)), name
            components[name] = len(signals[0]) - 1
        indices = list(range(start + 1, len(capacity)))
        methods = {}
        for model in MODELS:
            path = ROOT / 'results' / f'{model}-{key}.csv'
            with path.open(newline='', encoding='utf-8-sig') as f:
                records = list(csv.DictReader(f))
            actual = [float(r['Last_Column_Actual']) for r in records]
            predicted = [float(r['Summed_Predictions']) for r in records]
            assert len(actual) == len(indices), path.name
            assert all(math.isfinite(v) for v in actual + predicted), path.name
            assert all(abs(y-capacity[i]) < 1e-8 for i, y in zip(indices, actual)), path.name
            errors = [a-p for a, p in zip(actual, predicted)]
            # The saved Difference column retains higher precision than the CSV predictions.
            assert max(abs(e-float(r['Difference'])) for e, r in zip(errors, records)) < 2e-7, path.name
            mean = sum(actual)/len(actual)
            mse = sum(e*e for e in errors)/len(errors)
            tss = sum((a-mean)**2 for a in actual)
            true_eol = crossing(actual, indices, threshold)
            predicted_eol = crossing(predicted, indices, threshold)
            methods[model] = {'points': len(actual), 'rmse_Ah': math.sqrt(mse),
                'mae_Ah': sum(abs(e) for e in errors)/len(errors),
                'r2': 1-sum(e*e for e in errors)/tss,
                'true_eol_index': true_eol, 'predicted_eol_index': predicted_eol,
                'eol_absolute_error': abs(true_eol-predicted_eol) if true_eol is not None and predicted_eol is not None else None}
        report['datasets'][key] = {'dataset': dataset, 'battery': 'B0005' if key == 'B5' else 'CS2_38',
            'capacity_rows': len(capacity), 'target_index_range': [indices[0], indices[-1]],
            'failure_threshold_Ah': threshold, 'decomposition_components': components, 'methods': methods}
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report; no file is written by default.')
    args = parser.parse_args()
    report = json.dumps(evaluate(), indent=2, ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)
