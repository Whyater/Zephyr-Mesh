from __future__ import annotations
import json, tempfile
from pathlib import Path
from tools.d_run_dropout import _job, _thresholds, run

def test_runtime_thresholds_are_loaded_from_registered_file():
    assert _thresholds() == {0.01: 0.3, 0.1: 0.5, 0.3: 1.0}

def test_hand_prediction_formula_independent_check():
    p, period, timeout = 0.3, 0.1, 1.0
    assert abs(timeout + 0.05 - 0.05 + period * p / (1 - p) - 1.0428571428571427) < 1e-12

def test_pilot_is_reproducible_and_row_is_simulated():
    with tempfile.TemporaryDirectory() as temp:
        a, b = Path(temp) / 'a', Path(temp) / 'b'
        run(a, ['--pilot'], pilot=True, processes=1)
        run(b, ['--pilot'], pilot=True, processes=1)
        for name in ('per_seed.csv', 'summary.json', 'dropout_times.png', 'dropout_timeline.png'):
            assert (a / name).read_bytes() == (b / name).read_bytes()
        ma, mb = json.loads((a / 'manifest.json').read_text()), json.loads((b / 'manifest.json').read_text())
        for manifest in (ma, mb):
            for key in ('runtime_s', 'git_revision', 'git_tree_state', 'platform', 'n_processes', 'argv'):
                manifest.pop(key, None)
        assert ma == mb
        row = next(line for line in (a / 'per_seed.csv').read_text().splitlines()[1:] if ',0.0,' in line)
        fields = (a / 'per_seed.csv').read_text().splitlines()[0].split(',')
        values = dict(zip(fields, row.split(',')))
        direct = _job({'n': 4, 'k': 1, 'mode': 'silent', 'link_loss': 0.0, 'seed': int(values['seed']), 'timeout_s': _thresholds()[0.01]})
        assert float(values['detection_time_s']) == direct['detection_time_s']
        assert float(values['recoverage_time_s']) == direct['recoverage_time_s']
        assert (a / 'dropout_times.png').stat().st_size > 10_000
        assert (a / 'dropout_timeline.png').stat().st_size > 10_000
