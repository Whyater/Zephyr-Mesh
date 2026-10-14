import json

from sim.estimation.engine import CellSpec
from sim.s6_recovery import RecoveryConfig, seed_list
from tools.run_s6_tuning import build_arms, build_cells, job_cell


def test_grid_and_arm_registry():
    cells = build_cells()
    assert len(cells) == 12
    assert len({(c['target'], c['delay_s'], c['loss_probability']) for c in cells}) == 12
    assert len(build_cells(pilot=True)) == 2
    arms = build_arms()
    assert len(arms) == 51
    assert len({a.name for a in arms}) == 51
    assert 'cv_r1' not in {a.name for a in arms}


def test_q25_and_r02_are_weak_twins():
    """Scaling Q and R together leaves gain unchanged after P0 is forgotten, with a small early transient."""
    cell = next(c for c in build_cells()
                if (c['target'], c['delay_s'], c['loss_probability']) == ('cv', 0.0, 0.0))
    arms = [a for a in build_arms() if a.name in {'cv_q25', 'cv_r0.2'}]
    out = job_cell({'slice': 'main', 'cell': cell, 'arms': [a.as_dict() for a in arms], 'n_seeds': 2})
    by_arm = {name: [row['p95_error_m'] for row in out['per_seed'] if row['arm'] == name]
              for name in ('cv_q25', 'cv_r0.2')}
    assert len(by_arm['cv_q25']) == len(by_arm['cv_r0.2']) == 2
    assert all(abs(q25 - r02) <= 0.002 for q25, r02 in zip(by_arm['cv_q25'], by_arm['cv_r0.2']))


def test_job_is_json_serializable():
    cell = build_cells(pilot=True)[0]
    job = {'slice': cell['slice'], 'cell': cell, 'arms': [a.as_dict() for a in build_arms()], 'n_seeds': 1}
    result = job_cell(job)
    json.dumps(result, allow_nan=False)


def test_cv_mistuning_nis_direction():
    cell = {'slice': 'main', 'target': 'cv', 'delay_s': 0.0, 'loss_probability': 0.0,
            'noise_std_m': 0.05, 'rate_hz': 10.0, 'jitter_s': 0.0,
            'loss_model': 'independent', 'noise_model': 'gaussian'}
    arms = build_arms()
    selected = [a for a in arms if a.name in {'recovery', 'cv_r0.2', 'cv_r5'}]
    out = job_cell({'slice': 'main', 'cell': cell, 'arms': [a.as_dict() for a in selected], 'n_seeds': 3})
    assert out['arms']['cv_r0.2']['mean_nis_median'] > out['arms']['recovery']['mean_nis_median']
    assert out['arms']['cv_r5']['mean_nis_median'] < out['arms']['recovery']['mean_nis_median']


def test_naive_matches_recovery_reference():
    from sim.s6_recovery import simulate_run

    cell = next(c for c in build_cells()
                if (c['target'], c['delay_s'], c['loss_probability']) == ('cv', 0.1, 0.0))
    out = job_cell({'slice': 'main', 'cell': cell, 'arms': [build_arms()[0].as_dict()], 'n_seeds': 2})
    actual = [row['p95_error_m'] for row in out['per_seed'] if row['arm'] == 'naive']
    config = RecoveryConfig()
    expected = [simulate_run('constant_velocity', delay_s=0.1, loss_probability=0.0,
                             noise_std_m=0.05, seed=seed, config=config)['naive'].p95_error_m
                for seed in seed_list(2)]
    assert actual == expected
    assert all(abs(a - b) <= 1e-12 for a, b in zip(actual, expected))


def test_small_main_output_is_process_deterministic(tmp_path):
    from tools.run_s6_tuning import main

    args = ['--output', str(tmp_path / 'one'), '--test-small', '--seeds', '2',
            '--processes', '1', '--no-checkpoint']
    assert main(args) == 0
    args[1] = str(tmp_path / 'two')
    args[6] = '2'
    assert main(args) == 0
    names = ['summary.json', 'cells.csv', 'per_seed.csv.gz', 'consistency_check.json']
    for name in names:
        assert (tmp_path / 'one' / name).read_bytes() == (tmp_path / 'two' / name).read_bytes()
    manifest = json.loads((tmp_path / 'one' / 'manifest.json').read_text())
    assert manifest['replay_command'].startswith(
        'PYTHONPATH=. ./.venv/bin/python tools/run_s6_tuning.py ')
    assert (tmp_path / 'one' / 'manifest.json').is_file()
    summary = json.loads((tmp_path / 'one' / 'summary.json').read_text())
    assert all('per_seed' not in cell for cell in summary['cells'])
    assert set(summary['config']['recovery']) == set(RecoveryConfig.__dataclass_fields__)


def test_consistency_check_counts_windows_not_seed_means():
    """The honest-filter check must report the fraction of 10-NIS WINDOWS inside the 95 % band.
    With 10 seeds there are about 100 windows, so an honest filter leaves about 5 % outside and
    the fraction is very unlikely to be exactly 1.0. The repaired code once returned the fraction
    of per-seed MEAN NIS values inside the band instead, which is 1.0."""
    from sim.s6_recovery import seed_list
    from tools.run_s6_tuning import _consistency_check
    check = _consistency_check(seed_list(10))
    assert 0.85 <= check["fraction_windows_inside_band"] < 1.0
