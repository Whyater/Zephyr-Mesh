from desktop.windows_updater import _pid_is_listed


def test_windows_updater_matches_exact_pid_in_tasklist_csv():
    output = '"alpha.exe","112","Console","1","10 K"\n'
    assert _pid_is_listed(output, 112)
    assert not _pid_is_listed(output, 12)
