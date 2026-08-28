from datetime import time

from predictor.config import CONFIG


def test_phase0_params_match_spec():
    assert CONFIG.instrument.yf_ticker == "^NSEI"
    assert CONFIG.labeling.k == 1.0
    assert CONFIG.labeling.atr_window == 14
    assert CONFIG.labeling.vertical_barrier == time(15, 20)
    assert CONFIG.labeling.entry_start == time(9, 30)
    assert CONFIG.labeling.entry_end == time(14, 30)
    assert CONFIG.labeling.entry_freq_minutes == 15
    assert CONFIG.labeling.barriers_symmetric is True


def test_paths_are_absolute_and_under_root():
    assert CONFIG.paths.data_dir.is_absolute()
    assert CONFIG.paths.root in CONFIG.paths.data_dir.parents or \
        CONFIG.paths.data_dir.parent == CONFIG.paths.root
