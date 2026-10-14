import numpy as np
from sim.radio.channel import GilbertElliott
from sim.radio.heartbeat import *

def test_misses_and_formulas():
    assert max_misses_before_alarm(.3, 10) == 3
    assert max_misses_before_alarm(.25, 10) == 2
    assert max_misses_before_alarm(.35, 10) == 3
    assert max_misses_before_alarm(.1, 50) == 5
    assert abs(false_alarm_rate_iid_per_hour(.1, 10, .3) - 32.4) < 1e-9
    assert abs(false_alarm_rate_iid_per_hour(.01, 10, .3) - .03564) < 1e-6
    ge = GilbertElliott(.02222, .2)
    assert abs(false_alarm_rate_ge_per_hour(ge, 10, .3) - 461) < 2

def test_delay_samples():
    x = detection_delay_samples(10, .3, 10000, np.random.default_rng(3))
    assert x.min() > .2 and x.max() <= .3
    assert abs(x.mean() - .25) < .003

