"""Tests for the adaptive L1 controller's noise robustness.

A real run (layer 20 of the Nemotron sweep, metrics.jsonl) recorded the
controller freezing permanently after ~4 updates: window-averaged L0 swung
525 -> 6089 -> 2207 between adjacent windows, the authority reference was
taken from the first noisy window, and by the fourth update l1 had risen
authority_ratio x while the (noisy) L0 sample happened to sit above the
reference — so the guard fired and the run finished with the controller
dead. These tests encode that pathology and the safeguards against it.
"""

import itertools
import random

from kiji_inspector.training.trainer import _AdaptiveL1Controller


def test_guard_cannot_fire_before_min_updates():
    """Recorded layer-20 pathology: wild L0 swings while l1 ramps.

    With the reference deferred until the EMA stabilizes, the guard must not
    freeze within the first min_updates_before_guard updates no matter how
    noisy the input.
    """
    ctrl = _AdaptiveL1Controller(target_l0=75.0, initial_l1=0.005)
    # The actual recorded early-window sequence from layer_20 metrics.jsonl.
    recorded = [3732.9, 1705.2, 5161.4, 3768.4, 2166.8, 525.3, 6089.2, 2207.7, 2297.6]
    for l0 in recorded:
        ctrl.update(l0)
    assert not ctrl.frozen, "guard fired on noise within the deferred-arming window"


def test_ema_absorbs_l0_noise():
    """Alternating extremes around a high mean must not stall the push down.

    Raw window L0 alternates 13 / 6089 (both observed in the real run); the
    mean is far above target, so l1 should rise monotonically — no freeze,
    no direction flapping.
    """
    ctrl = _AdaptiveL1Controller(target_l0=75.0, initial_l1=0.005)
    l1_values = []
    for l0 in itertools.islice(itertools.cycle([13.0, 6089.0]), 40):
        l1_values.append(ctrl.update(l0))
    assert not ctrl.frozen
    assert l1_values[-1] > l1_values[0], "l1 did not rise despite mean L0 far above target"
    # Monotone non-decreasing after the EMA has burned in.
    settled = l1_values[10:]
    assert all(
        b >= a for a, b in zip(settled, settled[1:], strict=False)
    ), "l1 direction flapped on noise"


def test_freeze_then_rearm_when_l0_moves():
    """A freeze is a pause, not a death sentence.

    Force a legitimate freeze (l1 rises authority_ratio x with L0 flat), then
    feed L0 values far below the frozen level: the controller must unfreeze
    and resume adjusting.
    """
    ctrl = _AdaptiveL1Controller(
        target_l0=75.0, initial_l1=0.005, min_updates_before_guard=2, l0_ema_alpha=1.0
    )
    # Flat L0 well above target: l1 ratchets up until the guard freezes it.
    for _ in range(200):
        ctrl.update(500.0)
        if ctrl.frozen:
            break
    assert ctrl.frozen, "setup failed: controller never froze on flat high L0"
    frozen_l1 = ctrl.l1

    # L0 collapses (e.g. threshold adapted, resample event): re-arm expected.
    ctrl.update(100.0)
    assert not ctrl.frozen, "controller did not re-arm after L0 moved"
    for _ in range(5):
        ctrl.update(100.0)
    assert ctrl.l1 != frozen_l1, "controller re-armed but never adjusted again"


def test_legitimate_no_authority_case_still_freezes():
    """The guard's purpose survives: flat EMA L0 + big l1 rise must freeze.

    With a stable (noise-free) L0 stuck far above target, l1 escalates
    multiplicatively; once it has risen authority_ratio x past the (now
    stable) reference without L0 responding, the controller must freeze
    rather than run l1 to l1_max and wreck reconstruction.
    """
    ctrl = _AdaptiveL1Controller(target_l0=75.0, initial_l1=0.005)
    for _ in range(500):
        ctrl.update(500.0)
        if ctrl.frozen:
            break
    assert ctrl.frozen, "guard never fired on a genuine no-authority trajectory"
    assert ctrl.l1 < ctrl.l1_max, "l1 ran all the way to l1_max before freezing"


def test_noisy_but_responsive_l0_reaches_neighborhood_of_target():
    """End-to-end sanity: when l1 genuinely controls L0, the loop converges.

    Simulated plant: L0 responds to l1 (higher l1 -> lower L0) with
    multiplicative noise. The controller should bring L0 near the target
    without freezing.
    """
    rng = random.Random(0)
    ctrl = _AdaptiveL1Controller(target_l0=75.0, initial_l1=0.005)

    def plant(l1):
        base = 75.0 * (0.005 / l1)  # L0 == target exactly at l1 = 0.005 * (500/75)... scaled
        return base * 500 / 75 * rng.uniform(0.7, 1.3)

    l0 = 500.0
    for _ in range(300):
        l1 = ctrl.update(l0)
        l0 = plant(l1)

    assert not ctrl.frozen
    assert ctrl.l0_ema is not None
    assert 0.3 * 75 <= ctrl.l0_ema <= 3 * 75, f"L0 EMA {ctrl.l0_ema:.1f} far from target 75"
