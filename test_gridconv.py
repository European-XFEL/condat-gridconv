import numpy as np

from condat_gridconv.shift import inplace_roll, fractional_roll
from condat_gridconv.gridconv import hex2cart

def test_inplace_roll():
    a = np.random.random(10)
    orig = a.copy()

    # Forwards
    inplace_roll(a, 3)
    np.testing.assert_array_equal(a, np.roll(orig, 3))

    # Backwards
    inplace_roll(a, -5)
    np.testing.assert_array_equal(a, np.roll(orig, -2))

def test_fractional_roll_roundtrip():
    a = np.random.random(10)
    orig = a.copy()

    fractional_roll(a, 0.2)
    assert not np.allclose(orig, a)

    fractional_roll(a, -0.2)
    np.testing.assert_allclose(a, orig)

def test_hex2cart():
    a = np.zeros((128, 512))
    res = hex2cart(a)

    # The output should have roughly as many pixels as the input
    assert 0.9 < (res.size / a.size) < 1.1
