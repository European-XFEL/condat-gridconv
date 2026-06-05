import numpy as np

from condat_gridconv.shift import inplace_roll, fractional_roll
from condat_gridconv.gridconv import (hex2cart, cart2hex, rotate,
                                      pixel_hex2cart, pixel_cart2hex)

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

def test_reversible():
    tile = np.repeat(np.repeat(1.0+1.0*np.arange(16), 64)[:, np.newaxis],
                     64, axis=1)
    tile = np.vstack([tile.T[:, :512], tile.T[:, 512:]])
    ntile = cart2hex(hex2cart(tile))

    assert np.sum(np.abs(ntile - tile)) < 7.0

def test_rotate_reversible():
    img = np.zeros((64, 32))
    img[20:32, 10:20]= 100.0
    angle = 4.9
    err = np.sum(np.abs(img - rotate(rotate(img, angle, fill_value=0.0),
                                     -angle, fill_value=0.0)))
    assert err < 1.0

def test_pixel_position():
    x_hex = 512
    y_hex = 128
    x_cart, y_cart = pixel_hex2cart(x_hex, y_hex)
    x_hex2, y_hex2 = pixel_cart2hex(x_cart, y_cart)
    assert x_hex == x_hex2
    assert y_hex == y_hex2
