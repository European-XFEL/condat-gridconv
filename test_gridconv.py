import numpy as np
from scipy.ndimage import rotate as scipyrotate

from condat_gridconv.shift import inplace_roll, fractional_roll
from condat_gridconv.gridconv import (
    hex2cart,
    cart2hex,
    rotate,
    pixel_coord_hex2cart,
    pixel_coord_cart2hex,
    cart_shift,
    hex_shift,
    r_cart,
)


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
    tile = np.repeat(
        np.repeat(1.0 + 1.0 * np.arange(16), 64)[:, np.newaxis], 64, axis=1
    )
    tile = np.vstack([tile.T[:, :512], tile.T[:, 512:]])
    ntile = cart2hex(hex2cart(tile))
    np.testing.assert_allclose(ntile, tile, atol=0.5, rtol=0.05)


def test_rotate():
    img = np.zeros((64, 32))
    img[:, 10:14] = 100.0
    img3 = rotate(img, 90, fill_value=0)
    img2 = scipyrotate(img, -90, reshape=False)
    np.testing.assert_allclose(img2, img3, atol=1e-12, rtol=1e-15)


def test_rotate_reversible():
    img = np.zeros((64, 32))
    img[20:32, 10:20] = 100.0
    angle = 4.9
    img2 = rotate(rotate(img, angle, fill_value=0.0), -angle, fill_value=0.0)
    np.testing.assert_allclose(img2, img, atol=1e-1, rtol=1e-8)


def test_pixel_position():
    x_hex = np.array([1, 512])
    y_hex = np.array([1, 128])
    x_cart, y_cart = pixel_coord_hex2cart(x_hex, y_hex)
    x_hex2, y_hex2 = pixel_coord_cart2hex(x_cart, y_cart)
    np.testing.assert_equal(x_hex, x_hex2)
    np.testing.assert_equal(y_hex, y_hex2)


def test_shift():
    M, N = 128, 256
    img = np.zeros((M, N))
    y = np.arange(M) - M / 2
    x = np.arange(N) - N / 2
    xv, yv = np.meshgrid(x, y, indexing="xy")

    sigma = 12
    img = np.exp(-(xv**2 + yv**2) / sigma**2)
    dx, dy = -48.835, 8.8757
    img2 = np.exp(-((xv - dx) ** 2 + (yv - dy) ** 2) / sigma**2)
    img3 = cart_shift(img, dx, dy)
    np.testing.assert_allclose(img2, img3, atol=1e-15, rtol=1.0)


def test_hex_shift():
    M, N = 128, 256
    img = np.zeros((M, N))
    y = np.arange(M) - M / 2
    x = np.arange(N) - N / 2
    xv, yv = np.meshgrid(x, y, indexing="xy")

    sigma = 15
    cval = 1.0
    img = cval + np.exp(-(xv**2 + yv**2) / sigma**2)
    hex_img = cart2hex(img, fill_value=cval)

    dx, dy = 48.835, -12.8757
    img2 = cval + np.exp(-((xv - dx) ** 2 + (yv - dy) ** 2) / sigma**2)
    hex_img2 = cart2hex(img2, fill_value=cval)

    dr1, dr2 = np.dot(r_cart, [dx, dy])
    hex_img3 = hex_shift(hex_img, dr1, dr2, fill_value=cval)

    np.testing.assert_allclose(hex_img2, hex_img3, rtol=1e-5)
