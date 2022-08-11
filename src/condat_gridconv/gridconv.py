import numpy as np

from .shift import fractional_roll, inplace_roll

sqrt3 = np.sqrt(3)
r_hex = np.sqrt(2 / sqrt3) * np.array([[1, 1 / 2],
                                       [0, sqrt3 /2 ]])
r_cart = np.linalg.inv(r_hex)


def shear(arr, delay, axis):
    """Apply a shear operation in place

    Shift each column (axis=0) or row (axis=1) in the array by an offset
    proportional to its position. One column/row in the centre will stay in
    position
    """
    cx = round(arr.shape[0] / 2)
    cy = round(arr.shape[1] / 2)

    # Vertical shear
    if axis == 0:
        for k in range(2 * cy):
            shift = delay * (k - cy)
            full_pixel_shift = round(shift)

            # Full pixel shifts
            inplace_roll(arr[:, k], full_pixel_shift)

            # Sub-pixel shifts
            fractional_roll(arr[:, k], shift - full_pixel_shift)
    # Horizontal shear
    elif axis == 1:
        for k in range(2 * cx):
            shift = delay * (k - cx)
            full_pixel_shift = round(shift)

            # Full pixel shifts
            inplace_roll(arr[k, :], full_pixel_shift)

            # Sub-pixel shifts
            fractional_roll(arr[k, :], shift - full_pixel_shift)


def pad(tile):
    w0 = tile.shape[0] // 2
    w1 = tile.shape[1] // 2

    # Can we assume that a tile length is always a power of 2?
    width0 = (w0, w0) if tile.shape[0] % 2 == 0 else (w0, w0 + 1)
    width1 = (w1, w1) if tile.shape[1] % 2 == 0 else (w1, w1 + 1)

    return np.pad(tile, (width0, width1), mode="reflect")


def hex2cart(tile):
    padded_tile = pad(tile)
    cy = int(padded_tile.shape[0] / 2)
    cx = int(padded_tile.shape[1] / 2)

    # First we skew the image to a hexagonal shape
    height = tile.shape[0]
    y0 = height % 2  # Can we assume the tile lengths are always powers of 2?
    for y in range(-height // 2, height // 2 + y0 + 1):
        roll_amount = -int(np.floor((y + y0) / 2))
        offset = y - y0 + cy

        padded_tile[offset, :] = np.roll(padded_tile[offset, :], roll_amount)

    # Create shear coefficients
    sqrt3 = np.sqrt(3)
    delay1 = sqrt3 - np.sqrt(6 / sqrt3)
    delay2 = np.sqrt(sqrt3 / 6)
    delay3 = 2 - np.sqrt(6 / sqrt3)

    # Apply shears
    shear(padded_tile, delay3, axis=0)
    shear(padded_tile, delay2, axis=1)
    shear(padded_tile, delay1, axis=0)

    # Extract the rectangular box
    height = round(np.dot(r_cart, [tile.shape[0], 0])[0])
    width = round(np.dot(r_cart, [0, tile.shape[1]])[1])

    # Should we use -half_width or -width // 2? These are not the same
    # because of the behaviour of //, which does truncation towards 0:
    # -(119 // 2) == -59
    # -119 // 2 == -60
    half_width = width // 2
    half_height = height // 2

    return padded_tile[cy-half_height : cy+half_height+(height % 2),
                       cx-half_width : cx+half_width+(width % 2)]
