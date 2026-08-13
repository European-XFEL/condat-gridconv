import numpy as np

from .shift import fractional_roll, inplace_roll

sqrt3 = np.sqrt(3)
r_hex = np.sqrt(2 / sqrt3) * np.array([[1, 1 / 2],
                                       [0, sqrt3 /2 ]])
r_cart = np.linalg.inv(r_hex)

def pixel_hex2cart(x_hex, y_hex):
    """Converts hexagonal pixels position to cartesian.
    """
    x_cart = np.dot(r_cart, [0, x_hex])[1]
    y_cart = np.dot(r_cart, [y_hex, 0])[0]

    return x_cart, y_cart

def pixel_cart2hex(x_cart, y_cart):
    """Converts cartesian pixel position to hexagonal.
    """
    x_hex = np.dot(r_hex, [0, x_cart])[1]
    y_hex = np.dot(r_hex, [y_cart, 0])[0]

    return x_hex, y_hex

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

def inplace_shift(arr, dx, dy):
    """Apply a shift operation in place

    Shift each column (axis=0) or row (axis=1) in the array by an offset
    """
    cx = arr.shape[1]
    cy = arr.shape[0]

    # Vertical shear
    for k in range(cx):
        shift = dy
        full_pixel_shift = round(shift)

        # Full pixel shifts
        inplace_roll(arr[:, k], full_pixel_shift)

        # Sub-pixel shifts
        fractional_roll(arr[:, k], shift - full_pixel_shift)

    # Horizontal shear
    for k in range(cy):
        shift = dx
        full_pixel_shift = round(shift)

        # Full pixel shifts
        inplace_roll(arr[k, :], full_pixel_shift)

        # Sub-pixel shifts
        fractional_roll(arr[k, :], shift - full_pixel_shift)

def pad(tile, fill_value=None):
    w0 = tile.shape[0] // 2
    w1 = tile.shape[1] // 2

    # Can we assume that a tile length is always a power of 2?
    width0 = (w0, w0) if tile.shape[0] % 2 == 0 else (w0, w0 - 1)
    width1 = (w1, w1) if tile.shape[1] % 2 == 0 else (w1, w1 - 1)

    if fill_value is not None:
        return np.pad(
            tile, (width0, width1), mode="constant", constant_values=fill_value
        )
    else:
        return np.pad(tile, (width0, width1), mode="reflect")


def hex_shift(tile, dx, dy, fill_value=None):
    """Shift an hexagonal image with fractional pixel shifts."""
    padded_tile = pad(tile, fill_value)
    cy = int(padded_tile.shape[0] / 2)
    cx = int(padded_tile.shape[1] / 2)

    # First we skew the image to a hexagonal shape
    height = padded_tile.shape[0]
    for y in range(height):
        roll_amount = -int(np.floor((y - cy) / 2))
        padded_tile[y, :] = np.roll(padded_tile[y, :], roll_amount)

    # Apply shifts
    inplace_shift(padded_tile, dx, dy)

    # Unskew the image to a hexagonal shape
    for y in range(height):
        roll_amount = int(np.floor((y - cy) / 2))
        padded_tile[y, :] = np.roll(padded_tile[y, :], roll_amount)

    # Extract the rectangular box
    height = tile.shape[0]
    width = tile.shape[1]

    # // rounds towards zero, so -half_width differs from -width // 2.
    half_width = width // 2
    half_height = height // 2

    return padded_tile[cy-half_height : cy+half_height+(height % 2),
                       cx-half_width : cx+half_width+(width % 2)]

def hex2cart(tile, fill_value=None):
    padded_tile = pad(tile, fill_value)
    cy = int(padded_tile.shape[0] / 2)
    cx = int(padded_tile.shape[1] / 2)

    # First we skew the image to a hexagonal shape
    height = padded_tile.shape[0]
    for y in range(height):
        roll_amount = -int(np.floor((y - cy) / 2))
        padded_tile[y, :] = np.roll(padded_tile[y, :], roll_amount)

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
    width, height = pixel_hex2cart(tile.shape[1], tile.shape[0])
    height = round(height)
    width = round(width)

    # // rounds towards zero, so -half_width differs from -width // 2.
    half_width = width // 2
    half_height = height // 2

    return padded_tile[cy-half_height : cy+half_height+(height % 2),
                       cx-half_width : cx+half_width+(width % 2)]

def cart2hex(tile, fill_value=None):
    padded_tile = pad(tile, fill_value)
    cy = padded_tile.shape[0] // 2
    cx = padded_tile.shape[1] // 2

    # Create shear coefficients
    sqrt3 = np.sqrt(3)
    delay1 = sqrt3 - np.sqrt(6 / sqrt3)
    delay2 = np.sqrt(sqrt3 / 6)
    delay3 = 2 - np.sqrt(6 / sqrt3)

    # Apply shears in reverse order and opposite direction
    shear(padded_tile, -delay1, axis=0)
    shear(padded_tile, -delay2, axis=1)
    shear(padded_tile, -delay3, axis=0)

    # unskew the image to a hexagonal shape
    height = padded_tile.shape[0]
    for y in range(height):
        roll_amount = int(np.floor((y - cy) / 2))
        padded_tile[y, :] = np.roll(padded_tile[y, :], roll_amount)

    # Extract the rectangular box
    width, height = pixel_cart2hex(tile.shape[1], tile.shape[0])
    height = round(height)
    width = round(width)

    # // rounds towards zero, so -half_width differs from -width // 2.
    half_width = width // 2
    half_height = height // 2

    return padded_tile[cy-half_height : cy+half_height+(height % 2),
                       cx-half_width : cx+half_width+(width % 2)]

def rotate(tile, angle, fill_value=None):
    """Rotate an image reversibly using 3 shears.

    tile: array, image to rotate
    angle: float, angle of rotation in degree.
    fill_value: float, fill value for padding the tile, if None, then pad
        the image with reflections of itself.
    """
    padded_tile = pad(tile, fill_value)
    cy = padded_tile.shape[0] // 2
    cx = padded_tile.shape[1] // 2

    # Create shear coefficients
    theta = np.deg2rad(angle)
    delay1 = -np.tan(theta/2.0)
    axis1 = 1
    delay2 = np.sin(theta)
    axis2 = 0
    delay3 = -np.tan(theta/2.0)
    axis3 = 1

    # Apply shears in reverse order and opposite direction
    shear(padded_tile, delay1, axis=axis1)
    shear(padded_tile, delay2, axis=axis2)
    shear(padded_tile, delay3, axis=axis3)

    # Extract the rectangular box
    height = tile.shape[0]
    width = tile.shape[1]

    # // rounds towards zero, so -half_width differs from -width // 2.
    half_width = width // 2
    half_height = height // 2

    return padded_tile[cy-half_height : cy+half_height+(height % 2),
                       cx-half_width : cx+half_width+(width % 2)]
