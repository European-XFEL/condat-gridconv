cimport cython

@cython.cdivision(True)
cdef inline int get_index(int i, int i_max):
    # This helper function implements the behaviour of
    # Pythons modulo operator, which differs from C's remainder
    # operator. We need to do this ourselves because this is needed
    # in hot loops where it's expensive to call CPython's modulo function.
    return ((i % i_max) + i_max) % i_max

@cython.boundscheck(False)
@cython.wraparound(False)
def inplace_roll(double[:] array, int shift):
    cdef int arr_len = len(array)
    cdef int abs_shift = abs(shift)

    if arr_len == abs_shift or shift == 0:
        return

    cdef bint rshift = shift > 0

    # Create a buffer to hold some elements so there's space for the roll
    cdef int buffer_start_idx = arr_len - 1 - abs_shift if rshift else 0
    cdef int buffer_end_idx = arr_len if rshift else abs_shift
    cdef double[:] buffer = array[buffer_start_idx:buffer_end_idx].copy()

    # Roll all elements except for the buffered ones
    cdef int range_start = arr_len - shift - 1 if rshift else abs_shift
    cdef int range_end = -1 if rshift else arr_len
    cdef int range_step = -1 if rshift else 1

    cdef int i

    for i in range(range_start, range_end, range_step):
        array[get_index(i + shift, arr_len)] = array[i]

    # Roll the buffered elements
    cdef int buf_len = len(buffer)
    for i in range(buf_len):
        array[get_index(buffer_start_idx + i + shift, arr_len)] = buffer[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fractional_roll(double[:] signal, double tau):
    """Shift a 1D array of pixels by a fractional amount

    tau, the shift in pixels, should be between -0.5 & 0.5.

    This implements a Type II filter for N = 2, as described in
    Condat et al. 2008, *Reversible, Fast, and High-Quality Grid
    Conversions*, IEEE Transactions on Image Processing.
    There is a simple implementation for N = 2 (p 686 in the paper);
    generalising to other values of N would be more complex.

    The data in signal is shifted in place.
    """
    cdef:
        double a_numerator, a_plus_tau, a_minus_tau
        int j, n

    if tau < -0.5 or tau > 0.5:
        raise ValueError(f"Condat filter delay is out of range [-0.5, 0.5]: {tau}")

    a_numerator = -4 + tau**2 + (12 - 3 * tau**2) ** 0.5
    a_plus_tau = a_numerator / (3 * tau + 2 + tau**2)
    a_minus_tau = a_numerator / (-3 * tau + 2 + tau**2)

    n = len(signal)
    for j in range(1, n - 1):
        signal[j] += a_plus_tau * (signal[j - 1] - signal[j + 1])
    for j in range(n - 2, 0, -1):
        signal[j] += a_minus_tau * (signal[j + 1] - signal[j - 1])
