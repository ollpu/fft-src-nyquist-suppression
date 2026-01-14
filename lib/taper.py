"""
Taper functions for frequency domain filtering
"""

import warnings
from lib.util import db2amp, binary_search
import numpy as np
import scipy.signal

def get_window(win_spec, M, L):
    """Generate a window function S (Dolph-Chebyshev, DDC, or Scipy get_window)

    The window is normalized so that sum(S) = 1.

    Args:
        win_spec: Window name or tuple.

            For a Dolph-Chebyshev window, pass ('chebwin', R),
            where R is the main-to-sidelobe ratio in decibels.

            For the double Dolph-Chebyshev (DDC) window, pass ('ddc', R).
            The parameter alpha is solved according to Eq. (9), but
            it can also be fixed by passing e.g. ('ddc', R, 0.5).
            The parameter x0 may also be fixed by directly with ('ddc', 0, alpha, x0).

            For other window specs, this falls back to scipy.signal.get_window.

        M: The length of the complete tapering function,
            where this window is to be used as a kernel.
            Required by DDC to solve x0 according to Eq. (10).
        L: The length of the window

    Returns:
        Window function of length L and a dict of metadata
    """
    """M is total length of fft buffer (only used by DDC), L is length of window"""

    match win_spec:
        case ('chebwin', sidelobes_dB):
            # Seems to be faster than scipy implementation

            sidelobes_R = db2amp(sidelobes_dB)
            order_m = L-1

            x0 = np.cosh(np.acosh(sidelobes_R) / order_m)

            Lr = L//2 + 1

            t = np.arange(Lr) * np.pi/L
            x = x0 * np.cos(t)

            s = np.zeros(Lr, dtype=complex)
            x_mask = np.abs(x) <= 1
            s[x_mask] = np.cos(order_m*np.acos(x[x_mask]))
            x_mask = x > 1
            s[x_mask] = np.cosh(order_m*np.acosh(x[x_mask]))
            x_mask = x < -1
            s[x_mask] = (-1)**order_m * np.cosh(order_m*np.acosh(-x[x_mask]))

            if L%2 == 0:
                s *= np.exp(1j * t)

            S = np.fft.fftshift(np.fft.irfft(s, L))
            S /= np.sum(S)

            return S, { 'R': sidelobes_R, 'x0': x0, 'order': order_m }

        case ('ddc', sidelobes_dB, *rest):
            sidelobes_R = db2amp(sidelobes_dB)

            order_m = L-1

            rest_iter = iter(rest)
            set_alpha = next(rest_iter, None)
            set_x0 = next(rest_iter, None)


            def DQ(x0):
                """Computes the product D_c(0) * Q_m(x_0), as in Eq. (10)."""
                alpha = set_alpha if set_alpha else (1 + np.sqrt(x0**2-1)/x0)/2
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    Q = alpha*np.cosh(order_m*np.acosh(x0)) - (1-alpha)*np.cosh((order_m-2)*np.acosh(x0))
                    return (M - L) * Q

            # Find x0 by solving Eq. (10) using binary search
            x0 = set_x0 if set_x0 else binary_search(DQ, (1, 2), sidelobes_R)

            alpha = set_alpha if set_alpha else (1 + np.sqrt(x0**2-1)/x0)/2

            Lr = L//2 + 1

            t = np.arange(Lr) * np.pi/L
            x = x0 * np.cos(t)

            s = np.zeros(Lr, dtype=complex)
            x_mask = np.abs(x) <= 1
            s[x_mask] = alpha*np.cos(order_m*np.acos(x[x_mask])) - (1-alpha)*np.cos((order_m-2)*np.acos(x[x_mask]))
            x_mask = x > 1
            s[x_mask] = alpha*np.cosh(order_m*np.acosh(x[x_mask])) - (1-alpha)*np.cosh((order_m-2)*np.acosh(x[x_mask]))
            x_mask = x < -1
            s[x_mask] = (-1)**order_m * (alpha*np.cosh(order_m*np.acosh(-x[x_mask])) - (1-alpha)*np.cosh((order_m-2)*np.acosh(-x[x_mask])))

            if L%2 == 0:
                s *= np.exp(1j * t)

            S = np.fft.fftshift(np.fft.irfft(s, L))
            S /= np.sum(S)

            return S, { 'R': sidelobes_R, 'x0': x0, 'alpha': alpha, 'order': order_m }

        case _:
            S = scipy.signal.windows.get_window(win_spec, L, fftbins=False)
            S /= np.sum(S)
            return S, {}

def get_taper_transition(win_spec, M, L):
    """Generate the transition band of a tapering function

    Takes the (backwards) cumulative sum of a kernel window defined by win_spec.

    Returns:
        The taper transition band (L points) and a dict of metadata
    """
    window, meta = get_window(win_spec, M, L)
    meta['window'] = window
    return np.cumsum(window)[::-1], meta

def get_taper(taper_spec, M, L):
    """Generate full taper function of length M, with L point transition bands.

    Frequencies follow the Numpy FFT convention,
    where negative frequencies are at the end of the buffer.

    Args:
        taper_spec: Either a kernel window spec (see get_window)
            or 'box', meaning no tapering.
        M: Length of the tapering function (positive and negative side).
        L: Length of the kernel window.

    Returns:
        Taper function W of length M and a dict of metadata
    """
    W = np.zeros(M)

    end_pos = (M + 1) // 2
    start_pos = end_pos - L

    W[:end_pos] = 1

    meta = {}
    if taper_spec != 'box':
        transition, meta = get_taper_transition(taper_spec, M, L)
        W[start_pos:end_pos] = transition

    W[(M+2)//2:] = W[(M-1)//2:0:-1]
    return W, meta
