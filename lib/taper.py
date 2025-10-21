# Taper functions for frequency domain filtering

import warnings
from lib.util import *
import numpy as np
import scipy.signal

def get_window(win_spec, M, L):
    """M is total length of fft buffer (only used by DDC), L is length of window"""
    match win_spec:
        # case ('ultra', sidelobes_dB, alpha):
        #     sidelobes_R = db2amp(sidelobes_dB)

        #     order = L
        #     add = 0
        #     if order%2 == 0:
        #         order -= 1
        #         add = 1
        #     d = np.cosh(1/(order-1)* np.acosh(sidelobes_R))
        #     phi = np.arange(order)*np.pi/order
        #     z = d*np.cos(phi)
        #     c = [2*alpha*z, -alpha + 2*alpha*(1+alpha)*(z**2)]

        #     for k in range(3, order):
        #         c.append(2*(k+alpha-1)/k*z*c[-1] - (k+2*alpha-2)/k*c[-2])

        #     u = c[-1]

        #     win = np.fft.fftshift(np.real(np.fft.ifft(u)))
        #     win /= np.sum(win)
        #     win = np.pad(win, (0, add))
        #     return win, { 'R': sidelobes_R, 'd': d, 'order': order-1 }

        case ('chebwin', sidelobes_dB):
            # Seems to be faster than scipy implementation

            sidelobes_R = db2amp(sidelobes_dB)
            order = L-1

            x0 = np.cosh(np.acosh(sidelobes_R) / order)

            Lr = L//2 + 1

            phi = np.arange(Lr)*np.pi/L
            z = x0*np.cos(phi)

            w = np.zeros(Lr, dtype=complex)
            zm = np.abs(z) <= 1
            w[zm] = np.cos(order*np.acos(z[zm]))
            zm = z > 1
            w[zm] = np.cosh(order*np.acosh(z[zm]))
            zm = z < -1
            w[zm] = (-1)**order * np.cosh(order*np.acosh(-z[zm]))

            if L%2 == 0:
                w *= np.exp(1j * phi)

            W = np.fft.fftshift(np.fft.irfft(w, L))
            W /= np.sum(W)

            return W, { 'R': sidelobes_R, 'd': x0, 'order': order }

        case ('ddc', sidelobes_dB, *rest):
            sidelobes_R = db2amp(sidelobes_dB)

            order = L-1

            set_alpha = next(iter(rest), None)

            def DQ(y):
                alpha = set_alpha if set_alpha else (1 + np.sqrt(y**2-1)/y)/2
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    Q = alpha*np.cosh(order*np.acosh(y)) - (1-alpha)*np.cosh((order-2)*np.acosh(y))
                    return (M - L) * Q

            x0 = binary_search(DQ, (1, 2), sidelobes_R)

            alpha = set_alpha if set_alpha else (1 + np.sqrt(x0**2-1)/x0)/2

            Lr = L//2 + 1

            phi = np.arange(Lr)*np.pi/L
            z = x0*np.cos(phi)

            w = np.zeros(Lr, dtype=complex)
            zm = np.abs(z) <= 1
            w[zm] = alpha*np.cos(order*np.acos(z[zm])) - (1-alpha)*np.cos((order-2)*np.acos(z[zm]))
            zm = z > 1
            w[zm] = alpha*np.cosh(order*np.acosh(z[zm])) - (1-alpha)*np.cosh((order-2)*np.acosh(z[zm]))
            zm = z < -1
            w[zm] = (-1)**order * (alpha*np.cosh(order*np.acosh(-z[zm])) - (1-alpha)*np.cosh((order-2)*np.acosh(-z[zm])))

            if L%2 == 0:
                w *= np.exp(1j * phi)

            W = np.fft.fftshift(np.fft.irfft(w, L))
            W /= np.sum(W)

            return W, { 'R': sidelobes_R, 'd': x0, 'alpha': alpha, 'order': order }

        case _:
            W = scipy.signal.windows.get_window(win_spec, L, fftbins=False)
            W /= np.sum(W)
            return W, {}

def get_taper_transition(taper_spec, M, L):
    window, meta = get_window(taper_spec, M, L)
    meta['window'] = window
    return np.cumsum(window)[::-1], meta

def get_taper(taper_spec, M, L):
    S = np.zeros(M)

    end_pos = (M + 1) // 2
    start_pos = end_pos - L

    # mid = (start_pos + end_pos) // 2
    S[:end_pos] = 1

    meta = {}
    if taper_spec != 'box':
        transition, meta = get_taper_transition(taper_spec, M, L)
        S[start_pos:end_pos] = transition

    S[(M+2)//2:] = S[(M-1)//2:0:-1]
    return S, meta
