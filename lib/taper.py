# Taper functions for frequency domain filtering

import warnings
from lib.util import *
import numpy as np
import scipy.signal

def get_window(win_spec, M, L):
    """M is total length of fft buffer (only used by DDC), L is length of window"""
    match win_spec:
        case ('ultra', sidelobes_dB, alpha):
            sidelobes_R = db2amp(sidelobes_dB)

            N = L
            add = 0
            if N%2 == 0:
                N -= 1
                add = 1
            d = np.cosh(1/(N-1)* np.acosh(sidelobes_R))
            phi = np.arange(N)*np.pi/N
            z = d*np.cos(phi)
            c = [2*alpha*z, -alpha + 2*alpha*(1+alpha)*(z**2)]

            for k in range(3, N):
                c.append(2*(k+alpha-1)/k*z*c[-1] - (k+2*alpha-2)/k*c[-2])

            u = c[-1]

            win = np.fft.fftshift(np.real(np.fft.ifft(u)))
            win /= np.sum(win)
            win = np.pad(win, (0, add))
            return { 'win': win, 'R': sidelobes_R, 'd': d, 'order': N-1 }

        case ('ddc', sidelobes_dB, *rest):
            sidelobes_R = db2amp(sidelobes_dB)

            N = L-1
            if N%2: N -= 1

            set_alpha = next(iter(rest), None)

            def DQ(y):
                alpha = set_alpha if set_alpha else (1 + np.sqrt(y**2-1)/y)/2
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    Q = alpha*np.cosh(N*np.acosh(y)) - (1-alpha)*np.cosh((N-2)*np.acosh(y))
                return (M - L) * Q

            d = binary_search(DQ, (1, 2), sidelobes_R)

            alpha = set_alpha if set_alpha else (1 + np.sqrt(d**2-1)/d)/2

            phi = np.arange(L)*np.pi/L
            z = d*np.cos(phi)

            w = np.zeros(L)
            zm = np.abs(z) <= 1
            w[zm] = alpha*np.cos(N*np.acos(z[zm])) - (1-alpha)*np.cos((N-2)*np.acos(z[zm]))
            zm = z > 1
            w[zm] = alpha*np.cosh(N*np.acosh(z[zm])) - (1-alpha)*np.cosh((N-2)*np.acosh(z[zm]))
            zm = z < -1
            w[zm] = (-1)**N * (alpha*np.cosh(N*np.acosh(-z[zm])) - (1-alpha)*np.cosh((N-2)*np.acosh(-z[zm])))

            win = np.fft.fftshift(np.real(np.fft.ifft(w)))
            win /= np.sum(win)

            return { 'win': win, 'R': sidelobes_R, 'd': d, 'alpha': alpha, 'order': N }

        case _:
            win = scipy.signal.windows.get_window(win_spec, L, fftbins=False)
            win /= np.sum(win)
            return { 'win': win }

def get_taper(taper_spec, M, L):
    S = np.zeros(M)

    end_pos = (M + 1) // 2
    start_pos = end_pos - L

    mid = (start_pos + end_pos) // 2
    S[:mid] = 1

    result = {}
    if taper_spec != 'box':
        result = get_window(taper_spec, M, L)
        S[start_pos:end_pos] = np.cumsum(result['win'])[::-1]

    S[(M+2)//2:] = S[(M-1)//2:0:-1]
    result['taper'] = S
    return result
