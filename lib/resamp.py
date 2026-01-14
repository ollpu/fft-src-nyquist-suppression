"""
Functions to perform FFT-based resmpling
"""

import numpy as np

def fft_resample(input, taper, output_len):
    """Perform FFT-based resampling, given the full tapering function.

    Args:
        input: Real-valued input signal of length N
        taper: Taper function of length M = min(N, Nprime),
            or None for no tapering.
        output_len: Desired output signal length Nprime

    Returns:
        Resampled output signal of length Nprime
    """

    ratio = output_len / len(input)

    input_f = np.fft.fft(input)

    if ratio >= 1: # Upsample
        if taper is not None: input_f *= taper

        output_f = np.zeros(output_len, dtype='complex')
        pos_bins = (len(input)+1)//2
        neg_bins = (len(input)-1)//2
        output_f[:pos_bins] = input_f[:pos_bins]
        output_f[-neg_bins:] = input_f[-neg_bins:]
    else: # Downsample
        output_f = np.zeros(output_len, dtype='complex')
        pos_bins = (output_len+1)//2
        neg_bins = (output_len-1)//2
        output_f[:pos_bins] = input_f[:pos_bins]
        output_f[-neg_bins:] = input_f[-neg_bins:]

        if taper is not None: output_f *= taper

    return ratio * np.real(np.fft.ifft(output_f))

def fft_resample_transition(input, taper_transition, output_len):
    """Perform FFT-based resampling, given one transition band of the taper function.

    Args:
        input: Real-valued input signal of length N
        taper_transition: Positive transition band of the tapering function, length L.
            The topmost L frequency bands below Nyquist are tapered with this function.
            Should decrease from 1 to 0. Pass None for no tapering.
        output_len: Desired output signal length Nprime

    Returns:
        Resampled output signal of length Nprime
    """

    ratio = output_len / len(input)

    input_f = np.fft.fft(input)

    if ratio >= 1: # Upsample
        if taper_transition is not None: apply_taper_transition(input_f, taper_transition)

        output_f = np.zeros(output_len, dtype='complex')
        pos_bins = (len(input)+1)//2
        neg_bins = (len(input)-1)//2
        output_f[:pos_bins] = input_f[:pos_bins]
        output_f[-neg_bins:] = input_f[-neg_bins:]
    else: # Downsample
        output_f = np.zeros(output_len, dtype='complex')
        pos_bins = (output_len+1)//2
        neg_bins = (output_len-1)//2
        output_f[:pos_bins] = input_f[:pos_bins]
        output_f[-neg_bins:] = input_f[-neg_bins:]

        if taper_transition is not None: apply_taper_transition(input_f, taper_transition)

    return ratio * np.real(np.fft.ifft(output_f))

def apply_taper_transition(buf, transition):
    M = len(buf)
    L = len(transition)
    end_pos = (M + 1) // 2
    start_pos = end_pos - L
    buf[start_pos:end_pos] *= transition
    start_pos = end_pos + 1
    end_pos = start_pos + L
    buf[start_pos:end_pos] *= transition[::-1]
