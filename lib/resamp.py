# FFT-based resmpling

from lib.util import *
import numpy as np

def fft_resample(input, taper, output_len):
    """Takes full taper with length equal to smaller of input/output buffers"""

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
    """Takes only the positive side transition band of taper function (decreasing from 1 to 0)"""

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
    m = len(buf)
    l = len(transition)
    end_pos = (m + 1) // 2
    start_pos = end_pos - l
    buf[start_pos:end_pos] *= transition
    start_pos = end_pos + 1
    end_pos = start_pos + l
    buf[start_pos:end_pos] *= transition[::-1]
