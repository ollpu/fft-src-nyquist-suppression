# FFT-based resmpling

from lib.util import *
import numpy as np

def fft_resample(input, taper, output_len):
    """Length of taper should be smaller of input/output"""

    ratio = output_len / len(input)

    input_f = np.fft.fft(input)

    if ratio >= 1: # Upsample
        input_f *= taper

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

        output_f *= taper

    return np.sqrt(output_len / len(input)) * np.real(np.fft.ifft(output_f))
