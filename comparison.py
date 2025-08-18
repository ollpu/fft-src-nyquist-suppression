# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import sounddevice as sd

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plot = True


# %%

conversions = [
    # (44100, 96000, 1),
    # (44100, 96000, 10),
    # (44100, 96000, 100),

    (96000, 44100, 1),
    # (96000, 44100, 10),
    # (96000, 44100, 100),
]

L_prop = 0.05 # of smaller Nyquist

tapers = {
    'Box': 'box',
    'Cosine': 'cosine',
    'Hann': 'hann',
    'Blackman': 'blackman',
    'Dolph–Chebyshev': ('chebwin', 90),
    'DDC alpha=1/2': ('ddc', 145, 0.5),
    'DDC optimal': ('ddc', 145),
}
# %%

target_dB = -144
def time_to_dB(ir, Fs):
    attenuated = np.abs(ir) < db2amp(target_dB)
    left = np.argmin(attenuated)
    right = len(ir) - np.argmin(attenuated[::-1])
    width_ms = 1000 * (right-left) / Fs

    print("Time to decay to", target_dB, "dB:\t", width_ms, "ms")

    return left, right, width_ms

# Other analyses:
# - Main lobe width (how to find this?)
# - Energy outside "time_to_dB" width
# - Energy outside main lobe (integrated sidelobe level)



# %%

for Fs_in, Fs_out, duration in conversions:
    N = int(Fs_in * duration)
    M = int(Fs_out * duration)
    smaller = min(N, M)

    print("\n\n")
    print(("Upsample" if M > N else "Downsample"), Fs_in, "to", Fs_out, "duration", duration, "s")
    print("==========")

    for name, spec in tapers.items():
        print("\nTaper:", name)

        l = int(L_prop * smaller / 2)
        taper, meta = get_taper(spec, smaller, l)

        print("Meta:", meta, "\n")

        input = np.zeros(N)
        # TODO: how should we choose the exact impulse position?
        input[N // 2 + 3] = 1

        output = fft_resample(input, taper, M)
        output /= np.sqrt(Fs_out / Fs_in)

        left, right, width_ms = time_to_dB(output, Fs_out)

        if plot:
            plt.plot(amp2db(output))
            plt.axvline(left, c='red')
            plt.axvline(right, c='red')
            plt.axhline(target_dB, c='red')
            plt.title(name)
            plt.xlim(M//2-2000, M//2+2000)
            plt.ylim(-300, 5)
            plt.show()
