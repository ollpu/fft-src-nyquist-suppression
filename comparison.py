# %%
import re
import matplotlib.pyplot as plt
import scipy.signal

from lib.tabular import Tabular
from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plot = False


# %%

conversions = [
    (44100, 192000, 1),
    # (44100, 96000, 10),
    # (44100, 192000, 100),

    # (96000, 44100, 1),
    # (96000, 44100, 10),
    # (96000, 44100, 100),
]

L_prop = 0.05 # of smaller Nyquist

tapers = {
    # 'Box': 'box',
    'Cosine': 'cosine',
    # 'Hann': 'hann',
    # 'Blackman': 'blackman',
    'Dolph--Chebyshev': ('chebwin', 95.8),
    'DDC $\\alpha=1/2$': ('ddc', 150, 0.5),
    'DDC optimal': ('ddc', 150),
}

target_dB = -150
columns = {
    "name_and_psl": r"Window (PSL in dB)",
    "main_lobe_width": r"MLW\,(ms)",
    "integrated_sidelobe_level": r"ISL\,(dB)",
    "time_to_dB": fr"T{-target_dB}\,(ms)",
}

def highlight_cell(name, key):
    if name == "Cosine" and key == "main_lobe_width":
        return True
    if name == "DDC optimal" and key == "time_to_dB":
        return True
    if name == "Dolph--Chebyshev" and key == "integrated_sidelobe_level":
        return True
    return False

# %% Analyses

def peak_magnitude(table_row, ir):
    print(f"- Peak magnitude:\t\t {amp2db(np.max(np.abs(ir))):.2f} dB")

def time_to_dB(table_row, ir, Fs):
    tolerance = 0.05
    attenuated = np.abs(ir) < db2amp(target_dB + tolerance)
    left = np.argmin(attenuated)
    right = len(ir) - np.argmin(attenuated[::-1])
    width_ms = 1000 * (right-left) / Fs

    print(f"- Time above {target_dB} dB:\t\t {width_ms:.2f} ms")

    decimals = 1 if width_ms >= 10 else 2
    table_row["time_to_dB"] = f"${width_ms:.{decimals}f}$"
    if left == 0:
        table_row["time_to_dB"] = "--"
        left = None
        right = None

    return left, right, width_ms

def main_lobe_width(window, Fs, samples):
    buf = np.zeros(samples)
    buf[:len(window)] = window
    buf = amp2db(np.fft.fft(buf))
    peaks, _ = scipy.signal.find_peaks(-buf, distance=3)

    width = 2 * peaks[0]
    width_ms = 1000 * width / Fs

    print(f"- Main lobe width:\t\t {width_ms:.2f} ms")

    table_row["main_lobe_width"] = f"${width_ms:.2f}$"

    return width

def peak_sidelobe_level(table_row, ir, left, right):
    idx = np.arange(len(ir))
    mask = (idx < left) | (idx > right)

    psl = amp2db(np.max(ir[mask]))
    print(f"- Peak sidelobe level:\t\t {psl:.2f} dB")

    table_row["peak_sidelobe_level"] = f"${psl:.0f}$"
    table_row["name_and_psl"] = f"{table_row["name"]} ({table_row["peak_sidelobe_level"]})"

    return psl

def integrated_sidelobe_level(table_row, ir, left, right):
    ir_s = ir**2
    idx = np.arange(len(ir))
    mask = (idx < left) | (idx > right)

    isl = pow2db(np.sum(ir_s[mask]) / np.sum(ir_s[~mask]))
    print(f"- Integrated sidelobe level:\t {isl:.2f} dB")


    table_row["integrated_sidelobe_level"] = f"${isl:.0f}$"

    return isl

# %%

fract_offset = 0

for Fs_in, Fs_out, duration in conversions:
    N = int(Fs_in * duration)
    M = int(Fs_out * duration)
    smaller = min(N, M)

    print("\n\n")
    print(("Upsample" if M > N else "Downsample"), Fs_in, "to", Fs_out, "duration", duration, "s")
    print("==========")

    table = Tabular(columns, highlight_cell)

    for name, spec in tapers.items():
        print("\nTaper:", name)

        table_row = {"name": name}
        table.append(table_row)

        l = int(L_prop * smaller / 2)
        taper, meta = get_taper(spec, smaller, l)
        window = meta['window']
        del meta['window']

        print(f"Meta: L={l}, fract_offset={fract_offset}, {meta}\n")


        test_input = np.zeros(N)
        # The lengths are all even, so midpoint should have no fractional offset
        test_input[N // 2 + fract_offset] = 1

        output = fft_resample(test_input, taper, M)

        if fract_offset != 0:
            # Calibrate peak magnitude by having an impulse at 0, where we know there is no fractional offset
            cal_input = np.zeros(N)
            cal_input[0] = 1
            cal_output = fft_resample(cal_input, taper, M)
            norm_factor = 1 / cal_output[0]
        else:
            norm_factor = 1 / output[M // 2]

        output *= norm_factor

        peak_magnitude(table_row, output)
        time_to_dB(table_row, output, Fs_out)
        mlw = main_lobe_width(window, Fs_out, M)
        peak_pos = np.argmax(output)
        left = peak_pos - mlw/2
        right = peak_pos + mlw/2
        peak_sidelobe_level(table_row, output, left, right)
        integrated_sidelobe_level(table_row, output, left, right)


        if plot:
            plt.plot(amp2db(output))
            plt.axvline(left, c='red')
            plt.axvline(right, c='red')
            plt.axhline(target_dB, c='red')
            plt.title(name)
            plt.xlim(M//2-2000, M//2+2000)
            plt.ylim(-300, 5)
            plt.show()

    print()
    print(table.fmt_table())
