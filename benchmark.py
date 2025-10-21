# %%
import re
import time
import matplotlib.pyplot as plt
import scipy.signal

from lib.tabular import Tabular
from lib.util import *
from lib.taper import get_taper_transition
# from lib.resamp import fft_resample

plot = False

# %%

conversions = [
    (44100, 50),
]

L_prop = 0.05 # of smaller Nyquist

tapers = {
    'Cosine': 'cosine',
    'DDC optimal': ('ddc', 144),
    'Dolph--Chebyshev': ('chebwin', 90.76),
}

fir_length_s = 0.00617
fir_length_samp = int(44100 * fir_length_s)
if fir_length_samp % 2 == 0: fir_length_samp += 1
print("FIR points =", fir_length_samp)

columns = {
    "name": r"Method",
    "design_time": r"Design (ms)",
    "exec_time": r"Execution (ms)",
}

def highlight_cell(name, key):
    return "FIR" not in name and key == "exec_time"

# %%

def bench(fn, setup=None, target_time=1.0):
    # Warm up once to ensure imports etc are resolved
    if setup: setup()
    fn()

    init = time.perf_counter()
    runs = 0
    measured_time = 0.
    while time.perf_counter() - init < target_time:
        if setup: setup()
        start = time.perf_counter()
        fn()
        measured_time += time.perf_counter() - start
        runs += 1
    print(runs, "runs done")
    return measured_time / runs

def apply_taper_transition(buf, transition):
    m = len(buf)
    l = len(transition)
    end_pos = (m + 1) // 2
    start_pos = end_pos - l
    buf[start_pos:end_pos] *= transition
    start_pos = end_pos + 1
    end_pos = start_pos + l
    buf[start_pos:end_pos] *= transition[::-1]

def fmt_time(time_s):
    time_ms = time_s * 1000
    if time_ms > 10: return f"{time_ms:.1f}"
    else: return f"{time_ms:.2f}"

def fmt_design_time(table_row, time_s):
    num = fmt_time(time_s)
    print(f"- Design:\t {num} ms")
    table_row["design_time"] = f"${num}$"

def fmt_exec_time(table_row, time_s):
    num = fmt_time(time_s)
    print(f"- Execution:\t {num} ms")
    table_row["exec_time"] = f"${num}$"

for Fs, duration in conversions:
    M = int(Fs * duration)
    L = int(L_prop * M / 2)

    print("\n\n")
    print("Filter", Fs, "duration", duration, "s")
    print("==========")

    table = Tabular(columns, highlight_cell)

    sig = np.random.rand(M)

    for name, spec in tapers.items():
        print("\nTaper:", name)

        table_row = {"name": name}
        table.append(table_row)

        buf = np.fft.fft(sig)

        # Prepare
        def measure1():
            global transition
            transition, _ = get_taper_transition(spec, M, L)

        fmt_design_time(table_row, bench(measure1))

        # Perform
        def setup2():
            global buf
            buf = np.fft.fft(sig)

        def measure2():
            apply_taper_transition(buf, transition)

        fmt_exec_time(table_row, bench(measure2, setup2))

    print("\nFIR OLA")
    table_row = {"name": f"FIR OLA"}
    table.append(table_row)

    # Prepare
    def measure3():
        global kernel
        kernel = scipy.signal.firwin(fir_length_samp, 1 - L_prop, window=('chebwin', 150))

    fmt_design_time(table_row, bench(measure3))

    # Perform OLA convolution
    def measure4():
        global result_ms
        result_ms = scipy.signal.oaconvolve(sig, kernel, mode='same')

    fmt_exec_time(table_row, bench(measure4))

    print("\nFIR FFT")
    table_row = {"name": f"FIR FFT"}
    table.append(table_row)

    # Prepare frequency-domain FIR (FFT of zero padded kernel)
    def measure5():
        global kernel, freq_fir
        kernel = scipy.signal.firwin(fir_length_samp, 1 - L_prop, window=('chebwin', 150))
        kernel_padded = np.zeros(M)
        kernel_padded[:(len(kernel) + 1)//2] = kernel[len(kernel)//2:]
        kernel_padded[-(len(kernel)//2):] = kernel[:len(kernel)//2]
        freq_fir = np.fft.fft(kernel_padded)

    fmt_design_time(table_row, bench(measure5))

    # Perform frequency-domain multiplication
    def setup6():
        global buf
        buf = np.fft.fft(sig)

    def measure6():
        global buf
        buf *= freq_fir

    fmt_exec_time(table_row, bench(measure6, setup6))

    print()
    print(table.fmt_table())
