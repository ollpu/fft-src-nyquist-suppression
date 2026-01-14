"""
Table II: Computational overhead benchmark
"""

# %% Setup
import time
import matplotlib.pyplot as plt
import scipy.signal

from lib.resamp import apply_taper_transition, fft_resample, fft_resample_transition
from lib.tabular import Tabular
from lib.util import *
from lib.taper import get_taper_transition

plot = False
REPETITIONS = 100

# %% Parameters and table headings

conversions = [
    (44100, 48000, 50),
]

L_prop = 0.05 # of smaller Nyquist

tapers = {
    'Cosine': 'cosine',
    'FIR (OLA FFT)': 'fir_ola',
    'FIR (giant FFT)': 'fir_fft',
    'Dolph--Chebyshev': ('chebwin', 95.8),
    'DDC optimal': ('ddc', 150),
}

fir_length_s = 0.00652
fir_length_samp = int(44100 * fir_length_s)
if fir_length_samp % 2 == 0: fir_length_samp += 1
print("FIR points =", fir_length_samp)

columns = {
    "name": r"Method",
    "design_time": r"Design (\%)",
    "exec_time": r"Execution (\%)",
}

def highlight_cell(name, key):
    if "OLA" in name and key == "design_time": return True
    if "FIR" not in name and key == "exec_time": return True
    return False

# %% Run benchmark and produce Table II

def bench(fn, setup=None, repetitions=REPETITIONS):
    # Warm up once to ensure imports etc are resolved
    if setup: setup()
    fn()

    runs = 0
    measured_time = 0.
    for rep in range(repetitions):
        if setup: setup()
        start = time.perf_counter()
        fn()
        measured_time += time.perf_counter() - start
        runs += 1
    result = measured_time / runs
    print(runs, "runs done, average", 1000*result, "ms")
    return result

def fmt_time(time_s):
    time_ms = time_s * 1000
    if time_ms > 10: return f"{time_ms:.1f}"
    else: return f"{time_ms:.2f}"

def fmt_design_time(table_row, time_s, baseline_s):
    percent = time_s / baseline_s * 100
    num = f"{percent:.1f}"
    print(f"- Design:\t {num} %")
    table_row["design_time"] = f"${num}$"

def fmt_exec_time(table_row, time_s, baseline_s):
    percent = time_s / baseline_s * 100
    num = f"{percent:.1f}"
    print(f"- Execution:\t {num} %")
    table_row["exec_time"] = f"${num}$"

for Fs_in, Fs_out, duration in conversions:
    N_in = int(Fs_in * duration)
    N_out = int(Fs_out * duration)
    M = min(N_in, N_out)
    L = int(L_prop * M / 2)

    print("\n\n")
    print(("Upsample" if N_out > N_in else "Downsample"), Fs_in, "to", Fs_out, "duration", duration, "s, M =", M, ", L =", L)
    print("==========")

    table = Tabular(columns, highlight_cell)

    sig = np.random.rand(N_in)

    # Measure FFT SRC time without tapering
    def measure0():
        global result
        result = fft_resample(sig, None, N_out)

    baseline_s = bench(measure0)
    print(f"FFT SRC without taper:\t{baseline_s*1000} ms")

    for name, spec in tapers.items():
        print("\nMethod:", name)

        table_row = {"name": name}
        table.append(table_row)

        if spec == 'fir_ola':
            # Design
            def measure3():
                global kernel
                kernel = scipy.signal.firwin(fir_length_samp, 21000, window=('chebwin', 135), fs=Fs_in)

            fmt_design_time(table_row, bench(measure3), baseline_s)

            # Execute OLA convolution & plain FFT SRC
            def measure4():
                sig_filt = scipy.signal.oaconvolve(sig, kernel, mode='same')

            fmt_exec_time(table_row, bench(measure4), baseline_s)

        elif spec == 'fir_fft':
            # Design frequency-domain FIR (FFT of zero padded kernel)
            def measure5():
                global freq_fir
                measure3()
                kernel_padded = np.zeros(M)
                kernel_padded[:(len(kernel) + 1)//2] = kernel[len(kernel)//2:]
                kernel_padded[-(len(kernel)//2):] = kernel[:len(kernel)//2]
                freq_fir = np.fft.fft(kernel_padded)


            fmt_design_time(table_row, bench(measure5), baseline_s)

            if plot:
                plt.plot(np.fft.fftfreq(M, 1/Fs_in)[:M//2], amp2db(freq_fir[:M//2]))

            # Perform frequency-domain multiplication
            def setup6():
                global buf
                buf = np.fft.fft(sig)

            def measure6():
                global buf
                buf *= freq_fir

            fmt_exec_time(table_row, bench(measure6, setup6), baseline_s)

        else:
            # Design
            def measure1():
                global transition
                transition, _ = get_taper_transition(spec, M, L)

            fmt_design_time(table_row, bench(measure1), baseline_s)

            # Execute
            def setup2():
                global buf
                buf = np.fft.fft(sig)

            def measure2():
                apply_taper_transition(buf, transition)

            fmt_exec_time(table_row, bench(measure2, setup2), baseline_s)


    print()
    print(table.fmt_table())
