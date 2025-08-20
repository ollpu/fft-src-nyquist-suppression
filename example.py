# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import sounddevice as sd

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['pdf.fonttype'] = 42

# WARNING: Loud!
play = False

np.random.seed(4)

Fs_in = 10000
Fs_out = 48000

input_len = 500_000
output_len = int(Fs_out / Fs_in * input_len)
L = int(0.05 * input_len / 2)
print(output_len, L)

def norm(x):
    return x / np.max(np.abs(x))

# %% Input signal

input = 2*np.random.rand(input_len) - 1
time = np.arange(input_len) / Fs_in

envelope = np.exp(-40 * (time % 1))

input *= envelope

input = norm(input)

if play:
    sd.play(input[:4*Fs_in], Fs_in)
    sd.wait()

# %% FFT resampling without tapering

output_naive = norm(fft_resample(input, np.ones(input_len), output_len))

plt.plot(amp2db(output_naive[:5*Fs_out]))
plt.show()

if play:
    sd.play(output_naive[:4*Fs_out], Fs_out)
    sd.wait()


# %% FFT resampling with tapering

taper, _ = get_taper(('ddc', 150), input_len, L)
# taper, _ = get_taper(('cosine'), input_len, L)
output_tapered = norm(fft_resample(input, taper, output_len))

plt.plot(amp2db(output_tapered[:5*Fs_out]))
plt.show()

if play:
    sd.play(output_tapered[:4*Fs_out], Fs_out)
    sd.wait()

# %% Plot

# %matplotlib osx
# plt.close()

time_in = np.arange(input_len) / Fs_in
time_out = np.arange(output_len) / Fs_out

start = 2.0
stop = 6.0
mask_in = (time_in >= start) & (time_in < stop)
mask_out = (time_out >= start) & (time_out < stop)

fig = plt.figure(figsize=(5, 5.625))
fig.set_linewidth(1)

gs = GridSpec(4, 1, height_ratios=[1, 2, 2, 2], left=0.1, bottom=0.12, right=0.97, top=0.98, hspace=0.4)

ax1 = plt.subplot(gs[0])

decimate = 100
chunks = input.reshape((input_len//decimate, decimate))
low = np.min(chunks, axis=-1)
hi = np.max(chunks, axis=-1)

plt.fill_between(time_in[::decimate], low, hi, linewidth=1, ec='face')
plt.xlim(0, input_len / Fs_in)
plt.ylim(-1, 1)
plt.axvspan(start, stop, -0.1, 1.1, color=(0, 0, 0, 0.12), linewidth=1, ec='black', ls='--')
ax1.set_title('(a)', y=0, pad=-17)
ax1.tick_params(pad=2)

ax2 = plt.subplot(gs[1])
plt.plot(time_in[mask_in], input[mask_in])
plt.xlim(start, stop)
plt.ylim(-0.01, 0.01)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_title('(b)', y=0, pad=-17)

ax3 = plt.subplot(gs[2], sharex=ax2, sharey=ax2)
plt.plot(time_out[mask_out], output_naive[mask_out])
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_title('(c)', y=0, pad=-17)

ax4 = plt.subplot(gs[3], sharex=ax2, sharey=ax2)
plt.plot(time_out[mask_out], output_tapered[mask_out])
ax4.set_xlabel("Time (s)", labelpad=2)
ax4.set_title('(d)', y=0, pad=-42)


plt.show()

fig.savefig("example.pdf")
