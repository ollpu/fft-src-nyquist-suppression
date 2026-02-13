"""
Example conversion with and without tapering.

Used to generate Fig. 1 and the sound examples on the website.

Set `play = True` below to hear the original signal and its conversions.
"""

# %% Setup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

import soundfile as sf

plt.style.use('plots.mplstyle')

plt.rcParams['figure.dpi'] = 300

# WARNING: Loud!
play = False
write = False

if play:
    import sounddevice as sd

np.random.seed(4)

Fs_in = 8000
Fs_out = 22050

input_len = 400_000
output_len = int(Fs_out / Fs_in * input_len)
L = int(0.10 * input_len / 2)
print(output_len, L)


# %% Input signal

input = 2*np.random.rand(input_len) - 1
time = np.arange(input_len) / Fs_in

envelope = np.exp(-40 * (time % 1))

input *= envelope

input = input / np.max(np.abs(input))

clip = input[:10*Fs_in]

if play:
    print("Playing input signal. Resampled on the fly by Portaudio or your system!")
    sd.play(clip, Fs_in)
    sd.wait()

if write:
    sf.write("website/example_in.wav", clip, Fs_in)

# %% FFT resampling without tapering

output_naive = fft_resample(input, np.ones(input_len), output_len)

plt.plot(amp2db(output_naive[:5*Fs_out] / np.max(np.abs(output_naive))))
plt.show()

clip = output_naive[:10*Fs_out]

if play:
    print("Playing resampled signal without tapering. Notice the ringing.")
    sd.play(clip, Fs_out)
    sd.wait()

if write:
    sf.write("website/example_naive.wav", clip, Fs_out)


# %% FFT resampling with tapering

taper, _ = get_taper(('ddc', 150), input_len, L)
output_tapered = fft_resample(input, taper, output_len)

plt.plot(amp2db(output_tapered[:5*Fs_out] / np.max(np.abs(output_tapered))))
plt.axhline(-120)
plt.show()

clip = output_tapered[:10*Fs_out]

if play:
    print("Playing resampled signal with tapering.")
    sd.play(clip, Fs_out)
    sd.wait()

if write:
    sf.write("website/example_tapered.wav", clip, Fs_out)

# %% Produce Fig. 1

time_in = np.arange(input_len) / Fs_in
time_out = np.arange(output_len) / Fs_out

start = 1.0
stop = 3.0
pad = 0.05
mask_in = (time_in >= start - pad) & (time_in < stop + pad)
mask_out = (time_out >= start - pad) & (time_out < stop + pad)

fig = plt.figure(figsize=(3.5, 2))

gs = GridSpec(2, 4, height_ratios=[2, 5], width_ratios=[1, 1, 1, 0.1], left=0.1, bottom=0.21, right=0.91, top=0.97, hspace=0.70)

ax1 = plt.subplot(gs[0, :3])

input_half = input[:Fs_in*5]
decimate = 20
chunks = input_half.reshape((len(input_half)//decimate, decimate))
low = np.min(chunks, axis=-1)
hi = np.max(chunks, axis=-1)

plt.fill_between(time_in[:len(input_half):decimate], low, hi, linewidth=1, ec='face')
plt.xlim(0, len(input_half) / Fs_in)
plt.ylim(-1, 1)
plt.axvspan(start, stop, -0.1, 1.1, color=('black', 0.16), linewidth=2, ec='none', ls='-')
ax1.set_ylabel("$x(n)$", labelpad=2)
ax1.set_xlabel("Time (s)", labelpad=-2)
ax1.set_title('(a)', y=0, pad=-24)
ax1.tick_params(pad=2)

def spectrogram(sig, **params):
    return librosa.amplitude_to_db(np.abs(librosa.stft(sig, window=('chebwin', 160), center=False, **params)), ref=np.max, amin=1e-10, top_db=150)

ax2 = plt.subplot(gs[1, 0])
stft = spectrogram(input[mask_in], n_fft=512)
img_t = librosa.frames_to_time(np.arange(stft.shape[1]), sr=Fs_in, hop_length=512//4) + start
img = librosa.display.specshow(stft, sr=Fs_in, x_coords=img_t, x_axis='time', y_axis='linear', cmap='viridis')

ax2.set_xlabel("Time (s)", labelpad=1)
ax2.set_ylabel("Frequency (kHz)", labelpad=2)
ax2.locator_params('y', min_n_ticks=6)
ax2.set_title('(b)', y=0, pad=-28)
ax2.locator_params(axis='x', nbins=3)
xlim = (start-0.02, stop+0.02)
ax2.set_xlim(xlim)
ax2.set_ylim(0, Fs_out/2)

# Cross out area beyond Nyquist
ax2.axhspan(Fs_in/2, Fs_out/2, color='lightgray', lw=0)
ax2.plot([start, stop], [Fs_in/2, Fs_out/2], c='red', lw=0.5, rasterized=True)
ax2.plot([stop, start], [Fs_in/2, Fs_out/2], c='red', lw=0.5, rasterized=True)

ax2.axhspan(Fs_in/2, Fs_out/2, color='lightgray', lw=0)
ax2.plot([start, stop], [Fs_in/2, Fs_out/2], c='red', lw=0.5)
ax2.plot([stop, start], [Fs_in/2, Fs_out/2], c='red', lw=0.5)


ax3 = plt.subplot(gs[1, 1], sharex=ax2, sharey=ax2)
stft = spectrogram(output_naive[mask_out], n_fft=1024)
img_t = librosa.frames_to_time(np.arange(stft.shape[1]), sr=Fs_out, hop_length=1024//4) + start
img = librosa.display.specshow(stft, sr=Fs_out, x_coords=img_t, x_axis='time', y_axis='linear', cmap='viridis')
plt.setp(ax3.get_yticklabels(), visible=False)
ax3.set_xlabel("Time (s)", labelpad=1)
ax3.set_ylabel("")
ax3.set_title('(c)', y=0, pad=-28)
ax3.set_xlim(start, stop)

ax3.annotate("", xy=(1.5, Fs_in/2), xytext=(1.2, 7000), arrowprops=dict(fc='white', ec='none', arrowstyle="simple", shrinkB=1))

ax4 = plt.subplot(gs[1, 2], sharex=ax2, sharey=ax2)
stft = spectrogram(output_tapered[mask_out], n_fft=1024)
img_t = librosa.frames_to_time(np.arange(stft.shape[1]), sr=Fs_out, hop_length=1024//4) + start
img = librosa.display.specshow(stft, sr=Fs_out, x_coords=img_t, x_axis='time', y_axis='linear', cmap='viridis')
plt.setp(ax4.get_yticklabels(), visible=False)
ax4.locator_params(axis='x', nbins=3)
ax4.set_xlabel("Time (s)", labelpad=1)
ax4.set_ylabel("")
ax4.set_title('(d)', y=0, pad=-28)

ax4.annotate("", xy=(1.5, Fs_in/2), xytext=(1.2, 7000), arrowprops=dict(fc='white', ec='none', arrowstyle="simple", shrinkB=1))


ax2.yaxis.set_major_formatter(lambda x, p: str(int(x / 1000)))

ax5 = plt.subplot(gs[1, 3])
fig.colorbar(img, cax=ax5)
ax5.yaxis.set_major_formatter(lambda x, p: (str(int(x)) if x != 0 else "0 dB"))


plt.show()


fig.savefig("figures/fig1.pdf")
