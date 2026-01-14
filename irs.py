"""
Fig. 3: Time-domain impulse responses of tapering functions.
"""

# %% Setup
import matplotlib.pyplot as plt

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.style.use('plots.mplstyle')

N = 44100
M = 192000
time_s = 1
L = 1102

tapers = {
    'Box': 'box',
    'Cosine': 'cosine',
    'Dolph–\nCheby.': ('chebwin', 97),
    'DDC\t$\\alpha=1/2$': ('ddc', 150, 0.5),
    'DDC\t$\\alpha$ optimal': ('ddc', 150),
}


# %% Produce Fig. 3

input = np.zeros(N)
input[N//2] = 1
time = (np.arange(M) - M//2) / M * time_s * 1000
mask = np.abs(time) < 13

box_taper, _ = get_taper('box', N, L)
box_output = fft_resample(input, box_taper, M)
box_output /= np.max(np.abs(box_output))

fig = plt.figure(figsize=(3.5, 2.5))
fig.subplots_adjust(left=0.14, bottom=0.20, right=0.98, top=0.995, wspace=0.1, hspace=0.3)
axs = fig.subplots(len(tapers)//2, 2, sharex=True, sharey=True)

fax = axs[0, 0]
fax.set_xlim(-12.5, 12.5)
fax.set_ylim(-207, 7)
fax.locator_params(min_n_ticks=4, steps=[1, 2, 4, 10], axis='x')
fax.locator_params(min_n_ticks=4, steps=[1, 2, 4, 5, 10], axis='y')

for pos, (name, spec) in enumerate(tapers.items()):
    if spec == 'box': continue

    row = (pos-1)//2
    col = (pos-1)%2
    last_row = row == len(tapers)//2 - 1

    taper, _ = get_taper(spec, N, L)

    output = fft_resample(input, taper, M)
    output /= np.max(np.abs(output))

    ax = axs[row][col]
    ax.plot(time[mask], amp2db(box_output[mask]), label='Box', linewidth=0.7, color='#bbb')
    ax.plot(time[mask], amp2db(output[mask]), label=name, linewidth=0.7, c='tab:blue')

    ax.grid(True, lw=0.5, c='#ddd')

    pad = -11
    if last_row:
        pad = -32
        ax.set_xlabel("Time (ms)", labelpad=3)
    ax.set_title(f'({chr(ord('a')+pos-1)})', y=0, pad=pad)
    parts = name.split("\t")
    y = 0.964 if name.startswith("Dolph") else 0.93
    ax.text(0.05, y, parts[0], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    if len(parts) > 1:
        ax.text(0.95, 0.85, parts[1], horizontalalignment='right', verticalalignment='baseline', transform=ax.transAxes)


fig.supylabel("Magnitude (dB)", x=0.01, y=0.6)

plt.show()

fig.savefig("figures/fig3.pdf")
