# %%
import matplotlib.pyplot as plt

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.style.use('plots.mplstyle')

N = 44100
M = 192000
time_s = 1
L = 1000

tapers = {
    'Box': 'box',
    'Cosine': 'cosine',
    'Hann': 'hann',
    'Blackman': 'blackman',
    'Dolph–Chebyshev': ('chebwin', 71),
    'DDC $\\alpha=1/2$': ('ddc', 120, 0.5),
    'DDC optimal': ('ddc', 120),
}


# %%
%matplotlib inline

input = np.zeros(N)
input[N//2] = 1
time = (np.arange(M) - M//2) / M * time_s * 1000
mask = np.abs(time) < 22

box_taper, _ = get_taper('box', N, L)
box_output = fft_resample(input, box_taper, M)
box_output /= np.max(np.abs(box_output))

fig = plt.figure(figsize=(3.5, 4))
fig.subplots_adjust(left=0.14, bottom=0.13, right=0.98, top=0.995, wspace=0.1, hspace=0.3)
axs = fig.subplots(len(tapers)//2, 2, sharex=True, sharey=True)

for pos, (name, spec) in enumerate(tapers.items()):
    if spec == 'box': continue

    row = (pos-1)//2
    col = (pos-1)%2
    last_row = row == len(tapers)//2 - 1

    taper, _ = get_taper(spec, N, L)

    output = fft_resample(input, taper, M)
    output /= np.max(np.abs(output))

    ax = axs[row][col]
    ax.plot(time[mask], amp2db(box_output[mask]), label='Box', linewidth=0.7, color='#ccc')
    # ax.fill_between(time[mask], -300, amp2db(box_output[mask]), linewidth=1, color='#ccc', ec='#ccc')
    ax.plot(time[mask], amp2db(output[mask]), label=name, linewidth=0.7, c='tab:blue')

    # ax.axhline(-100)
    # ax.axhline(0)

    # ax.legend(loc='upper right')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-165, 5)
    ax.locator_params(min_n_ticks=4, steps=[1, 2, 5, 10], axis='x')
    ax.locator_params(min_n_ticks=4, steps=[1, 2, 4, 10], axis='y')

    pad = -13
    if last_row:
        pad = -34
        ax.set_xlabel("Time (ms)", labelpad=3)
    ax.set_title(f'({chr(ord('a')+pos-1)}) {name}', y=0, pad=pad)

fig.supylabel("Magnitude (dB)", x=0.01)

plt.show()

fig.savefig("irs.pdf")
