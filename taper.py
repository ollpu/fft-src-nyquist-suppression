"""
Fig. 2: Frequency-domain taper functions.
"""

# %% Setup
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.style.use('plots.mplstyle')

M = 400_000
time_s = 50
L = 20_000

# %% Produce Fig. 2

fig = plt.figure(figsize=(3.5, 2))

gs = GridSpec(2, 1, height_ratios=[2, 5], left=0.14, bottom=0.21, right=0.97, top=0.97, hspace=0.45, wspace=0.26)

freq = (np.arange(M + 1) - M//2) / time_s / 1000

lin = plt.subplot(gs[0])
db_coarse = plt.subplot(gs[1])

lin.set_xmargin(0.01)
lin.set_ymargin(0.10)
lin.locator_params(min_n_ticks=6, axis='x')
db_coarse.set_xlim(3.590, 4.010)
db_coarse.set_ylim(-110, 10)
lin.grid(True, lw=0.5, c='#ddd')
db_coarse.grid(True, lw=0.5, c='#ddd')

lin.axvspan(3.590, 4.010, color=('black', 0.16), lw=0, zorder=2)

lin.set_ylabel('$W(k)$')
db_coarse.set_ylabel('Magnitude (dB)')

fig.supxlabel('Frequency (kHz)', x=0.56, y=0.06)

taper, _ = get_taper('cosine', M, L)
taper = np.append(np.fft.fftshift(taper), [0])
col = 'black'
lin.plot(freq, taper, ':', c=col)
db_coarse.plot(freq, amp2db(taper), ':', c=col, label='Cosine')

taper, _ = get_taper(('chebwin', 96), M, L)
taper = np.append(np.fft.fftshift(taper), [0])
lin.plot(freq, taper)
db_coarse.plot(freq, amp2db(taper), ls=(1, (3.7, 1.6)), label='Dolph–Chebyshev ($R=96\\,$dB)', lw=2, c='tab:orange')

taper, _ = get_taper(('ddc', 150), M, L)
taper = np.append(np.fft.fftshift(taper), [0])
lin.plot(freq, taper, '--')
db_coarse.plot(freq, amp2db(taper), label='DDC optimal ($R=150\\,$dB)', c='tab:blue')

db_coarse.legend()

plt.show()

fig.savefig("figures/fig2.pdf")
