# %%
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.style.use('plots.mplstyle')

M = 500_000
time_s = 50
L = 25_000

# %%

%matplotlib inline

fig = plt.figure(figsize=(3.5, 2))

gs = GridSpec(2, 1, height_ratios=[2, 5], left=0.14, bottom=0.24, right=0.97, top=0.97, hspace=0.45, wspace=0.26)

freq = (np.arange(M + 1) - M//2) / time_s / 1000

# axs[0].set_title("(a) Cosine", y=0, pad=-17)
# axs[0].set_xmargin(0.02)
# axs[0].set_ymargin(0.05)
# axs[0].set_ylim(0, 1)

# axs[1].set_xlabel("Frequency (Hz)", labelpad=1)
# axs[1].set_title("(b) DDC optimal", y=0, pad=-42)

lin = plt.subplot(gs[0])
db_coarse = plt.subplot(gs[1])
# db_fine = plt.subplot(gs[1, 1])

lin.set_xmargin(0.01)
lin.set_ymargin(0.10)
lin.locator_params(min_n_ticks=6, axis='x')
db_coarse.set_xlim(4.485, 5.015)
db_coarse.set_ylim(-110, 10)
lin.grid(True, lw=0.5, c='#ddd')
db_coarse.grid(True, lw=0.5, c='#ddd')
# db_fine.set_xlim(3.990, 4.210)
# db_fine.set_ylim(-1.1, 0.1)

lin.axvspan(4.485, 5.015, color=('black', 0.16), lw=0, zorder=2)

lin.set_ylabel('Lin.')
db_coarse.set_ylabel('Magnitude (dB)')

fig.supxlabel('Frequency (kHz)', x=0.56, y=0.06)

taper, _ = get_taper('cosine', M, L)
taper = np.append(np.fft.fftshift(taper), [0])
col = 'black'
lin.plot(freq, taper, ':', c=col)
db_coarse.plot(freq, amp2db(taper), ':', c=col, label='Cosine')
# db_fine.plot(freq, amp2db(taper), '-')

taper, _ = get_taper(('ddc', 150), M, L)
taper = np.append(np.fft.fftshift(taper), [0])
lin.plot(freq, taper)
db_coarse.plot(freq, amp2db(taper), label='DDC optimal')
# db_fine.plot(freq, amp2db(taper), ls=(1, (1, 1.65)), c=col)

taper, _ = get_taper(('chebwin', 96), M, L)
taper = np.append(np.fft.fftshift(taper), [0])
lin.plot(freq, taper)
db_coarse.plot(freq, amp2db(taper), label='Dolph–Cheby.')
# db_fine.plot(freq, amp2db(taper), '-')


db_coarse.legend()

plt.show()

fig.savefig("taper.pdf")
