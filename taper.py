# %%
import matplotlib.pyplot as plt

from lib.util import *
from lib.taper import get_taper
from lib.resamp import fft_resample

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['lines.linewidth']=1

M = 10000
time_s = 1
L = 1000

# %%

%matplotlib inline

fig, axs = plt.subplots(1, 1, figsize=(5, 2.75), sharex=True, sharey=True)
fig.subplots_adjust(left=0.04, bottom=0.24, right=0.98, top=0.97, wspace=0.1, hspace=0.45)
freq = (np.arange(M) - M//2) / time_s

taper, _ = get_taper('cosine', M, L)
taper = np.fft.fftshift(taper)
axs.plot(freq, amp2db(taper))
axs.set_title("(a) Cosine", y=0, pad=-17)
axs.set_xmargin(0.02)
axs.set_ymargin(0.05)
axs.set_ylim(-120, 0)
axs.locator_params(min_n_ticks=4, steps=[1, 5, 10], axis='x')

taper, _ = get_taper(('ddc', 100), M, L)
taper = np.fft.fftshift(taper)
axs.plot(freq, amp2db(taper))
axs.set_xlabel("Frequency (Hz)", labelpad=1)
axs.set_title("(b) DDC optimal", y=0, pad=-42)

plt.show()

fig.savefig("taper.pdf")
