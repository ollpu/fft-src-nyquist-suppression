# Suppression of Nyquist Ringing in FFT-Based Sample Rate Conversion

_Roope Salmi and Vesa Välimäki_

This repository contains the source code and resources related to the Letter accepted for publication in IEEE Signal Processing Letters, to appear.

Website with sound examples: <https://ollpu.github.io/fft-src-nyquist-suppression>

## Scripts

- [example.py](example.py): Reproduces [Fig. 1](figures/fig1.pdf) and the sound examples available on the website.
- [tapers.py](tapers.py): [Fig. 2](figures/fig2.pdf)
- [irs.py](irs.py): [Fig. 3](figures/fig3.pdf)
- [comparison.py](comparison.py): Computes the data for Table I. Output saved [here](output/comparison.txt)
- [benchmark.py](benchmark.py): Runs the benchmark for Table II. Output saved [here](output/benchmark.txt)

### Running the scripts

Install the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager and run `uv sync`.

Then, do `uv run python [script]` or open the scripts in your IDE of choice.

## Library code

Under [lib/](lib/), there are utilities for [performing](lib/resamp.py) FFT-based sample rate conversion
and [constructing](lib/taper.py) various taper functions. This code is available for use under the terms of the MIT license.
