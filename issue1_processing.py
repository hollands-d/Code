"""Legacy Issue 1 vibration-processing algorithm.

This module intentionally operates on Fourier *amplitudes*.  It must remain
separate from the Issue 2 Welch-PSD and Gordon-band integration path in the
GUI: the two methods answer different historical/reference requirements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


G0 = 9.80665
ISSUE1_SAMPLE_RATE_HZ = 200.0
ISSUE1_FFT_N = 128
ISSUE1_OVERLAP = 0.5
ISSUE1_ENSEMBLE_COUNT = 19
ISSUE1_LOW_HZ = 3.5
ISSUE1_HIGH_HZ = 90.0
ISSUE1_FIR_TAPS = 101
ISSUE1_HANN_RANDOM_SCALE = 1.63

# Nominal one-third-octave centres and Gordon Office RMS velocities reproduced
# from Issue 1 Table 2. Acceleration, PSD and Fourier limits are derived from
# these source quantities below rather than copying the plotted FT ordinates.
ISSUE1_BAND_CENTRES_HZ = np.array([
    4.0, 5.0, 6.3, 7.9, 10.0, 12.6, 15.8,
    20.0, 25.1, 31.6, 39.8, 50.1, 63.1, 79.4,
])
ISSUE1_GORDON_VELOCITY_UM_S = np.array([
    802.6, 639.7, 509.9, 406.4, 406.4, 406.4, 406.4,
    406.4, 406.4, 406.4, 406.4, 406.4, 406.4, 406.4,
])
ONE_THIRD_OCTAVE_EDGE = 2.0 ** (1.0 / 6.0)


@dataclass(frozen=True)
class Issue1Result:
    """Intermediate and final values from one three-axis Issue 1 ensemble."""

    filtered_xyz_g: np.ndarray
    window_starts: np.ndarray
    frequency_hz: np.ndarray
    individual_amplitude_g: np.ndarray
    ensemble_amplitude_g: np.ndarray
    threshold_amplitude_g: np.ndarray
    exceeded: np.ndarray
    worst_axis: str
    worst_frequency_hz: float
    worst_measured_g: float
    worst_threshold_g: float
    worst_ratio: float


def issue1_hop_samples(fft_n: int = ISSUE1_FFT_N,
                       overlap: float = ISSUE1_OVERLAP) -> int:
    """Return the Issue 1 FFT start-to-start spacing in samples."""
    if fft_n <= 0 or not 0 <= overlap < 1:
        raise ValueError("FFT length must be positive and overlap must be in [0, 1).")
    hop = int(round(fft_n * (1.0 - overlap)))
    if hop < 1:
        raise ValueError("Overlap leaves no forward FFT hop.")
    return hop


def issue1_required_samples(fft_n: int = ISSUE1_FFT_N,
                            overlap: float = ISSUE1_OVERLAP,
                            ensemble_count: int = ISSUE1_ENSEMBLE_COUNT) -> int:
    """Return samples spanning the requested overlapping FFT ensemble."""
    if ensemble_count < 1:
        raise ValueError("Ensemble count must be at least one.")
    return fft_n + (ensemble_count - 1) * issue1_hop_samples(fft_n, overlap)


def design_issue1_fir(sample_rate_hz: float = ISSUE1_SAMPLE_RATE_HZ,
                      low_hz: float = ISSUE1_LOW_HZ,
                      high_hz: float = ISSUE1_HIGH_HZ,
                      num_taps: int = ISSUE1_FIR_TAPS) -> np.ndarray:
    """Design the Issue 1 Hamming-windowed 3.5--90 Hz FIR band-pass.

    The ideal band-pass impulse response is the difference of two sinc
    low-pass responses.  A Hamming window makes the finite truncation usable,
    and the coefficients are normalised to unity gain near the passband centre.
    """
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be finite and greater than zero.")
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("FIR cut-offs must satisfy 0 < low < high < Nyquist.")
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("FIR tap count must be an odd integer of at least three.")

    centred = np.arange(num_taps, dtype=float) - (num_taps - 1) / 2.0
    high_lp = 2.0 * high_hz / fs * np.sinc(2.0 * high_hz / fs * centred)
    low_lp = 2.0 * low_hz / fs * np.sinc(2.0 * low_hz / fs * centred)
    taps = (high_lp - low_lp) * np.hamming(num_taps)

    reference_hz = (low_hz + high_hz) / 2.0
    response = np.sum(taps * np.exp(-2j * np.pi * reference_hz
                                    * np.arange(num_taps) / fs))
    taps /= abs(response)
    return taps


def apply_issue1_fir(xyz_g: np.ndarray, taps: np.ndarray | None = None) -> np.ndarray:
    """Apply the vibration FIR independently to X, Y, and Z."""
    data = np.asarray(xyz_g, dtype=float)
    one_dimensional = data.ndim == 1
    if one_dimensional:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("Vibration data must be a sample-by-axis array.")
    coefficients = design_issue1_fir() if taps is None else np.asarray(taps, dtype=float)
    filtered = np.column_stack([
        np.convolve(data[:, axis], coefficients, mode="same")
        for axis in range(data.shape[1])
    ])
    return filtered[:, 0] if one_dimensional else filtered


def issue1_fourier_threshold(frequency_hz: np.ndarray,
                             resolution_hz: float) -> np.ndarray:
    """Derive the Issue 1 peak Fourier-amplitude threshold from Gordon Office.

    For each Issue 1 Table 2 band, Gordon Office RMS velocity is converted to
    RMS acceleration. Assuming constant PSD across that band's exact
    one-third-octave bandwidth gives::

        PSD = acceleration_rms**2 / one_third_octave_bandwidth

    A one-sided Fourier bin of width ``resolution_hz`` then has the equivalent
    peak sinusoidal amplitude ``sqrt(2 * PSD * resolution_hz)``.  Values outside
    the Issue 1 software passband are infinite so they cannot be exceedances.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(resolution_hz) or resolution_hz <= 0:
        raise ValueError("FFT frequency resolution must be greater than zero.")
    threshold = np.full(frequency.shape, np.inf, dtype=float)
    acceleration_rms_g = (
        2.0 * np.pi * ISSUE1_BAND_CENTRES_HZ
        * ISSUE1_GORDON_VELOCITY_UM_S * 1e-6 / G0
    )
    octave_bandwidth_hz = ISSUE1_BAND_CENTRES_HZ * (
        ONE_THIRD_OCTAVE_EDGE - 1.0 / ONE_THIRD_OCTAVE_EDGE
    )
    psd_g2_hz = acceleration_rms_g ** 2 / octave_bandwidth_hz
    band_fourier_g = np.sqrt(2.0 * psd_g2_hz * resolution_hz)

    # Issue 1 compares each retained FFT coordinate with the constant Fourier
    # limit for the one-third-octave band containing that coordinate. With a
    # 128-point FFT the lowest band contains no frequency bin, as the document
    # explicitly notes.
    for centre, limit in zip(ISSUE1_BAND_CENTRES_HZ, band_fourier_g):
        lower = centre / ONE_THIRD_OCTAVE_EDGE
        upper = centre * ONE_THIRD_OCTAVE_EDGE
        in_band = ((frequency >= lower) & (frequency < upper)
                   & np.isfinite(frequency))
        threshold[in_band] = limit
    return threshold


def issue1_amplitude_from_psd(psd_frequency_hz: np.ndarray,
                              psd_g2_per_hz: np.ndarray,
                              sample_rate_hz: float = ISSUE1_SAMPLE_RATE_HZ,
                              fft_n: int = ISSUE1_FFT_N,
                              apply_fir_response: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Convert a measured one-sided acceleration PSD to Issue-1 bin amplitudes.

    No synthetic time history is created.  For each Issue-1 rFFT coordinate,
    the measured PSD is integrated over that FFT bin and converted to the
    equivalent peak Fourier amplitude using ``sqrt(2 * mean_square)``.  This
    is the same PSD-to-Fourier convention used by
    :func:`issue1_fourier_threshold`.

    When ``apply_fir_response`` is true, the frequency-domain magnitude-squared
    response of the Issue-1 3.5--90 Hz Hamming FIR is applied before bin
    integration.  That makes a PSD reference equivalent to a time-domain
    reference passed through the normal Issue-1 preprocessing path.
    """
    f=np.asarray(psd_frequency_hz,dtype=float)
    p=np.asarray(psd_g2_per_hz,dtype=float)
    good=np.isfinite(f)&np.isfinite(p)&(f>=0)&(p>=0)
    f=f[good]; p=p[good]
    if f.size<2:
        raise ValueError("Issue 1 PSD conversion requires at least two valid frequency/PSD rows.")
    order=np.argsort(f); f=f[order]; p=p[order]
    uf,inv=np.unique(f,return_inverse=True)
    if uf.size!=f.size:
        sums=np.zeros(uf.size,float); counts=np.zeros(uf.size,float)
        np.add.at(sums,inv,p); np.add.at(counts,inv,1.0)
        f=uf; p=sums/counts

    fs=float(sample_rate_hz)
    if fft_n<=0 or fs<=0:
        raise ValueError("Issue 1 FFT length and sample rate must be positive.")
    coords=np.fft.rfftfreq(int(fft_n),1.0/fs)
    df=fs/float(fft_n)

    if apply_fir_response:
        taps=design_issue1_fir(fs)
        omega=2.0*np.pi*f/fs
        n=np.arange(len(taps),dtype=float)
        h=np.exp(-1j*np.outer(omega,n))@taps
        p=p*np.abs(h)**2

    amp=np.full(coords.shape,np.nan,dtype=float)
    for i,fc in enumerate(coords):
        lo=max(0.0,fc-df/2.0)
        hi=min(fs/2.0,fc+df/2.0)
        if hi<=lo or lo<f[0] or hi>f[-1]:
            continue
        inside=(f>lo)&(f<hi)
        fb=np.r_[lo,f[inside],hi]
        pb=np.r_[np.interp(lo,f,p),p[inside],np.interp(hi,f,p)]
        mean_square=float(np.trapezoid(pb,fb))
        amp[i]=np.sqrt(max(0.0,2.0*mean_square))
    return coords,amp


def issue1_ensemble_fft(xyz_g: np.ndarray,
                        sample_rate_hz: float = ISSUE1_SAMPLE_RATE_HZ,
                        fft_n: int = ISSUE1_FFT_N,
                        overlap: float = ISSUE1_OVERLAP,
                        ensemble_count: int = ISSUE1_ENSEMBLE_COUNT,
                        start: int = 0,
                        fir_taps: np.ndarray | None = None) -> Issue1Result:
    """Run the legacy Issue 1 amplitude-ensemble algorithm on three axes."""
    data = np.asarray(xyz_g, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Issue 1 processing requires an N-by-3 X/Y/Z array.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Issue 1 input contains missing or non-finite samples.")
    fs = float(sample_rate_hz)
    hop = issue1_hop_samples(fft_n, overlap)
    required = issue1_required_samples(fft_n, overlap, ensemble_count)
    if start < 0 or start + required > len(data):
        raise ValueError(
            f"Issue 1 ensemble needs {required} samples from start {start}; "
            f"only {len(data)} are available."
        )

    taps = design_issue1_fir(fs) if fir_taps is None else np.asarray(fir_taps, dtype=float)
    filtered = apply_issue1_fir(data, taps)
    starts = start + np.arange(ensemble_count, dtype=int) * hop
    window = np.hanning(fft_n)
    amplitudes = []
    for offset in starts:
        spectra = np.fft.rfft(filtered[offset:offset + fft_n] * window[:, None], axis=0)
        # Issue 1 equation (13): combine the mirrored-frequency factor (2),
        # the document's Hann correction for random signals (1.63), and 1/N
        # normalisation to obtain peak Fourier amplitudes.
        amplitude = (2.0 * ISSUE1_HANN_RANDOM_SCALE / fft_n) * np.abs(spectra)
        amplitude[0] *= 0.5
        if fft_n % 2 == 0:
            amplitude[-1] *= 0.5
        amplitudes.append(amplitude.T)

    individual = np.stack(amplitudes, axis=0)
    ensemble = np.mean(individual, axis=0)
    frequency = np.fft.rfftfreq(fft_n, 1.0 / fs)
    threshold = issue1_fourier_threshold(frequency, fs / fft_n)
    exceeded = ensemble > threshold[None, :]
    ratio = np.divide(
        ensemble,
        threshold[None, :],
        out=np.zeros_like(ensemble),
        where=np.isfinite(threshold[None, :]) & (threshold[None, :] > 0),
    )
    worst_flat = int(np.argmax(ratio))
    worst_axis_i, worst_bin = np.unravel_index(worst_flat, ratio.shape)
    return Issue1Result(
        filtered_xyz_g=filtered,
        window_starts=starts,
        frequency_hz=frequency,
        individual_amplitude_g=individual,
        ensemble_amplitude_g=ensemble,
        threshold_amplitude_g=threshold,
        exceeded=exceeded,
        worst_axis="XYZ"[worst_axis_i],
        worst_frequency_hz=float(frequency[worst_bin]),
        worst_measured_g=float(ensemble[worst_axis_i, worst_bin]),
        worst_threshold_g=float(threshold[worst_bin]),
        worst_ratio=float(ratio[worst_axis_i, worst_bin]),
    )
