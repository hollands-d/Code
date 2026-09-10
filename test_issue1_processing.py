"""Regression tests for the deliberately separate Issue 1 processing path."""
from __future__ import annotations

from unittest import SkipTest

import numpy as np
import pandas as pd

from issue1_processing import (
    G0,
    ISSUE1_BAND_CENTRES_HZ,
    ISSUE1_ENSEMBLE_COUNT,
    ISSUE1_FFT_N,
    ISSUE1_SAMPLE_RATE_HZ,
    ISSUE1_GORDON_VELOCITY_UM_S,
    ONE_THIRD_OCTAVE_EDGE,
    design_issue1_fir,
    issue1_ensemble_fft,
    issue1_fourier_threshold,
    issue1_hop_samples,
    issue1_required_samples,
)


def _three_axis_sine(amplitude: float = 0.1, frequency_hz: float = 20.3125):
    count = issue1_required_samples()
    time_s = np.arange(count, dtype=float) / ISSUE1_SAMPLE_RATE_HZ
    return np.column_stack((
        amplitude * np.sin(2 * np.pi * frequency_hz * time_s),
        np.zeros(count),
        np.zeros(count),
    ))


def _gui_app():
    import tkinter as tk
    import vibration_signal_processing_gui as gui

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SkipTest(f"Tk display unavailable: {exc}") from exc
    root.withdraw()
    return gui, root, gui.App(root)


def test_issue1_resolution_is_1_5625_hz():
    frequency = np.fft.rfftfreq(ISSUE1_FFT_N, 1 / ISSUE1_SAMPLE_RATE_HZ)
    assert frequency[1] - frequency[0] == 1.5625


def test_issue1_half_overlap_is_64_sample_hop():
    assert issue1_hop_samples() == 64


def test_issue1_uses_19_fft_windows():
    result = issue1_ensemble_fft(_three_axis_sine())
    assert len(result.window_starts) == ISSUE1_ENSEMBLE_COUNT == 19
    assert result.individual_amplitude_g.shape == (19, 3, 65)


def test_issue1_fir_attenuates_below_and_above_passband():
    taps = design_issue1_fir()
    sample_index = np.arange(len(taps), dtype=float)

    def gain(frequency_hz):
        kernel = np.exp(-2j * np.pi * frequency_hz * sample_index / 200.0)
        return abs(np.sum(taps * kernel))

    passband_gain = gain(20.0)
    assert gain(1.0) < passband_gain * 0.05
    assert gain(95.0) < passband_gain * 0.05


def test_passband_sinusoid_has_expected_fft_bin_and_amplitude():
    result = issue1_ensemble_fft(_three_axis_sine(amplitude=0.1))
    peak_bin = int(np.argmax(result.ensemble_amplitude_g[0]))
    assert result.frequency_hz[peak_bin] == 20.3125
    # Issue 1 applies its documented 1.63 random-signal Hann correction, so a
    # coherent pure sine is intentionally not corrected back to exactly 0.1 g.
    assert abs(result.ensemble_amplitude_g[0, peak_bin] - 0.08096) < 0.002


def test_issue1_ensemble_averages_amplitude_not_psd():
    data = _three_axis_sine()
    data[:, 0] *= np.linspace(0.25, 1.75, len(data))
    result = issue1_ensemble_fft(data)
    arithmetic_amplitude = np.mean(result.individual_amplitude_g, axis=0)
    rms_amplitude = np.sqrt(np.mean(result.individual_amplitude_g ** 2, axis=0))
    assert np.allclose(result.ensemble_amplitude_g, arithmetic_amplitude)
    peak_bin = int(np.argmax(result.ensemble_amplitude_g[0]))
    assert not np.isclose(result.ensemble_amplitude_g[0, peak_bin],
                          rms_amplitude[0, peak_bin])


def test_issue1_threshold_is_derived_for_fft_resolution():
    frequency = np.fft.rfftfreq(128, 1 / 200.0)
    threshold = issue1_fourier_threshold(frequency, 1.5625)
    index = int(np.where(frequency == 12.5)[0][0])
    band = int(np.where(ISSUE1_BAND_CENTRES_HZ == 12.6)[0][0])
    acceleration_rms_g = (2 * np.pi * ISSUE1_BAND_CENTRES_HZ[band]
                          * ISSUE1_GORDON_VELOCITY_UM_S[band] * 1e-6 / G0)
    bandwidth_hz = ISSUE1_BAND_CENTRES_HZ[band] * (
        ONE_THIRD_OCTAVE_EDGE - 1 / ONE_THIRD_OCTAVE_EDGE)
    expected = np.sqrt(2 * acceleration_rms_g ** 2 / bandwidth_hz * 1.5625)
    assert np.isclose(threshold[index], expected)
    assert np.isinf(threshold[2])  # 3.125 Hz: below the specification.
    assert np.isfinite(threshold[3])  # 4.6875 Hz: first retained FFT bin.


def test_switching_back_to_issue2_restores_existing_welch_result():
    gui, root, app = _gui_app()
    try:
        rng = np.random.default_rng(1174)
        values = rng.normal(size=(4096, 3))
        frame = pd.DataFrame(values, columns=['X_g', 'Y_g', 'Z_g'])
        frame.insert(0, 'time_s', np.arange(len(frame)) / 200.0)
        proc = gui.Processor('Regression')
        proc.set(frame, gui.Cols('time_s', 'X_g', 'Y_g', 'Z_g'), 200.0)
        expected = proc.welch('X', 0, 512, 0.5, 4)[2]

        app.processing_method.set(gui.ISSUE1_METHOD)
        app._processing_method_changed()
        app.apply_accel_processing_settings()
        app.processing_method.set(gui.ISSUE2_METHOD)
        app._processing_method_changed()
        app.apply_accel_processing_settings()

        actual = proc.welch('X', 0, int(app.n.get()),
                            float(app.ov.get()) / 100, int(app.segs.get()))[2]
        assert np.array_equal(actual, expected)
    finally:
        root.destroy()


def test_mode_toggles_grey_controls_and_tabs_at_the_correct_stage():
    gui, root, app = _gui_app()
    try:
        def tab_state(label):
            tab = next(tab for tab in app.tabs.tabs()
                       if app.tabs.tab(tab, 'text') == label)
            return app.tabs.tab(tab, 'state')

        assert str(app.accel_fft_n_combo.cget('state')) == 'readonly'
        assert tab_state('Issue 1 FFT ensemble') == 'disabled'
        assert tab_state('PSD comparison') == 'normal'

        app.processing_method.set(gui.ISSUE1_METHOD)
        app._processing_method_changed()
        assert str(app.accel_odr_combo.cget('state')) == 'disabled'
        assert str(app.accel_fft_n_combo.cget('state')) == 'disabled'
        assert str(app.accel_overlap_entry.cget('state')) == 'disabled'
        assert str(app.accel_averages_entry.cget('state')) == 'disabled'
        assert str(app.stage_block_combo.cget('state')) == 'disabled'
        assert str(app.units_combo.cget('state')) == 'readonly'
        assert str(app.live_threshold_mode_combo.cget('state')) == 'disabled'
        assert str(app.live_k_load_button.cget('state')) == 'disabled'
        assert 'press Apply' in app.processing_status.get()
        # Tabs describe the applied method, not merely the pending selection.
        assert tab_state('Issue 1 FFT ensemble') == 'disabled'

        app.apply_accel_processing_settings()
        assert tab_state('Issue 1 FFT ensemble') == 'normal'
        assert tab_state('PSD comparison') == 'disabled'
        assert tab_state('Calculation stages') == 'normal'
        assert tab_state('Gordon comparison') == 'normal'
        assert tab_state('Translation function') == 'normal'

        app.processing_method.set(gui.ISSUE2_METHOD)
        app._processing_method_changed()
        assert str(app.accel_fft_n_combo.cget('state')) == 'readonly'
        assert str(app.accel_overlap_entry.cget('state')) == 'normal'
        assert str(app.stage_block_combo.cget('state')) == 'readonly'
        assert str(app.live_threshold_mode_combo.cget('state')) == 'readonly'
        app.apply_accel_processing_settings()
        assert tab_state('Issue 1 FFT ensemble') == 'disabled'
        assert tab_state('PSD comparison') == 'normal'
    finally:
        root.destroy()


def test_shared_stage_and_gordon_views_follow_the_applied_mode():
    """Both live and offline common views must dispatch from the applied mode."""
    gui, root, app = _gui_app()
    try:
        xyz = _three_axis_sine()
        frame = pd.DataFrame(xyz, columns=['X_g', 'Y_g', 'Z_g'])
        frame.insert(0, 'time_s', np.arange(len(frame)) / 200.0)
        columns = gui.Cols('time_s', 'X_g', 'Y_g', 'Z_g')
        app.reader.set(frame.copy(), columns, 200.0, source='Reader')
        for axis in 'XYZ':
            baseline = gui.Processor(f'Baseline {axis}')
            baseline_frame=frame.copy()
            baseline_frame[['X_g','Y_g','Z_g']]*=2.0
            baseline.set(baseline_frame, columns, 200.0, source=f'Baseline {axis}')
            app.baselines[axis] = baseline
        app.live_t = frame.time_s.tolist()
        app.live_x, app.live_y, app.live_z = [xyz[:, i].tolist() for i in range(3)]

        # The default applied method remains the pre-existing Issue 2 path.
        app.plot_stage()
        assert '512-sample block' in app.fig['Selected stage'][0].axes[0].get_title()
        app.plot_bands()
        assert app.fig['Gordon bands'][0].axes[0].get_title().startswith('Issue 2')
        app._update_live_stage_plot()
        assert app.live_analysis_fig['Live stages'][0]._suptitle.get_text().startswith('Live X-axis')
        app._update_live_threshold_plot()
        assert len(app.live_analysis_fig['Live threshold'][0].axes) == 1

        app.processing_method.set(gui.ISSUE1_METHOD)
        app._processing_method_changed()
        app.apply_accel_processing_settings()
        app.plot_stage()
        assert app.fig['Selected stage'][0]._suptitle.get_text().startswith('Issue 1')
        app.plot_bands()
        issue1_axis = app.fig['Gordon bands'][0].axes[0]
        assert issue1_axis.get_title().startswith('Issue 1')
        assert len(app.fig['Gordon bands'][0].axes) == 1
        assert 'µm/s' in issue1_axis.get_ylabel()

        # Unit changes are display conversions of the same Issue 1 Fourier
        # components, so their Gordon comparison ratio remains unchanged.
        x_frequency = np.asarray(issue1_axis.lines[0].get_xdata(), float)
        bin_index = int(np.argmin(np.abs(x_frequency - 20.3125)))
        velocity = float(issue1_axis.lines[0].get_ydata()[bin_index])
        app.units.set('RMS acceleration (g)')
        app._display_units_changed()
        acceleration = float(app.fig['Gordon bands'][0].axes[0].lines[0].get_ydata()[bin_index])
        assert np.isclose(velocity,
                          acceleration * G0 / (2 * np.pi * x_frequency[bin_index]) * 1e6)
        app.units.set('RMS displacement (µm)')
        app._display_units_changed()
        displacement = float(app.fig['Gordon bands'][0].axes[0].lines[0].get_ydata()[bin_index])
        assert np.isclose(displacement,
                          velocity / (2 * np.pi * x_frequency[bin_index]))

        app.plot_translation()
        translation_figure = app.fig['Translation function'][0]
        assert translation_figure._suptitle.get_text().startswith('Issue 1')
        assert translation_figure.axes[0].get_title().startswith('Issue 1 direct K')
        correction = np.asarray(translation_figure.axes[0].lines[0].get_ydata(), float)
        assert np.allclose(correction[np.isfinite(correction)], 2.0)
        baseline_values = np.asarray(translation_figure.axes[1].lines[0].get_ydata(), float)
        corrected_values = np.asarray(translation_figure.axes[3].lines[0].get_ydata(), float)
        valid = np.isfinite(baseline_values) & np.isfinite(corrected_values)
        assert np.allclose(corrected_values[valid], baseline_values[valid])
        app.units.set('RMS acceleration (g)')
        app._display_units_changed()
        assert 'RMS g' in app.fig['Translation function'][0].axes[1].get_ylabel()

        app._update_live_stage_plot()
        assert app.live_analysis_fig['Live stages'][0]._suptitle.get_text().startswith('Live Issue 1')
        app._update_live_threshold_plot()
        assert app.live_analysis_fig['Live threshold'][0].axes[0].get_title().startswith('Live Issue 1')
        assert len(app.live_analysis_fig['Live threshold'][0].axes) == 1
    finally:
        root.destroy()


def test_tilt_average_setting_does_not_change_vibration_or_shock():
    gui, root, app = _gui_app()
    try:
        count = issue1_required_samples()
        time_s = np.arange(count) / 200.0
        xyz = _three_axis_sine()
        xyz[500, 2] += 1.0
        app.live_t = time_s.tolist()
        app.live_x, app.live_y, app.live_z = [xyz[:, i].tolist() for i in range(3)]
        vibration_before = issue1_ensemble_fft(xyz).ensemble_amplitude_g.copy()
        shock_before = app._live_shock_series()[1].copy()

        app.tilt_average_samples.set(200)

        vibration_after = issue1_ensemble_fft(xyz).ensemble_amplitude_g
        shock_after = app._live_shock_series()[1]
        assert np.array_equal(vibration_after, vibration_before)
        assert np.array_equal(shock_after, shock_before)
    finally:
        root.destroy()


def test_shock_path_uses_raw_unfiltered_samples():
    _, root, app = _gui_app()
    try:
        count = 200
        xyz = np.zeros((count, 3))
        xyz[100, 0] = 1.0
        app.live_t = (np.arange(count) / 200.0).tolist()
        app.live_x, app.live_y, app.live_z = [xyz[:, i].tolist() for i in range(3)]
        _, magnitude, _, _ = app._live_shock_series()

        # With a 50-point trailing steady-state average, the raw 1 g impulse
        # produces 1 - 1/50 g at its aligned output sample.
        assert np.isclose(np.max(magnitude), 49.0 / 50.0)
    finally:
        root.destroy()
