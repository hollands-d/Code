# 1174 Vibration Signal Processing Explorer

Python/Tkinter desktop application for analysing Project 1174 accelerometer
captures, comparing reference/base vibration with reader-mounted measurements,
and deriving translation coefficients for installed-system use.

This is an engineering analysis and characterisation tool. It is not production
firmware and is not formal design-verification software.

## What the App Does

- Loads reference/base and reader accelerometer data from CSV or Excel files.
- Displays raw X/Y/Z acceleration and intermediate vibration-processing stages.
- Calculates vibration levels using either:
  - **Issue 2 - PSD / Gordon bands**, based on Hann-windowed PSDs integrated
    into Gordon one-third-octave bands.
  - **Issue 1 - FFT ensemble**, based on the fixed Issue 1 FIR and 19-window
    FFT-amplitude ensemble.
- Compares measured vibration with the Gordon Office criterion in acceleration,
  velocity or displacement units.
- Derives reader-to-reference translation factors for X-, Y- and Z-driven
  characterisation runs.
- Stores driven-axis translation results and exports complete XYZ translation
  sets with settings/traceability JSON.
- Supports live STM32/VMM accelerometer capture over serial, live plots, raw
  count logging, live translation, threshold checks, tilt display and shock
  diagnostics.
- Provides offline shock analysis and shock baseline/reader comparison for
  X-, Y- and Z-applied shock captures.

## Current Processing Modes

### Issue 2 - PSD / Gordon Bands

The Issue 2 path is the main Gordon-band vibration workflow.

Default processing settings:

- Sample rate: `200 Hz`
- FFT/PSD block length: `512 samples`
- Window: Hann
- Block overlap: `50%`
- PSD averages: `4`
- Total span for one calculation: `1280 samples`
- Frequency resolution: `0.390625 Hz`
- Gordon band centres: `4, 5, 6.3, 8, 10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80 Hz`

For each selected block the app removes the mean, applies the Hann window,
calculates a one-sided PSD in `g^2/Hz`, averages the requested overlapping PSDs,
then integrates the result into Gordon one-third-octave bands.

### Issue 1 - FFT Ensemble

The Issue 1 path is a separate fixed algorithm for the Issue 1 reference method.

Key settings:

- Sample rate: `200 Hz`
- FIR passband: `3.5-90 Hz`
- FFT length: `128 samples`
- FFT overlap: `50%`
- Ensemble count: `19 FFT windows`
- Frequency resolution: `1.5625 Hz`

Issue 1 works on FFT amplitudes, not PSD/Gordon-band RMS values. PSD-only
reference files are converted directly to equivalent Issue 1 FFT-bin amplitudes
by integrating PSD over each FFT bin and applying the Issue 1 FIR magnitude
response. The app does not create synthetic time histories for this conversion.

## Supported Input Files

### Time-Domain CSV/XLSX

Reader and reference time-domain files should contain:

- A time column such as `time_s`, `time` or `timestamp`.
- X/Y/Z acceleration columns such as `X_g`, `Y_g`, `Z_g`.

The app also accepts supported raw VMM count logs and converts them to canonical
time plus X/Y/Z acceleration in g.

### Simple PSD CSV/XLSX

Frequency-domain reference files may provide:

- `frequency_hz`
- `psd_g2_per_hz`

PSD references support Gordon-band comparison, Issue 2 translation and Issue 1
PSD-to-FFT translation. PSD-only files do not support time overlays, phase, H1
or coherence because they do not contain synchronous time samples.

### Integrated Technologies Test-House Files

Integrated Technologies paired Hz/magnitude tables are supported from CSV or
Excel `.xlsx` files.

- `Ctl` is the measured control accelerometer PSD and is the only source used
  for translation calculations.
- `Ref` is retained for diagnostic baseline plotting only.
- `Ref` never contributes to K or translation factors.
- Alarm and CA channels are discarded.
- Lateral files can populate X, Y or both X+Y depending on the selected
  orientation/mapping.
- Vertical files populate Z.
- Reader PSD imports are Ctl-only and represent one selected reader axis.

Excel loading requires `openpyxl`, which is listed in `requirements.txt`.

## Characterisation Workflow

1. Load reference/base vibration files with **Load baseline/reference file(s)**.
2. Load an X-, Y- or Z-excited reader capture with **Load reader CSV/XLSX**.
3. Confirm the driven axis and baseline mapping.
4. Review raw data, PSD/Gordon views, Issue 1 views, transfer diagnostics and
   translation plots as appropriate for the selected processing method.
5. Use **Store driven-axis K** to retain the approved result for the driven
   axis.
6. Repeat for X, Y and Z.
7. Use **Export final XYZ K** to write one complete method-consistent
   translation set.

Only the deliberately driven axis is used to create the primary translation for
that run. Orthogonal channels remain useful diagnostics for cross-axis response,
but they are not silently promoted into primary correction factors.

Final XYZ export rejects mixed Issue 1 / Issue 2 sets. Re-derive all three axes
using the same method before exporting a complete set.

## Exported Files

Translation CSV files contain one row per coefficient:

- Issue 2 exports 14 Gordon-band coefficients per axis.
- Issue 1 exports one coefficient per active Issue 1 FFT coordinate per axis.

Exports also write a companion `*_settings.json` file containing traceability
information such as:

- application title
- processing method
- source files
- selected reference axis
- reference domain
- processing settings
- parser metadata
- exported coefficient rows

Shock summary export writes a CSV plus settings JSON describing the shock
processing settings and per-axis results.

## Main Views

- **Raw comparison**: time-domain baseline/reader X/Y/Z overlay where
  synchronous samples are available.
- **Selected stage**: selected processing block, window, FFT and PSD stages.
- **PSD comparison**: reference/reader PSD overlays, including Ctl/Ref baseline
  diagnostics for test-house data.
- **Gordon bands**: Gordon-band or Issue 1 threshold comparison in the selected
  display units.
- **Transfer function**: same-axis H1/coherence diagnostics for paired
  time-domain captures.
- **Translation function**: derived reader-to-reference translation factors.
- **Issue 1 FFT ensemble**: Issue 1 filtered time history and FFT-amplitude
  ensemble details.
- **Tilt comparison**: gravity-vector tilt and mounting offset diagnostics.
- **Shock analysis / Shock translation / Shock summary**: offline shock
  processing and baseline/reader shock comparison.
- **Live STM data and live analysis tabs**: streaming raw values, Gordon bands,
  Issue 1 live ensemble, live translation, thresholds, tilt and raw-count
  diagnostics.

## Live Capture Notes

Live capture uses the STM32/VMM serial streaming path implemented in
`vmm_stream.py` and the direct ST-Link VCP helpers in `live_stm_serial.py`.

Important notes:

- Live serial work runs off the Tk main thread; GUI updates are queued back to
  the main thread.
- Raw-count logging writes converted samples and capture settings for
  traceability.
- Live translation is method-aware. Issue 1 translation files only apply in
  Issue 1 mode; Issue 2 translation files only apply in Issue 2 mode.
- Live threshold displays compare the displayed translated values when live
  translation is enabled, and raw reader values otherwise.
- Shock detection uses raw converted X/Y/Z acceleration and is independent of
  vibration translation factors.

## Installation

Use Python 3.11 or newer on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `pyserial`
- `openpyxl`

## Running

```powershell
python vibration_signal_processing_gui.py
```

## Tests

Run the regression tests with:

```powershell
python -m pytest
```

Some tests instantiate Tkinter windows. They require a working Tcl/Tk
installation and, on headless systems, an available display or virtual display.

## Repository Layout

- `vibration_signal_processing_gui.py`: main Tkinter application, offline plots,
  translation workflow, live UI and exports.
- `issue1_processing.py`: Issue 1 FIR, FFT ensemble, threshold and PSD-to-FFT
  conversion functions.
- `accelerometer_csv.py`: accelerometer CSV/raw-count normalisation helpers.
- `vmm_stream.py`: VMM serial frame protocol and live sample decoding.
- `live_stm_serial.py`: direct ST-Link VCP protobuf request/response helpers.
- `live_session.py`: live session timing, renewal and operator-timeout state.
- `test_*.py`: regression and smoke tests for processing, file import,
  translation and GUI workflows.

## Important Interpretation Notes

- Translation factors are empirical characterisation data for the reader
  installation and test setup. They should be reviewed before use.
- `Ctl` is authoritative for test-house PSD calculations; `Ref` is diagnostic.
- Cross-axis response is diagnostic unless a workflow explicitly stores that
  axis as the driven-axis result.
- PSD-only files cannot prove synchronous phase, H1 or coherence.
- Shock analysis is a separate raw-time-domain path and should not be
  interpreted as a vibration translation result.
