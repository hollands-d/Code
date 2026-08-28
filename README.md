# 1174 Vibration Signal Processing Explorer

A Windows-friendly Python/Tkinter desktop application for offline engineering analysis of Project 1174 accelerometer data.

The application is intended to support development and characterisation of the reader vibration-detection method described in the revised 1174-Y-056 accelerometer signal-processing work.

It is an engineering analysis tool only. It is not production firmware and is not formal design-verification software.

---

## 1. Purpose

The application is intended to help visualise and evaluate the complete vibration-processing chain from raw accelerometer samples through to Gordon-band vibration metrics and installed-system correction.

It supports two related activities:

1. **Signal-processing development**
   - Inspect every major processing stage.
   - Confirm PSD scaling, windowing, averaging and one-third-octave integration.
   - Compare acceleration, velocity and displacement representations.
   - Compare measured vibration with the Gordon Office criterion.

2. **Installed-system characterisation**
   - Compare a calibrated baseline/reference accelerometer with the reader-mounted accelerometer.
   - Identify amplification, attenuation/deadening, resonances and cross-axis response.
   - Derive a practical translation/correction from the reader measurement back to equivalent base excitation.
   - Compare simple and band-dependent correction models.
   - Visualise small static mounting/tilt offsets.

The production accelerometer is mounted on its own PCB fixed to the reader base above the front foot. Because the reader structure, PCB, feet and mounting can alter the vibration seen by the sensor, the reader accelerometer should not automatically be assumed to measure the same vibration as the support surface.

---

## 2. Current processing approach

The default vibration-processing parameters follow the revised 1174-Y-056 approach:

- Accelerometer sample rate: **200 Hz**
- Accelerometer operating assumption: high-performance mode
- FFT / PSD block length: **512 samples**
- Block duration: **2.56 s**
- Window: **Hann**
- Block overlap: **50%**
- New spectral block every: **1.28 s**
- PSD estimates averaged: **4**
- Total data span for four overlapping blocks: **1280 samples / approximately 6.4 s**
- Frequency resolution:

```text
Delta f = fs / N = 200 / 512 = 0.390625 Hz
```

Before the FFT:

1. The mean of each selected block is removed.
2. The mean-removed samples are multiplied by a Hann window.
3. A real FFT is calculated.
4. The FFT is converted to a one-sided power spectral density in `g^2/Hz`.
5. Hann window power normalisation is applied.
6. Several overlapping PSD estimates are averaged.
7. The averaged PSD is integrated within the Gordon one-third-octave bands.

No additional 3.5–90 Hz application-level FIR filter is applied in the current revised vibration path. The initial revised approach relies on the accelerometer's internal anti-aliasing behaviour, mean subtraction, the FFT, and selection/integration of the required frequency bands.

---

## 3. Gordon one-third-octave bands

The application uses the following nominal Gordon band-centre frequencies:

```text
4
5
6.3
8
10
12.5
16
20
25
31.5
40
50
63
80 Hz
```

The plots use logarithmic frequency spacing, but every Gordon band centre is explicitly labelled on the x-axis.

This makes the one-third-octave structure easier to read than a normal logarithmic axis showing only powers of ten.

---

## 4. Gordon Office criterion

The application represents the Gordon Office criterion as:

- **400 micrometres/second RMS constant velocity from 8 Hz upwards**
- a **constant-acceleration continuation below 8 Hz**

The 8 Hz transition acceleration is:

```text
a = 2*pi*f*V
  = 2*pi*8*400e-6
  = approximately 0.0201 m/s^2
  = approximately 0.00205 g RMS
```

Therefore, below 8 Hz the equivalent allowable RMS velocity rises as frequency decreases.

Examples:

| Band centre | Office RMS velocity | Office RMS acceleration |
|---|---:|---:|
| 4 Hz | ~800 µm/s | ~0.00205 g |
| 5 Hz | ~640 µm/s | ~0.00205 g |
| 6.3 Hz | ~508 µm/s | ~0.00205 g |
| 8 Hz | 400 µm/s | ~0.00205 g |
| 10 Hz | 400 µm/s | ~0.00256 g |
| 20 Hz | 400 µm/s | ~0.00512 g |
| 40 Hz | 400 µm/s | ~0.01025 g |
| 80 Hz | 400 µm/s | ~0.0205 g |

---

## 5. Band integration

For each one-third-octave band, the acceleration mean-square value is calculated by integrating the PSD:

```text
a_rms_band^2 = sum(PSD(f_k) * Delta f)
```

Then:

```text
a_rms_band = sqrt(sum(PSD(f_k) * Delta f))
```

The initial implementation assigns an FFT bin to the band containing that bin's centre frequency.

This is intentionally simple. More detailed fractional-bin boundary weighting can be added later if physical/numerical testing shows that it is required.

---

## 6. Selectable Gordon-band units

The Gordon Bands tab can display the processed result in three different forms.

### RMS acceleration

```text
a_rms
```

Displayed in:

```text
g RMS
```

### RMS velocity

For each band:

```text
v_rms = a_rms * g0 / (2*pi*f_c)
```

Displayed in:

```text
µm/s RMS
```

where:

- `g0 = 9.80665 m/s^2`
- `f_c` is the one-third-octave band centre frequency

### RMS displacement

Displacement is derived from RMS velocity:

```text
x_rms = v_rms / (2*pi*f_c)
```

Displayed in:

```text
µm RMS
```

The Gordon Office reference curve is converted into the equivalent units for whichever display mode is selected.

---

## 7. Input data format

The application accepts CSV data.

A typical file contains:

```csv
time_s,X_g,Y_g,Z_g
0.000,0.0012,-0.0008,1.0009
0.005,0.0014,-0.0007,1.0011
...
```

Expected units:

- Time: seconds
- X acceleration: g
- Y acceleration: g
- Z acceleration: g

If a time column is not present, the application can construct time using the configured sample rate.

---

## 8. Baseline/reference and reader datasets

The paired-analysis workflow uses two datasets.

### Baseline/reference dataset

This represents the vibration applied at the support/base.

For real physical characterisation this should come from the calibrated reference accelerometer mounted as close as practical to the reader support/contact region.

### Reader dataset

This represents the output of the accelerometer installed on the production-style reader PCB.

Both datasets normally contain X, Y and Z acceleration channels.

---

## 9. Single driven-axis characterisation

A very important assumption in the current characterisation plan is:

**each dataset represents only one intentionally driven vibration axis.**

For example, during an X-axis test:

- the shaker intentionally excites the reader in X;
- the reference accelerometer records X, Y and Z where available;
- the reader accelerometer records X, Y and Z;
- only **reference X versus reader X** is used to derive the X-axis transfer/correction function.

The measured reader Y and Z responses are still useful, but they are treated as cross-axis diagnostic information rather than being included in the X-axis correction.

The same applies for Y- and Z-driven tests.

The GUI therefore has two independent selectors:

### Axis selection (v22: local to axis-specific tabs)

Defines the axis intentionally excited during the current test.

This axis is used for:

- amplitude transfer ratio
- production correction factor
- H1 frequency-response estimate
- coherence
- translation-function model selection
- exported correction coefficients

### Displayed / inspection axis

Allows X, Y or Z to be inspected independently in the raw, PSD and other diagnostic views.

Changing the inspection axis does not alter which axis is used to derive the transfer function.

---

## 10. Cross-axis response

Although non-driven axes are not used in the translation calculation, they should still be retained and reviewed.

For an X-driven test, reader Y or Z motion can indicate:

- structural coupling
- rocking of the reader
- PCB flexure
- foot/contact effects
- local structural resonances
- accelerometer axis misalignment
- mechanical asymmetry

The current production translation model does **not** use these cross-axis channels.

If future physical testing shows that cross-axis coupling is sufficiently large and repeatable to require correction, a more complex matrix-based model could be considered later.

That is not the default approach.

---

## 11. Transfer function

For the driven axis, the band amplitude transfer is:

```text
H_axis,band = Reader_RMS_axis,band / Base_RMS_axis,band
```

Interpretation:

```text
H > 1    reader locally amplifies the base vibration
H = 1    reader follows the base vibration directly
H < 1    reader attenuates / deadens the base vibration
```

For example:

```text
H = 1.40
```

means the installed reader reports 40% greater vibration amplitude than the true/base input in that band.

---

## 12. Production correction factor

The STM32 requires the inverse relationship:

```text
K_axis,band = Base_RMS_axis,band / Reader_RMS_axis,band
```

or:

```text
K_axis,band = 1 / H_axis,band
```

The recommended production translation is therefore:

```text
Base_est_axis,band =
    K_axis,band * Reader_RMS_axis,band
```

For example, if:

```text
H_X,20Hz = 1.40
```

then:

```text
K_X,20Hz = 1 / 1.40 = 0.714
```

If the reader measures:

```text
560 µm/s RMS
```

the estimated base vibration becomes:

```text
560 * 0.714 = approximately 400 µm/s RMS
```

---

## 13. Where the correction should be applied

The preferred approach is to apply the correction **after** one-third-octave band integration and after the square root has produced a band RMS amplitude.

The processing sequence is therefore:

```text
Raw X/Y/Z samples
        |
Mean removal
        |
Hann window
        |
FFT
        |
One-sided PSD
        |
Average PSD estimates
        |
Integrate Gordon band
        |
Square root
        |
Reader band RMS
        |
Multiply by K
        |
Estimated base band RMS
        |
Compare with vibration criterion / threshold
```

This is preferred because it is simple and computationally cheap.

If the PSD itself were corrected before the square root, the multiplier would need to be:

```text
K^2
```

because PSD is proportional to amplitude squared.

---

## 14. Translation-function model selection

For one single-axis characterisation dataset, the application compares two candidate implementations.

### Model 1 — one correction factor for the driven axis

```text
Base_est_axis = K_axis * Reader_axis
```

This is the simplest implementation.

### Model 2 — one correction factor per Gordon band

```text
Base_est_axis,band =
    K_axis,band * Reader_axis,band
```

This uses 14 coefficients for one driven axis.

The application divides the paired data into training and validation windows and compares the residual error of the candidate correction structures.

The default model-selection logic prefers the simpler single-K model when the validation fit is sufficiently good. Otherwise it recommends the frequency-band-dependent correction.

Separate X-, Y- and Z-driven characterisation results can later be combined.

If all three axes require band-dependent correction, the final production implementation would contain:

```text
3 axes * 14 bands = 42 coefficients
```

---

## 15. Translation Function tab

The Translation Function tab shows the production-oriented interpretation of the characterisation results.

It displays:

1. **Recommended STM translation equation**
2. **Model validation error**
3. **Derived correction coefficient(s)**
4. **Comparison of:**
   - true/base vibration
   - raw reader vibration
   - corrected reader vibration

The purpose is to make it visually obvious whether the proposed correction actually recovers the baseline/reference excitation.

---

## 16. Transfer Function tab

The Transfer Function tab is focused on engineering characterisation.

For the selected driven axis it shows:

- band amplitude transfer:

```text
H = Reader / Base
```

- production correction:

```text
K = Base / Reader
```

- H1 frequency-response magnitude
- magnitude-squared coherence

The H1/coherence plots are engineering diagnostics and are not intended to be reproduced directly inside the STM32.

They are useful for identifying:

- structural resonances
- damping
- phase-related behaviour
- noisy/unreliable frequency regions
- whether the reader/base relationship is sufficiently repeatable for a simple band correction

---

## 17. H1 frequency-response estimate

The application calculates an H1-style frequency-response estimate for the driven axis from paired baseline and reader signals.

Conceptually:

```text
H1(f) = G_yx(f) / G_xx(f)
```

where:

- `x` is the baseline/reference signal
- `y` is the installed-reader signal

This is used as a characterisation diagnostic.

The production STM translation is still expected to use the much simpler Gordon-band correction factors unless physical evidence shows that a more complex implementation is required.

---

## 18. Coherence

Magnitude-squared coherence is also shown for the driven axis.

Values near:

```text
1
```

indicate a strong repeatable linear relationship between baseline and reader response at that frequency.

Low coherence can indicate:

- unrelated noise
- non-linearity
- poor synchronisation
- insufficient excitation
- changing mounting/contact conditions
- cross-axis/structural effects
- measurement artefacts

A correction factor derived from a low-coherence region should be treated cautiously.

---

## 19. Tilt processing

The raw accelerometer data are also used for tilt inspection.

A configurable moving average is used to estimate the steady gravity vector.

Simple roll-like and pitch-like inspection angles are then calculated.

These plots are intended for engineering visualisation and should not be interpreted as the final production tilt-coordinate implementation.

---

## 20. Tilt mounting offsets

The installed reader accelerometer will not normally report exactly:

```text
X = 0 g
Y = 0 g
Z = 1 g
```

when the complete reader is mechanically level.

Small offsets can result from:

- accelerometer package tolerance
- accelerometer placement on the PCB
- PCB mounting tolerance
- screw / spacer / support tolerance
- reader base tolerance
- assembled reader structure

The paired dummy datasets therefore include a small static mounting-angle offset so the GUI can demonstrate this behaviour.

---

## 21. Tilt Comparison tab

The Tilt Comparison tab provides several graphical views.

It includes:

- baseline and reader mean gravity vector in the X-Z plane
- baseline and reader mean gravity vector in the Y-Z plane
- mean roll-like and pitch-like angles
- reader-minus-baseline relative tilt versus time

This makes very small mounting errors visible in a much clearer way than simply looking at X/Y/Z offsets in raw g.

---

## 22. Tilt commissioning concept

The preferred production concept is to keep tilt calibration separate from dynamic vibration correction.

During commissioning:

1. Place the fully assembled reader on a calibrated level surface.
2. Keep the reader stationary.
3. Average X, Y and Z acceleration long enough to reduce noise.
4. Store the resulting gravity-vector reference for that individual instrument.
5. Use this stored reference as the reader's zero-tilt condition.

This compensates the complete assembled sensor/PCB/mechanical alignment rather than relying on nominal accelerometer alignment.

The vibration `K` correction and the tilt zero-reference calibration solve different problems and should remain separate.

---

## 23. Shock view

The application also retains the shock-processing concept.

Shock is treated as the instantaneous change from the moving steady-state acceleration vector.

Conceptually:

```text
Delta X = X_raw - X_steady
Delta Y = Y_raw - Y_steady
Delta Z = Z_raw - Z_steady
```

and:

```text
Shock magnitude =
sqrt(Delta X^2 + Delta Y^2 + Delta Z^2)
```

This is intentionally separate from the vibration PSD-averaging path because spectral averaging would reduce short transient peaks.

---

## 24. Main GUI tabs

### Raw comparison

Shows baseline/reference and reader X/Y/Z time histories.

Useful for:

- synchronisation
- offsets
- gross amplification
- cross-axis behaviour
- clipping
- dropouts
- transients

### Selected stage

Shows the processing of the selected block step-by-step:

- raw block
- mean-removed block
- Hann window
- windowed signal
- FFT magnitude
- PSD

### PSD comparison

Shows individual and/or averaged PSD behaviour.

Useful for checking:

- resonances
- noise
- frequency content
- baseline-reader differences

### Gordon bands

Shows baseline, reader and Gordon Office reference in selectable:

- RMS g
- RMS µm/s
- RMS µm

### Transfer function

Shows driven-axis:

- H
- K
- H1 magnitude
- coherence

### Translation function

Shows:

- recommended production equation
- model selection
- derived K values
- translated/corrected reader result versus true baseline

### Tilt comparison

Shows:

- gravity-vector difference
- roll/pitch-like offsets
- relative mounting tilt

---

## 25. Exporting correction coefficients

The application exports one combined X/Y/Z correction CSV using each axis's mapped baseline.

Example:

```csv
driven_axis,fc_hz,recommended_model,K_base_over_reader
X,4,Band-specific K,1.04
X,5,Band-specific K,1.07
X,6.3,Band-specific K,1.09
...
X,80,Band-specific K,0.94
```

The intention is that separate X-, Y- and Z-driven datasets are analysed and their final approved coefficients are then combined into the controlled production configuration.

---

## 26. Dummy datasets supplied with the application

The ZIP includes synthetic engineering datasets so the processing can be explored before physical test data are available.

### `1174_dummy_office_level_lateral_axes.csv`

Represents the baseline/reference vibration.

It is based on an Office-level lateral vibration profile and is intended to represent support/base excitation.

### `1174_dummy_reader_pcb_mounting_effects.csv`

Represents a synthetic reader-mounted accelerometer response.

It deliberately includes:

- frequency-dependent amplification
- frequency-dependent attenuation/deadening
- local resonance-like behaviour
- small phase effects
- cross-axis response
- sensor noise
- small static tilt/mounting offset

These effects are intentionally illustrative and are **not** predictions of the real production reader.

### `1174_dummy_reader_transfer_truth.csv`

Contains the nominal synthetic transfer behaviour used when creating the dummy reader dataset.

This allows the calculated transfer/correction to be compared with the known simulated relationship.

---

## 27. Recommended physical characterisation workflow

For real characterisation, perform separate controlled runs for:

```text
X excitation
Y excitation
Z excitation
```

For each run:

1. Apply vibration in one known axis only.
2. Record the calibrated baseline/reference accelerometer.
3. Record all X/Y/Z channels from the reader accelerometer.
4. Record sample rate, timing, orientation and test metadata.
5. Check timebase/synchronisation.
6. Review raw response.
7. Calculate PSDs.
8. Integrate Gordon bands.
9. Calculate transfer only on the intentionally driven axis.
10. Review non-driven axes for cross-axis behaviour.
11. Calculate `H`.
12. Calculate `K`.
13. Review H1 and coherence.
14. Determine whether one K or band-dependent K is required.
15. Repeat over multiple vibration levels.
16. Repeat with multiple readers.
17. Include placement/remount repeats.
18. Evaluate uncertainty and production variation.
19. Select a conservative production correction structure.
20. Verify the implemented production method separately.

---

## 28. Multiple vibration levels

The final correction should not be based on a single vibration amplitude.

Characterisation should include several levels below, at and above Office.

For each driven axis and band, compare:

```text
Base_RMS
```

with:

```text
Reader_RMS
```

over multiple levels.

If the relationship is approximately linear, a single slope/correction coefficient can be used.

A useful fitting form is:

```text
Base_RMS = K * Reader_RMS
```

with the fit constrained through zero where physically justified.

This is generally more robust than calculating one individual ratio from one noisy measurement.

---

## 29. Multiple readers and mounting variation

Production correction coefficients should be based on representative hardware rather than one specially selected reader.

Characterisation should therefore consider:

- multiple representative readers
- repeated removals/replacements on the test surface
- normal accelerometer PCB mounting variation
- normal foot/contact variation
- normal assembly tolerance

The final coefficient set should be the simplest implementation that remains representative and sufficiently conservative across the expected production population.

---

## 30. Production STM32 implementation concept

The expected production calculation is lightweight.

For a band-dependent implementation:

```c
base_est_rms = K[axis][band] * reader_rms;
```

If all three axes require separate 14-band correction:

```text
K[3][14]
```

contains 42 coefficients.

The coefficients can be stored as controlled configuration/calibration constants.

The computational overhead is extremely small relative to the FFT/PSD processing.

---

## 31. Important distinction: RMS correction vs PSD correction

If correction is applied after RMS calculation:

```text
RMS_corrected = K * RMS_measured
```

Use:

```text
K
```

If correction is applied directly to PSD:

```text
PSD_corrected = K^2 * PSD_measured
```

Use:

```text
K^2
```

The application and recommended production architecture use the first method.

---

## 32. Important limitations

This application is intended for engineering development.

It does not currently:

- replace formal verification
- establish the final production vibration threshold
- prove the final LIS2DUX12 anti-alias filter configuration
- automatically correct clock drift or time alignment between unrelated DAQ systems
- implement a full 3x3 cross-axis transfer matrix
- establish measurement uncertainty
- automatically pool multiple readers/runs into a controlled production coefficient set
- replace physical characterisation
- prove that the chosen K structure is conservative across production

Those activities require physical evidence and controlled engineering review.

---

## 33. Installation

Python 3.10 or newer is recommended.

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run the application:

```powershell
py vibration_signal_processing_gui.py
```

If the `py` launcher is not installed:

```powershell
python -m pip install -r requirements.txt
python vibration_signal_processing_gui.py
```

Tkinter is normally included with the standard Windows Python installer.

---

## 34. Typical quick-start workflow

To explore the bundled dummy data:

1. Start the application.
2. Click **Load paired demo**.
3. Review the **Correction baseline mapping** (normally X→X, Y→Y, Z→Z). Axis selectors now live only on axis-specific diagnostic tabs.
4. Review **Raw comparison**.
5. Review **Selected stage**.
6. Review **PSD comparison**.
7. Review **Gordon bands**.
8. Switch Gordon units between:
   - RMS velocity
   - RMS acceleration
   - RMS displacement
9. Review **Transfer function**.
10. Review **Translation function**.
11. Review **Tilt comparison**.
12. Export the driven-axis correction CSV if required.

---

## 35. File contents

Typical package contents:

```text
vibration_signal_processing_gui.py
requirements.txt
README.md
correction_factors_example.csv
1174_dummy_office_level_lateral_axes.csv
1174_dummy_reader_pcb_mounting_effects.csv
1174_dummy_reader_transfer_truth.csv
```

---

## 36. Engineering interpretation

The intended overall approach is:

```text
Reference/base vibration
          |
          | physical reader structure
          v
Installed accelerometer response
          |
          | signal processing
          v
Reader Gordon-band RMS
          |
          | K correction derived from characterisation
          v
Estimated equivalent base Gordon-band RMS
          |
          v
Production vibration decision
```

The characterisation tool is intended to establish and justify the `K` relationship.

The production firmware should then use the simplest correction structure that is demonstrated to be adequate by the physical data.


---

## 37. Variable dataset duration and automatic cropping

Physical acquisition files do not need to be exactly 60 seconds long.

Version 8 added a toggle:

```text
Auto-crop paired datasets to common usable duration
```

The original imported CSV files are not modified.

When auto-crop is enabled, the application:

1. Restores the complete original baseline and reader datasets.
2. If both files contain usable timestamps, identifies the common overlapping time interval.
3. Crops both analysis copies to that common interval.
4. Reduces them to the same paired sample count.
5. Calculates the length of one complete PSD-analysis span:

```text
span = N + (PSD_segments - 1) * hop
```

For the default settings:

```text
N = 512
overlap = 50%
hop = 256
PSD segments = 4
span = 512 + 3*256 = 1280 samples
```

At 200 Hz:

```text
1280 / 200 = 6.4 seconds
```

6. Trims only the final unused tail so that the retained dataset contains a whole number of complete 6.4-second analysis spans.

This avoids silently using an incomplete final PSD set in the translation-model fitting and validation.

For example, a paired recording of 58.7 seconds does not need to be manually edited to 60 seconds. The application can retain the largest complete common analysis duration automatically.

When auto-crop is disabled, the complete imported datasets are retained. Paired calculations still require sufficient samples to exist in both datasets for the selected analysis range.

The status bar and dataset labels show the original and retained sample counts.

---

## 38. Selected Stage tab and the four overlapping FFT blocks

The **Selected Stage** tab does **not** represent the complete four-PSD average on a single plot.

It is deliberately a detailed inspection of **one 512-sample FFT block at a time** so that every processing step can be inspected clearly.

With the default settings:

```text
FFT length N = 512 samples
sample rate = 200 Hz
block duration = 2.56 seconds
overlap = 50%
hop = 256 samples = 1.28 seconds
```

A complete four-block PSD-average set beginning at sample `S` uses:

```text
Block 1: S       to S + 511
Block 2: S + 256 to S + 767
Block 3: S + 512 to S + 1023
Block 4: S + 768 to S + 1279
```

Version 8 adds a **Selected-stage FFT block** control.

The user can select block:

```text
1
2
3
4
```

and inspect that exact 512-sample segment through:

```text
Raw samples
    ↓
Mean removal
    ↓
Hann window
    ↓
Windowed signal
    ↓
FFT
    ↓
Single-block PSD
```

The **PSD Comparison** and Gordon-band calculations still use the averaged PSD from all four overlapping blocks.

The `PSD-set start sample` field defines `S`, the beginning of the complete four-block set.

Therefore:

- Selected Stage = one chosen 512-sample component of the PSD set.
- PSD Comparison = averaged result from all configured overlapping blocks.
- Gordon Bands = band integration of the averaged PSD.
- Transfer / Translation = calculations based on those processed band results.

This separation is intentional because it allows the effect of mean subtraction, windowing and the FFT to be inspected without visually combining four different time segments.

---

## 39. Live STM32 accelerometer acquisition (v9)

Version 9 adds a **Live STM data** tab so the signal-processing application can also act as a live engineering monitor for the reader-mounted accelerometer.

The implementation is based on the connection and broker transport approach in the supplied `psyros-platform-feature-hw-diagnostics-laptop-gui` development branch.

### Connection architecture

The live connection follows this path:

```text
Windows analysis laptop
        |
        | SSH
        v
Toradex SOM
        |
        | local TCP relay
        v
/run or ~/run/psyros/broker.sock
        |
        | broker unified-wire framing / protobuf
        v
STM32 slave-board firmware
        |
        v
Accelerometer FIFO buffers
```

The supplied diagnostics tooling already contains:

- a Paramiko-based helper that keeps the SOM broker TCP relay alive;
- an SSH local-port-forward helper;
- the broker unified-wire framing implementation;
- generated Python protobuf classes matching the STM32 message definitions.

The v9 application bundles the required runtime subset of those modules so the vibration tool does not depend on a separately installed copy of the colleague GUI.

### SOM prerequisite

The live connection assumes the same development setup as the supplied HW diagnostics tool:

```text
broker.sock is running on the SOM
/home/torizon/broker_tcp_relay.py exists on the SOM
SSH access to the SOM is available
```

If these are not present, the Connect operation reports the relay/tunnel error rather than silently falling back to another interface.

### Existing firmware messages used

No new STM32 message type is required for this first implementation.

The supplied firmware already supports:

```text
mGetAccelerometerNumBuffers
        -> mAccelerometerNumBuffers

mGetAccelerometerDataMsg
        -> mAccelerometerDataMsg
```

The first request reports how many accelerometer memory-pool buffers are ready for transfer.

The second request returns one ready buffer as a byte array.

The application only requests accelerometer data after the firmware reports at least one ready buffer. This is important because the current firmware data handler expects a valid ready-buffer pointer when a data request is made.

### STM accelerometer packet format

The firmware packs each accelerometer sample as:

```c
int16_t X
int16_t Y
int16_t Z
```

Therefore each sample occupies:

```text
6 bytes
```

The Python application decodes the returned byte array as little-endian signed 16-bit X/Y/Z triples.

### Current FIFO behaviour

The supplied firmware uses an accelerometer FIFO threshold of approximately 220 samples and an accelerometer memory pool containing six buffers.

At the nominal 200 Hz sample rate:

```text
220 / 200 = approximately 1.1 seconds
```

Consequently, **live** in the current application means buffered live acquisition rather than a new USB/UART packet every 5 ms.

The graph normally updates when a FIFO buffer becomes available, approximately once per second at the present firmware settings. The application immediately drains all reported ready buffers to minimise loss if more than one has accumulated.

This is appropriate for the engineering vibration display because the production spectral decision itself operates over several seconds of data.

### Sample timing

The current `AccelerometerDataMsg` contains raw XYZ sample bytes but does not include a timestamp or sample counter for each buffer/sample.

The application therefore reconstructs capture time as:

```text
time_s = sample_index / configured_sample_rate
```

with 200 Hz as the normal setting.

This means the live graph shows the accelerometer's nominal sample timeline, not the arrival time of each network/SSH packet.

For formal synchronisation with an external shaker/reference DAQ, timestamping or another synchronisation method is still required. Network arrival timing should not be treated as accelerometer sample timing.

---

## 40. Live-data scaling

The current `hw_test_vmm` path returns raw signed LIS2DUX12 counts and includes
the active full-scale range, `fs_g`, in every sample block. The GUI therefore
uses the protocol-defined conversion:

```text
g_per_count = fs_g / 32768
sample_g = raw_count * g_per_count
```

For a reported ±2 g range this is approximately `0.061035 mg/LSB`. The former
editable `0.061 mg/LSB` LIS2DS12 assumption is not used by the active VMM path.
The displayed full scale and sample rate come from each received block.

---

## 41. Live STM Data tab

The Live STM Data tab provides the following controls.

### Connection settings

```text
SOM IP / hostname
SSH username
SSH private-key path
SSH password (optional)
Local forwarded broker port
SOM broker-relay TCP port
Counts-to-g scale in mg/LSB
Live plot time window
```

### Connection controls

```text
Connect
Disconnect
```

Connect starts the SOM relay using the same approach as the supplied laptop HW diagnostics tool, opens a local SSH tunnel and then connects the Python application to the broker transport.

### Acquisition controls

```text
Start capture
Stop capture
Clear plot/capture
```

A **Clear existing STM buffer backlog at start** option is enabled by default.

When enabled, any already-completed accelerometer buffers are consumed before the new capture starts. This helps ensure that the displayed time origin corresponds approximately to the requested start of the new capture rather than data that accumulated beforehand.

### Data controls

```text
Save capture CSV
Use capture as Reader dataset
```

A live capture can therefore be saved in the same format as the offline analysis files:

```csv
time_s,X_g,Y_g,Z_g
```

It can also be transferred directly into the application's Reader dataset without first saving/reloading the CSV.

If a baseline/reference dataset is already loaded, the normal paired offline analysis can then be run on the captured reader data.

---

## 42. Live graphical display

The live display is arranged to show both immediate mechanical behaviour and the processing method used elsewhere in the application.

### Scrolling raw XYZ acceleration

The upper graph displays a configurable recent time window of:

```text
X acceleration
Y acceleration
Z acceleration
```

in g.

This is useful for identifying:

- axis orientation;
- obvious vibration response;
- structural cross-coupling;
- knocks/shocks;
- static mounting offsets;
- sensor clipping or unexpected jumps.

### Live Gordon-band calculation

Once enough live samples have accumulated for a complete spectral estimate, the lower-left graph processes the most recent data using the same configured method as the offline tool:

```text
512 sample FFT
50% overlap
4 PSD estimates
Hann window
mean subtraction
one-sided PSD
Gordon one-third-octave integration
```

For the default parameters this requires:

```text
1280 samples = approximately 6.4 seconds at 200 Hz
```

The Live STM data summary plots **X, Y and Z Gordon bands together**. In v22 there is no global driven-axis selector: the Transfer function has its own local axis selector, Selected stage has its own local axis selector, and general comparison/translation views show all three axes.

It can be displayed as:

- RMS acceleration in g;
- RMS velocity in µm/s;
- RMS displacement in µm.

The Gordon Office reference is displayed in the same selected units.

This live spectral display is intended for engineering visibility. It does not by itself constitute the production vibration alarm implementation.

### Live tilt summary

The lower-right graph on the **Live STM data** summary is the two-axis tilt target formerly shown on the separate Live tilt tab. It displays relative pitch and roll about the stored/automatic zero reference, together with the averaged gravity vector, |g| and dynamic RMS validity information.

The dedicated Live tilt tab has been removed to avoid duplication. The Zero tilt, averaging and maximum dynamic RMS controls remain available in the collapsible Direct STM32 connection section.

---

## 43. Using live capture during vibration characterisation

A practical engineering workflow is:

1. Connect the laptop vibration tool to the reader/SOM.
2. Select the intended driven excitation axis.
3. Start the live accelerometer capture.
4. Confirm raw XYZ orientation and that the expected axis responds.
5. Observe the non-driven axes for structural cross-coupling.
6. Wait for enough data for the live Gordon-band display.
7. Apply the controlled shaker level.
8. Use the live plots as an operator/engineering check.
9. Stop the capture at the end of the stage/run.
10. Save the raw live capture CSV.
11. Preserve the independent calibrated reference accelerometer data.
12. Use the saved paired datasets for the full offline transfer-function and translation analysis.

The live view should not replace the calibrated reference accelerometer used to define true/base excitation.

---

## 44. Important live-data limitations

The first live implementation intentionally reuses the existing firmware/protobuf transport rather than introducing a new streaming protocol.

Current limitations include:

- accelerometer samples are received in FIFO chunks rather than sample-by-sample;
- the returned buffer contains no per-sample timestamp;
- time is reconstructed from nominal sample rate;
- the present STM accelerometer driver in the supplied branch targets LIS2DS12 while current PCB hardware is LIS2DUX12;
- the counts-to-g conversion must therefore be verified for the final hardware/firmware combination;
- external reference/shaker data are not currently streamed into the same live graph;
- network/SSH packet timing must not be used as the vibration timebase;
- live plots are engineering diagnostics, not formal verification evidence;
- the broker TCP relay must already be supported by the SOM development setup.

A later enhancement could add a timestamp/sample counter to the STM accelerometer data message and optionally stream the calibrated reference DAQ into the same application for synchronised live base-versus-reader comparison.


---

## 39. Live STM32 connection — direct ST-Link Virtual COM (v10)

The live connection in version 10 follows the **direct laptop-to-STM32 diagnostics method** found in the supplied firmware branch.

The relevant source is `firmware/firmware_tool`, whose README states that bench diagnostics connect through the **STM32Link / ST-Link Virtual COM port** and that the target must run the `hw_test` firmware variant.

The path is:

```text
Windows laptop
      |
      | USB
      v
ST-Link pod
      |
      | Virtual COM Port
      v
STM32 USART2 debug/R&D UART
      |
      | ProtoComms framing + protobuf
      v
STM32 firmware
```

### Important JTAG/SWD terminology

The ST-Link pod is physically attached through the STM32 debug connector and can also provide SWD/JTAG debugging and SWO/ITM trace.

However, the **HW diagnostics command/data method in the supplied source does not transfer protobuf accelerometer data over JTAG/SWD**.

It uses the ST-Link's USB **Virtual COM Port**, which is wired to the STM32 debug UART.

This is useful because it gives a direct laptop-to-STM32 connection with no Toradex SOM, broker, SSH tunnel or network path.

### Firmware requirement

The supplied firmware tool explicitly requires the:

```text
HW_TEST
```

firmware variant, which defines:

```text
DEBUG_RND_UART
```

In this mode the debug UART is enabled for ProtoComms and the normal SOM communications path is disabled.

A normal production firmware image may therefore not respond on the ST-Link Virtual COM port.

### Serial settings

The supplied diagnostics tool uses:

```text
115200 baud
8 data bits
no parity
1 stop bit
```

The GUI defaults to the same settings.

### Connection validation

When `Connect ST-Link VCP` is pressed, the application first sends the same firmware-version request used by the colleague diagnostics tool.

A successful version response confirms that:

- the selected COM port is correct;
- the ST-Link VCP is open;
- ProtoComms framing is working;
- protobuf decoding is working;
- the STM32 is running a diagnostics-compatible firmware path.

### Framing

The direct serial framing matches `fw_tool_connection_manager.py`:

```text
[4-byte little-endian payload length]
[protobuf payload]
[4-byte CRC-32/MPEG-2]
```

The CRC covers:

```text
length field + protobuf payload
```

The STM32 sends an immediate `mResp` acknowledgement and then the requested application response. The application discards the ACK and waits for the requested message, matching the colleague tool.

### Accelerometer requests

The existing firmware interface is reused:

```text
mGetAccelerometerNumBuffers
    -> mAccelerometerNumBuffers

mGetAccelerometerDataMsg
    -> mAccelerometerDataMsg
```

The returned accelerometer data are packed triples:

```text
int16 X
int16 Y
int16 Z
```

or six bytes per sample.

The application converts these counts to g using the configurable `Scale mg/LSB` value.

### Live display

The live tab displays:

- scrolling X/Y/Z acceleration;
- current Gordon-band result for the selected driven axis;
- selectable RMS acceleration / velocity / displacement units;
- live tilt/gravity-vector view;
- total captured samples and buffers;
- firmware version returned at connection.

Captured data can be:

- saved to CSV;
- cleared;
- loaded directly into the offline `Reader` dataset slot.

### Timing limitation

The current returned accelerometer buffer does not include an STM timestamp or sample counter.

The live application therefore reconstructs sample time from:

```text
sample_index / configured_sample_rate
```

For formal synchronisation against an independent calibrated reference DAQ, adding an STM sample counter or timestamp remains preferable.

### Live diagnostic and threshold views

Separate live tabs expose the decoded raw X/Y/Z samples, the selected-axis
processing stages, FFT magnitude and phase, single-block and Welch-averaged
PSD, and a zeroable two-axis tilt target.

The live vibration threshold is evaluated from the integrated Gordon
one-third-octave band results for all three axes. It can be configured as:

- Gordon Office criterion multiplied by a user-entered factor;
- fixed RMS acceleration in g;
- fixed RMS velocity in µm/s;
- fixed RMS displacement in µm; or
- disabled.

Fixed thresholds are converted into the currently selected display units at
each band centre. The overview overlays the threshold on the active live band
plot. The `Live threshold` tab shows all 42 axis-band comparisons, highlights
exceeded points in red, and reports the worst axis, band, and ratio to limit.

These are band-integrated vibration limits. They are deliberately not drawn as
horizontal limits on the raw time trace or individual FFT/PSD bins, because
those quantities are not directly equivalent to one-third-octave RMS limits.

## Current bench acquisition path — `accelerometer-streaming`

This section supersedes the older ProtoComms/SOM live-acquisition descriptions
elsewhere in this historical README.

The prepared unit runs the `hw_test_vmm` image from Git tag
`accelerometer-streaming`. Do not reflash it for this application. Live data use
only:

```text
LIS2DUX12
  → hw_test_vmm sample block
  → STM32 USART2
  → ST-LINK Virtual COM Port at 460800 baud
  → COBS + IEEE CRC32 VMM protocol v1
  → vmm_stream.VmmStreamClient
```

ProtoComms, the System Diagnostics UART, protobuf, SOM HTTP, SWD and JTAG are
not part of this live-data path. The embedded client is copied unchanged from:

```text
tools/hw-diagnostics/hw_diagnostics/vmm_stream.py
```

The source-of-truth handoff and wire protocol are:

```text
tools/hw-diagnostics/ACCELEROMETER_STREAMING.md
vibration-test-ctr/PROTOCOL.md
```

The GUI opens the selected ST-LINK COM port, sends `START`, polls
`SAMPLE_BLOCK` packets, verifies COBS framing and CRC32, and sends `STOP` on
capture stop/close. Each block supplies sequence, timestamp, sample period,
ODR, full-scale range, sensor status and signed X/Y/Z counts. Sequence gaps and
invalid frames are reported in the live overview.


---

## 45. Accelerometer configuration controls

The Live STM Data tab now exposes the recommended LIS2DUX12 configuration and
host processing parameters.

The defaults are:

| Parameter | Default |
|---|---|
| Device | LIS2DUX12 |
| Full-scale range | ±2 g |
| Output data rate | 200 Hz |
| Operating mode | High Performance |
| Anti-alias filter | Enabled |
| Anti-alias bandwidth | ODR/2 = 100 Hz at 200 Hz ODR |
| Nominal sensitivity | approximately 0.061 mg/LSB at ±2 g |
| Data format | signed int16 X/Y/Z |
| Application pre-filter | None |
| Application processing rate | 200 samples/s/axis |
| FFT length | 512 |
| FFT window | Hann |
| FFT overlap | 50% |
| PSD averaging | 4 FFT blocks |
| Vibration bands | Gordon one-third-octave 4–80 Hz |

The nominal sensitivity display is calculated from the selected full-scale:

```text
sensitivity_mg_per_lsb = full_scale_g / 32768 * 1000
```

For ±2 g this is approximately:

```text
0.061035 mg/LSB
```

### Host processing vs sensor programming

`Apply processing settings` updates the host-side processing controls
immediately:

```text
sample rate
FFT length
FFT overlap
PSD averaging
```

The live stream still uses the ODR and full-scale reported by the STM sample
block for scaling and timing.

### Runtime firmware configuration

The `accelerometer-streaming-fw-config` firmware implements VMM `CONFIGURE`
(`0x03`). The `Program accelerometer` button sends ODR, full scale, HP/LP mode,
bandwidth, FIFO watermark, INT1 routing and default stream timeout while live
capture is running.

The requested configuration can also be exported as JSON.

### Read-back verification

Each VMM sample block already reports:

```text
odr_hz
fs_g
```

Those fields remain useful live metadata. In addition, the updated firmware
returns a CONFIG packet containing the effective register-read HP/LP, ODR,
full-scale, bandwidth, FIFO, INT1 and timeout settings. The GUI compares every
returned field with the controls and shows `VERIFIED` or `DIFFERS FROM REQUEST`.


---

## 46. Live tilt stability correction (v12)

The previous live-tilt implementation could establish its automatic zero
reference from the first partial averaging window. At 200 Hz with a short
0.25-second average, that meant the zero could effectively be based on only a
small number of initial samples. As the averaging window filled, the gravity
estimate could move substantially and appear as a large false change in roll
or pitch.

Version 12 changes this behaviour.

### Dedicated gravity averaging

Live tilt now defaults to:

```text
2.0 seconds
```

of X/Y/Z data.

At 200 Hz this is:

```text
400 samples
```

The application does not calculate or auto-zero tilt until the complete
averaging window is available.

### Gravity-vector method

The application first calculates the mean gravity vector:

```text
G = [mean(X), mean(Y), mean(Z)]
```

and its magnitude:

```text
|G| = sqrt(Gx^2 + Gy^2 + Gz^2)
```

The mean vector is normalised before calculating roll and pitch direction.

### Dynamic-vibration validity check

Tilt is intended to come from the quasi-static gravity vector, not from the
vibration component.

The application therefore also calculates the RMS dynamic acceleration about
the mean gravity vector.

The default validity conditions are:

```text
0.80 g <= mean gravity magnitude <= 1.20 g
dynamic RMS <= 0.05 g
```

If these conditions are not satisfied, the tilt result is marked invalid and
the application will not establish a new automatic zero reference.

The dynamic RMS limit is exposed in the Tilt controls within the Direct STM32 connection section on the Live STM data tab.

### Zero reference

The automatic zero is now captured only after:

1. a complete tilt averaging interval exists; and
2. the gravity estimate passes the stability checks.

The manual `Zero tilt at current position` button uses the same rules.

This is closer to the intended commissioning concept: tilt is based on a
stable averaged gravity vector, while the faster vibration signal is processed
separately through the PSD/Gordon-band path.


---

## 47. Tag-accurate VMM stream validation (v13)

Version 13 has been aligned directly to the user-supplied `accelerometer-streaming`
tag. The embedded `vmm_stream.py` is copied from:

```text
tools/hw-diagnostics/hw_diagnostics/vmm_stream.py
```

The active path remains only:

```text
ST-LINK VCP -> 460800 baud -> VMM v1 COBS + IEEE CRC32 -> SampleBlock.xyz
```

No ProtoComms, SOM HTTP, broker, JTAG memory reads or SWD reads are used.

### Important firmware finding

The tagged `VmmAcq_Configure()` selects the **Low Power** ODR enum at 200 Hz:

```c
md.odr = LIS2DUX12_200Hz_LP;
```

and when `antialias_enable` is true it currently selects:

```c
md.bw = LIS2DUX12_ODR_div_4;
```

The intended measurement configuration discussed for the vibration algorithm is
200 Hz High Performance with ODR/2 bandwidth. The running tagged image therefore
does not implement those intended mode/bandwidth settings.

The ST FIFO decoder also defines the low-power `XL_ONLY_2X_TAG` entry as two
8-bit XYZ samples packed into one FIFO entry. The tagged VMM acquisition code
copies only `raw.xl[0]` into each transmitted VMM sample. Consequently, if the
low-power FIFO tag is active, the transmitted raw stream has an 8-bit-left-
justified signature (values predominantly multiples of 256) and the second
sample in each 2X FIFO entry is not transmitted.

Version 13 detects that signature explicitly and marks the stream unsuitable
for tilt/spectral interpretation rather than presenting misleading results.

### VMM status decoding

The tagged acquisition source currently sets:

```text
0x02  block timing gap
0x04  raw sample reached +32767 or -32768
```

`vmm_types.h` documents bit 0 as an overrun bit, although the tagged acquisition
implementation does not currently set it.

The GUI now decodes these bits rather than displaying only a hexadecimal value.

### Raw-count diagnostics

A `Live raw counts` tab displays the exact signed values from `SampleBlock.xyz`
before conversion to g, including mean, standard deviation, min/max, RMS,
int16-rail hits and the fraction of samples whose low byte is zero.

For a stationary level ±2 g sensor, a sensible stream is approximately one
axis around ±16384 counts and the other two near zero, with gravity-vector
magnitude close to 1 g.

If almost every raw value is a multiple of 256, the GUI identifies the
low-power 8-bit-left-justified FIFO signature. Tilt will not auto-zero from such
a stream.

### Configuration button

With `accelerometer-streaming-fw-config`, the button is enabled while live
capture is running. An OK ACK means the complete stop/flush/program/read-back/
restart transaction succeeded; ACK detail carries the firmware error code on
failure. The subsequent CONFIG packet is the read-back proof shown by the GUI.

---

## 48. Bounded live memory and raw logging

The live display retains only the most recent **120 seconds** of converted
samples by default. The retention value is editable on the Live STM Data tab;
old display samples are removed from all plotted arrays as new blocks arrive.
This bounds the plotting and analysis workload during a long session.

`Start raw-count log` creates an independent CSV recording directly from each
decoded `SampleBlock`, before conversion to g and independently of display
retention. It records block/sample timing and metadata plus only the original
signed `x_counts`, `y_counts` and `z_counts`. Stop the logger with `Stop
raw-count log`; closing the application also flushes and closes an active log.

---

## 49. Live rendering performance

Live reception remains at the firmware-reported sample rate, but Matplotlib is
no longer asked to rebuild every live figure for every GUI polling pass. Only
the currently visible live tab is rendered:

- Live summary and raw traces: up to 2 redraws/second
- FFT, stages, raw-count diagnostics and threshold plots: up to 1 redraw/second
- Hidden and offline-analysis tabs: no live redraw work

Incoming sample blocks are still appended to the bounded display buffer and an
active raw-count logger still writes every sample. Queue processing is capped
per Tk callback so a temporary backlog cannot prevent buttons, tab changes and
window events from being handled.


## v14: CSV compatibility and continuous live capture

The comparison loader now accepts both of the following inputs directly:

1. Converted acceleration CSVs containing `time_s`, `X_g`, `Y_g`, `Z_g` (existing aliases remain supported).
2. Raw VMM count logs containing `x_counts`, `y_counts`, `z_counts` plus VMM metadata such as `sample_timestamp_us`, `odr_hz`, `fs_g`, and `status`.

Raw VMM counts are normalised internally using:

`g = counts * fs_g / 32768.0`

When available, sample time is reconstructed from `sample_timestamp_us` relative to the first sample. The loader also reports timestamp gaps, non-zero VMM status blocks, changing full-scale, changing ODR, and disagreement between timestamp-derived rate and reported ODR.

### Live capture timeout behaviour

The user-facing capture timeout is now separate from the finite firmware VMM START timeout. The default operator timeout is 5 minutes with action `Ask`. The available actions are:

- Ask
- Continue same duration
- Continue indefinitely
- Stop

`Enable capture timeout` can be cleared for an indefinite logical capture. The low-level VMM stream is still renewed automatically before its finite START lease expires, so an indefinite capture does not depend on an undocumented infinite timeout value.

The initial START clears stale VCP input bytes. Automatic renewals use a dedicated renewal command path that does not reset the serial input buffer or clear the logical capture. Capture elapsed time, sequence diagnostics, plotted history, and raw logging are preserved across renewals.

The application also watches the age of the last valid `SampleBlock`. If the stream stalls while the capture remains logically Running, it attempts controlled START renewal recovery and reports the attempt rather than silently leaving a frozen plot.

`Display retention` controls only the converted samples retained in memory for plotting. It does not set the capture duration and does not limit the raw-count CSV. Raw logging is buffered and flushed periodically rather than on every VMM block, then flushed/closed cleanly when capture stops.

Current live transport remains ST-LINK Virtual COM at 460800 baud using the VMM COBS + IEEE CRC32 stream. No ProtoComms/SOM HTTP/JTAG/SWD acquisition path is used by the live VMM monitor.

## Live multi-frequency K correction (v17)

The Live STM workflow can optionally apply installed-system translation coefficients to the live Gordon-band results.

- Use **Load K-factor CSV(s)** in the **Direct STM32 connection** section.
- A complete correction requires 14 Gordon-band `K = Base / Reader` coefficients for each of X, Y and Z (42 coefficients total).
- The loader accepts one combined CSV or multiple axis-specific CSVs. The normal exported correction format (`driven_axis`, `fc_hz`, `K_base_over_reader`) is supported.
- Enable **Apply live multi-frequency K correction** to use the translated values in the Live STM summary Gordon plot and the live threshold decision.
- The **Live K correction** tab shows raw reader band RMS values and the corrected values side-by-side for X, Y and Z, regardless of whether the live toggle is currently ON.
- The K correction is multiplicative and frequency-band-specific. It is applied after Gordon-band RMS integration. It is not a DC offset and it does not modify the raw time-domain acceleration stream or the tilt calculation.

## v18 multi-axis baseline / reader correction workflow

The correction workflow now uses one three-axis Reader CSV together with up to three baseline/reference CSVs, one for each intentionally excited axis.

- `Load reader CSV` loads one file containing X, Y and Z measured acceleration.
- `Load baseline CSV(s)` accepts multiple files at once. The application infers X/Y/Z from each filename using common axis naming patterns such as `X`, `_X_`, `X-axis`, `axis_Y`, or a trailing axis letter. If a filename is ambiguous, the GUI asks the operator to assign X, Y or Z.
- The normal mapping is X correction <- X baseline, Y correction <- Y baseline, Z correction <- Z baseline.
- The left-hand `Correction baseline mapping` selectors allow a missing baseline to explicitly reuse another loaded baseline axis. When, for example, the X baseline is missing and Y is selected as the surrogate, the Y-driven acceleration channel from the Y baseline file is used as the reference profile for the X reader correction. Cross-axis response from the surrogate baseline file is not used accidentally.
- `Export derived correction` exports one combined CSV containing 42 frequency-dependent coefficients: 14 Gordon bands for each of X, Y and Z. The file includes `driven_axis`, `baseline_source_axis`, `fallback_used`, source filenames, `fc_hz`, the model recommendation, the single-K candidate, and the exported band-specific `K_base_over_reader` value.
- The combined export is directly compatible with the live multi-frequency K-correction loader.

## v19 frequency-domain baseline CSV support

The multi-axis baseline workflow now accepts two baseline source types:

1. Time-domain accelerometer CSVs containing X/Y/Z acceleration channels.
2. Frequency-domain reference PSD CSVs containing `frequency_hz` and `psd_g2_per_hz`.

The frequency-domain format is intended for laboratory baseline spectra such as the X/Y lateral and Z reference files prepared from the vibration baseline workbooks. For these files the application integrates the acceleration PSD over each Gordon one-third-octave band to obtain the baseline band RMS acceleration. It then compares that band RMS with the corresponding reader-axis band RMS derived from the uploaded three-axis Reader CSV and exports `K_base_over_reader` for all 14 bands on X, Y and Z.

Because a PSD baseline has no synchronous time history, it cannot populate the paired Raw comparison, Selected stage, H1 transfer-function, or time-domain tilt comparison views. It is, however, directly valid for `Export derived correction`, which is the intended workflow for these reference PSD files.

The exported correction CSV records `baseline_source_format=baseline_psd` when a frequency-domain baseline was used.


## NumPy compatibility

PSD baseline integration uses `numpy.trapezoid` on NumPy 2.x with a fallback to `numpy.trapz` on older NumPy releases.


## v22 multi-axis offline analysis UI

The global **Driven excitation axis** and global **Displayed/inspection axis** controls have been removed from the left panel. The left panel now contains only the correction baseline mapping and common processing controls.

- **Raw comparison** shows X, Y and Z using each target axis's mapped baseline.
- **PSD comparison** overlays X/Y/Z; dashed lines are baseline/reference and solid lines are reader.
- **Gordon bands** overlays X/Y/Z plus the Gordon Office criterion.
- **Translation function** shows all three axis-specific K curves and before/after band-RMS views using each axis's mapped baseline.
- **Selected stage** contains its own local X/Y/Z selector and FFT-block selector.
- **Transfer function** contains its own local X/Y/Z selector because H1/coherence remain a same-axis paired diagnostic.
- **Live stages** has a local axis selector; **Live FFT + PSD** uses the same local selection only for the phase panel while magnitude/PSD continue to show X/Y/Z.

This supersedes older README text that describes a single global driven-axis selector.

## Live shock detector (v23)

The **Live STM data** summary includes a shock detector next to the two-axis tilt target. It follows the method in `1174-Y-056 Proposed Issue 2 Accelerometer Signal Processing`, section 6.2.2:

- maintain a 50-sample rolling X/Y/Z steady-state average (0.25 s at the nominal 200 Hz ODR);
- calculate the instantaneous changes from steady state;
- calculate the resultant shock magnitude `S = sqrt((X-Xavg)^2 + (Y-Yavg)^2 + (Z-Zavg)^2)`;
- compare `S` with a user-configurable threshold in g.

The Issue 2 draft defines the calculation and requires a configurable shock threshold, but it does **not** define the numerical threshold value. The GUI therefore exposes the threshold explicitly; the current 1.0 g value is an engineering UI default and must not be treated as a released product limit.

The summary plots the recent resultant shock magnitude, the configured threshold, current value, window peak, capture peak and threshold-event count. **Reset peak/events** clears the accumulated engineering diagnostics. The calculation uses the raw converted X/Y/Z acceleration and is deliberately unaffected by the multi-frequency vibration K correction.

Because the live sensor is normally configured to ±2 g per axis, impacts that saturate an axis cannot be quantified accurately beyond the sensor range; the existing VMM rail/status diagnostics remain relevant.

## v24 - raw-log and correction traceability

Before `Start raw-count log` starts recording, the application asks for Reader ID, excited axis (X/Y/Z), requested vibration level, run number and optional notes. These values are used to propose a descriptive CSV filename and are written to a companion `*_settings.json` file together with the capture, accelerometer and processing settings in use at the start of the log.

`Export derived correction` now also writes a companion `*_settings.json` file. This records the reader source file, baseline source files, baseline-axis mapping, PSD processing settings and the exported K-factor rows so the configuration used to generate the correction can be traced later.


## v26 horizontal sidebar collapse

The complete offline controls area containing Datasets, Correction baseline mapping and Common analysis controls can now be collapsed horizontally using the narrow arrow button at the left edge of the main graph area. The individual sections still retain their existing up/down collapsible controls. This allows the graph tabs to use nearly the full application width when setup controls are not needed.


## v30 shock plot cleanup

The Live STM Data shock plot no longer draws the PASS/current/window peak/capture peak/events annotation box inside the graph. The plot now contains only the shock magnitude trace, configured threshold, axes and legend. Detailed shock status remains available in the live controls sidebar.

## v32 - offline shock analysis

A new **Shock analysis** tab is placed immediately after **Raw comparison**. It operates on the loaded Reader CSV and does not require a baseline file. The tab applies the Issue-2 time-domain shock method using a configurable rolling steady-state window (default 50 samples) and displays the X, Y and Z acceleration changes from that rolling vector, plus the resultant shock magnitude S and configurable detection threshold. Contiguous threshold exceedances are grouped as individual events and highlighted on the plots.

## v33 - grouped shock events and X/Y/Z shock baseline comparison

- Shock event counting now uses a configurable release time: once the resultant crosses the threshold, the event stays open until the signal has remained below threshold continuously for the release period. This prevents ring-down from one impact being counted as several shocks.
- Added dedicated `Load shock baseline CSV(s)` and `Load shock reader CSV(s)` controls on the Shock analysis tab.
- Shock filenames are expected to contain `shock` and the applied X/Y/Z axis; ambiguous names prompt for the axis.
- Stores separate X/Y/Z baseline and reader shock captures, while retaining all three measured channels in every file.
- Selecting Applied shock axis chooses the X-, Y- or Z-applied test to review.
- When baseline and reader shock captures are both loaded for the selected axis, their Delta-X/Delta-Y/Delta-Z and resultant traces are overlaid and aligned to their respective peak shock for comparison.
- Summary reports reader peak, dominant measured axis, grouped physical event count, baseline peak and Reader/Baseline peak ratio.
