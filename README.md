# FLIR ATS → temperature TIFF

A batch converter that turns FLIR `.ats` recordings (from cameras like the
X6900sc, A6750sc, etc.) into **multi-page float32 TIFF stacks where each
pixel is a calibrated temperature**.

The TIFFs open straight into Fiji / ImageJ as a 32-bit grayscale stack
(hover the cursor over a pixel and the status bar shows the temperature
directly) and load into Python as a `(n_frames, H, W) float32` ndarray
via `tifffile.imread`.

Per input `Rec-NNNNNN.ats` the tool writes three files into your output
folder:

| File | Content |
|---|---|
| `Rec-NNNNNN_eps0.92_temp_C.tif` (or `_temp_K.tif`) | Multi-page BigTIFF, every page is one camera frame stored as `float32` Celsius (default) or Kelvin. The `_epsX.XX_` slot records the emissivity actually used to produce the temperature values (two-decimal-place fixed format; per-file recorded value, or the override if one was applied). Bit-for-bit the SDK's `Unit.TEMPERATURE_FACTORY` output (optionally shifted by 273.15). zlib level 5. |
| `Rec-NNNNNN_eps0.92_meta.json` | Camera (model, serial, lens, filter), recording (frame rate, integration time, temperature range), object parameters (emissivity, distance, reflected / atmospheric temperature, humidity), and the crop + flip applied to every page. |
| `Rec-NNNNNN_eps0.92_preview.png` | One 8-bit grayscale preview frame, selected as the frame with the highest mean temperature across the whole recording (= the overall hottest moment). The frame index is recorded in the JSON. |

### Why float32 temperature instead of raw counts?

Because temperature is what you actually want to look at: hover any pixel
in Fiji and the status bar shows the value directly, no Planck inversion
required. Storing the SDK's float32 output preserves it bit-for-bit, so
the temperature you read back from the TIFF is exactly the temperature
the SDK computes from the source `.ats` (plus the optional 273.15
shift). The raw 14-bit ADC counts remain recoverable at any time by
reopening the source `.ats` with `f.unit = fnv.Unit.COUNTS`; this tool
never modifies the source files.

## Quick start

```powershell
# 0. one-off setup (per machine)
msiexec /i installers\FLIRScienceFileSDK-2026.1.2+10-Windows-x64.msi
pip install "<wheel matching your python>"   # see "Install" below
pip install numpy tifffile pillow

# 1. run interactively (recommended)
python flir_ats_batch.py
#  -> answer the prompts:
#       test/batch  -> batch
#       input dir   -> D:\Recordings\modulated
#       output dir  -> E:\converted\modulated
#       file select -> Enter (= all) or '1-10'
#       overrides   -> Enter (= use recorded object_parameters)
#       crop        -> Enter (= no crop) or '100-540 50-460'
#       flip        -> Enter (= rot180, the X6900sc default)

# 2. read the output in Fiji or Python (see "How to read the output")
```

Every prompt remembers your answer; the next run can be driven entirely
by pressing Enter.  See **Modes**, **Crop and flip**, and **Remembered
defaults** below.  See **Temperature calculation and emissivity** for
the physics and the per-step error budget.

## What you need to download

1. **This repository.** Clone or download as a ZIP.
2. **FLIR Science File SDK** — the Windows MSI is bundled in
   [`installers/`](installers/) for convenience
   (`FLIRScienceFileSDK-2026.1.2+10-Windows-x64.msi`). If you would
   rather get the latest version directly from FLIR, you can register a
   free account at <https://flir.custhelp.com/app/account/fl_login> and
   download from the Software downloads section.
3. **Python 3.10, 3.11, 3.12, 3.13, or 3.14** for Windows x64 — any
   recent CPython will do.
4. **Python packages** — `numpy`, `tifffile`, `pillow` (see
   `pip install` below).

## Install

1. Run the MSI from `installers/`. By default it places the SDK at
   `C:\Users\<you>\AppData\Local\Programs\FLIR Systems\sdks\file\` (on a
   per-user install) or
   `C:\Program Files\FLIR Systems\sdks\file\` (on a system-wide install).

2. From an elevated terminal, install the Python wheel that matches
   your Python version. The MSI drops the wheels at
   `<install dir>\python\dist\`. For Python 3.12 on a per-user install:

   ```powershell
   pip install "C:\Users\<you>\AppData\Local\Programs\FLIR Systems\sdks\file\python\dist\FileSDK-2026.1.2-cp312-cp312-win_amd64.whl"
   ```

   Wheels for cp310, cp311, cp312, cp313, cp314 are all in that folder —
   pick the one whose `cpXY` matches your Python (`python --version`).

3. Install the rest of the Python dependencies:

   ```powershell
   pip install numpy tifffile pillow matplotlib
   ```

   `matplotlib` is a soft dependency: only used to draw the post-sweep
   summary PNGs in test mode.  The conversion itself works without it
   (the script just prints a one-line note and skips the plots).

4. Verify the SDK loads:

   ```powershell
   python -c "import fnv, fnv.file; print('FLIR SDK ready:', fnv.__file__)"
   ```

## Run

Interactive (recommended for first use):

```powershell
python flir_ats_batch.py
```

It will prompt:

```
Input folder (contains .ats files):  D:\Recordings\modulated
Output folder for .tif + .json + .png:  E:\converted\modulated
```

Non-interactive:

```powershell
python flir_ats_batch.py --input D:\Recordings\modulated --output E:\converted\modulated
```

Useful flags:

```
--unit kelvin       write Kelvin into the TIFF instead of the default Celsius
--mode {test,batch} skip the start-up mode prompt and go straight to that mode
--no-recurse        only scan the top level of --input (default scans subfolders)
--overwrite         re-convert files whose outputs already exist
--no-confirm        skip both interactive prompts (file selection AND radiometric
                    parameter override) and process every file with the values
                    recorded inside each .ats verbatim (implies batch mode)
--files "1-10"      non-interactive subset selection (same syntax as the
                    interactive prompt; implies batch mode)
```

### Modes

On start-up the script asks whether you want:

```
What would you like to do?
  [1] Test mode  -- sweep emissivity values on ONE .ats file
  [2] Batch mode -- convert MANY .ats files with shared parameters
Choice [1/2]:
```

**Test mode** is for parameter-sensitivity studies. After the usual
input/output folder prompts it lists the `.ats` files, you pick exactly
one, and then type the emissivity values you want to try:

```
Emissivity values to test
  range form:  [start, end, step]   e.g.  [0.3, 0.9, 0.1]
  list form:   0.3 0.5 0.7   or   0.3,0.5,0.7
Emissivities: [0.3, 0.9, 0.1]
  -> 7 value(s): 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900
Proceed? [y/N/edit]:
```

Test mode is deliberately frugal:

1. It does ONE fast pass over the recording (`Unit.COUNTS`, uint16)
   to find the frame with the highest mean ADC count -- that's the
   hottest frame.  The same scan feeds the crop preview PNG so the
   file is never scanned twice.
2. That single frame is decoded once in `Unit.TEMPERATURE_FACTORY`
   using the emissivity baked into the recording.
3. **Test mode now sweeps two parameters at once**: emissivity (`ε`)
   AND the effective wavelength (`λ_eff`) used by the exact-Planck
   post-correction.  Each prompt accepts the same `[start, end, step]`
   or list syntax.  With `n_ε` emissivities and `n_λ` lambdas, the
   sweep produces `n_λ × n_ε` pages — outer loop is `λ`, inner loop is
   `ε`, so scrolling the TIFF in Fiji walks all `ε` at the first `λ`,
   then jumps to the next `λ`, etc.
4. Each page is exact-Planck-post-corrected at its `(λ_eff, ε)` pair
   (or SDK-native-decoded when the SDK actually honours live ε edits,
   in which case `λ_eff` collapses to a single value with a console
   note), cropped + flipped according to your settings, and a 28-row
   label strip is **prepended at the top** of the page (white text
   `lambda=X.XX um  eps=Y.YYY` on a dark band — or just
   `emissivity = X.XXX` when only `ε` was swept).  The label sits
   above the picture so it never occludes any pixel of the real
   scene.
5. After the sweep finishes the script writes **summary PNGs** next to
   the TIFF (requires `matplotlib`; install with `pip install
   matplotlib`):
     - 1-D sweep (only `ε` *or* only `λ` varied):
       `*_eps_sweep_plot_vs_{eps,lambda}.png` — single line chart with
       `T_min` and `T_max` curves vs the varied parameter.
     - 2-D sweep (both varied):
       * `*_eps_sweep_plot_by_eps_stacked.png` — vertically stacked
         subplots, one per `λ`, x = `ε`, two curves per subplot
         (`T_min`, `T_max`).
       * `*_eps_sweep_plot_by_lambda_stacked.png` — same idea,
         transposed: one subplot per `ε`, x = `λ`.
       * `*_eps_sweep_plot_heatmap_t_max.png` and
         `*_eps_sweep_plot_heatmap_t_min.png` — 2-D pcolormesh of
         `T_max` and `T_min` with `λ` on the y-axis and `ε` on the
         x-axis.  The heatmap is the fastest way to see where the
         "interesting" region of the parameter space lives.
   If matplotlib is missing the rest of the sweep still runs; only
   the plot step is skipped.

#### About the emissivity correction

Test mode picks the best method per-file at startup:

- **`sdk_native` path (preferred).**  When the file's
  `can_change_object_parameters` is True, the script sets
  `f.object_parameters.emissivity` to each requested test value,
  re-reads the hottest frame in `Unit.TEMPERATURE_FACTORY`, and lets
  the FLIR SDK perform its **full band-integrated radiometric
  inversion** using the camera's factory spectral response curve.
  No single-wavelength approximation, no `λ_eff` to choose, no
  reflected-radiance term dropped — gold standard.

- **`exact_planck_single_wavelength` fallback.**  When the SDK
  reports `can_change_object_parameters = False` (so live ε edits
  are silently ignored), the script reads the hottest frame once at
  the recorded ε and post-corrects it via the closed-form Planck
  inversion at a single effective wavelength:

  ```
                         C2
  T_new = ───────────────────────────────────────────────────────
          λ_eff · ln(1 + (ε_new/ε_assumed) · (exp(C2/(λ_eff·T_assumed)) − 1))
  ```

  with `C2 = 14388 µm·K`, `λ_eff = 3.5 µm` (centre of the X6900sc's
  2-5 µm MWIR pass-band), and the file's recorded emissivity as
  `ε_assumed`.  Reflected-radiance term omitted (valid when scene
  `T` ≫ reflected-environment `T`).  No Wien high-T approximation, so
  the result stays well-defined above 1500 °C, where Wien would
  overestimate `T_new` by tens of percent.

Each `*_eps_sweep_meta.json` records `emissivity_correction.method`
plus the relevant parameters (`λ_eff`, `C2`, `ε_assumed` for the
Planck path; `eps_recorded_in_ats` for the SDK-native path), so the
output is self-describing and any output is reproducible without the
SDK.

The startup line `(SDK can_change_object_parameters = True/False)`
tells you which path test mode took on the current file.

**See *Temperature calculation and emissivity* below for the
derivation and an honest error budget.**

```
Rec-000548_eps_sweep_temp_C.tif    one file, one page per emissivity
Rec-000548_eps_sweep_meta.json     hottest frame index + per-page stats
```

Typical sweep takes a few seconds and produces a TIFF in the single-
digit-MB range even for ten emissivity values.

After the sweep finishes you're asked whether to run another sweep
(pick another file and / or other emissivity values), hand off to
batch mode, or quit.

**Batch mode** is the original whole-folder converter that uses one
shared set of parameters for every file. Batch mode never loops back —
when it finishes the program exits.

### Crop and flip (applied to every output)

After the file-selection (batch) or after picking the test file +
emissivity values (test), the script asks you to define a crop region
and an output flip.  Both options are shared across every file
processed in the same session and are remembered for the next run.

**Crop is described in PRE-FLIP coordinates** so the numbers match
what you see in a viewer (Fiji, the Windows photo app, etc.) opened
directly on the source frame.  The grammar is `x_min-x_max y_min-y_max`
in 1-based inclusive pixel indices; `x` is the width axis, `y` is the
height axis.  Press Enter / type `none` / `full` to skip the crop.

```
Crop region (pre-flip coordinates)
  image size: 640 x 512  (W x H)
  syntax:   x_min-x_max y_min-y_max  (1-based inclusive)
  example:  100-540 50-460
  press Enter (default below), or type 'none' / 'full' to skip crop
Crop  [Enter = last: 100-540 50-460]: 100-540 50-460
  -> crop x:100-540 y:50-460  (440 x 410 px)
  preview PNG written: E:\converted\modulated\_crop_preview.png
    open it now -- the red rectangle shows what will be kept
Proceed? [y=confirm / r=reset / c=cancel crop]: y
```

The preview PNG (`_crop_preview.png`, shared, overwritten on each
attempt) is built from the hottest frame of the first file in the
batch (or the file you picked for the test sweep) with a red rectangle
drawn around the proposed crop region.  Open it in Explorer to
verify, then `y` to confirm, `r` to enter a different range, or `c` to
disable the crop entirely.

**Flip is applied AFTER the crop**:

```
Flip the output image?
  [1] none     -- no flip
  [2] hflip    -- horizontal flip (left <-> right)
  [3] vflip    -- vertical flip (top <-> bottom)
  [4] rot180   -- 180-degree rotation (= hflip + vflip)
Choice [1/2/3/4]  [Enter = 4=rot180]:
```

180-degree rotation is the default because the X6900sc used on this
project produces upside-down + mirrored frames; the default makes the
TIFF land in the natural orientation without any further user action.

The applied crop (in 0-based half-open `[x0:x1, y0:y1]` form) and flip
are recorded in every `*_meta.json` so the run is reproducible.

### Remembered defaults (press Enter to re-use last run)

Every interactive choice you make is written to
`~/.flir_ats_batch_state.json` (on Windows that is
`C:\Users\<you>\.flir_ats_batch_state.json`). On the next launch the
script prints the remembered values and offers them as the
press-Enter default at each prompt:

```
[state] remembered defaults from C:\Users\me\.flir_ats_batch_state.json
          mode             = test
          input_dir        = D:\Recordings\modulated
          output_dir       = E:\converted\modulated
          file_selection   = 1-5 10
          test_file_name   = Rec-000548.ats
          emissivity_spec  = [0.1, 0.95, 0.05]
          lambda_eff_spec  = [2.5, 4.5, 0.5]
          crop_spec        = 100-540 50-460
          flip_mode        = rot180
          object_overrides = {'emissivity': 0.85}
```

Anything stored is fully overridable: just type a new value at the
prompt instead of pressing Enter, and the new value replaces the old
default in the JSON. There is no separate "reset" command — delete
`~/.flir_ats_batch_state.json` if you want to start fresh.

### File-selection prompt (default)

After finding the `.ats` files under `--input` the script lists them
with 1-based indices and prompts you to pick which ones to process:

```
Found 31 .ats files under D:\Recordings\modulated:
   1  Rec-000548.ats                            ( 3.49 GB)
   2  Rec-000549.ats                            ( 3.52 GB)
   3  Rec-000550.ats                            ( 3.45 GB)
  ...
  31  Rec-000578.ats                            ( 3.49 GB)

Which files do you want to convert?
  press Enter (or 'all') -> every file
  '1-10'                 -> a range (inclusive)
  '1 3 5' or '1,3,5'     -> individual files
  '1-5 10 15-20'         -> mix of ranges and individuals
Selection:
```

The chosen subset is summarised back, you confirm with `y`, and the
batch begins. To skip the prompt entirely pass `--files "1-10"` (or
`--files all`) on the command line.

### Radiometric parameter inspection (default)

Before processing any file, the script prints the radiometric inversion
parameters recorded in the first `.ats` and lets you override any of them:

```
  Radiometric inversion parameters (recorded inside the first .ats):
  name                            value  unit         description
  ----------------------------    -----  ------------ --------------------
  emissivity                     0.9200  0-1 (-)      surface emissivity
  reflected_temp               293.1500  Kelvin       reflected (background) temperature
  distance                       1.0000  metres       target distance
  atmosphere_temp              293.1500  Kelvin       atmospheric temperature
  relative_humidity              0.3000  0-1 (-)      relative humidity
  atmospheric_transmission       0.9735  0-1 (-)      atmospheric transmission
  ext_optics_temp              293.1500  Kelvin       external optics temperature
  ext_optics_transmission        1.0000  0-1 (-)      external optics transmission

  Do you want to override any of them? [y/N]:
```

Press Enter at any per-parameter prompt to keep its value. Any
overrides you confirm are applied to **every** file in the batch
before the FLIR SDK computes temperatures. The recorded values and
the applied overrides are both written into each `*_meta.json` so the
run is fully reproducible.

The source `.ats` files are never modified by this tool. If you want
to abandon an overridden run and re-process with the original
parameters, delete the outputs (or pass `--overwrite`) and re-run with
`--no-confirm`.

The tool reports a per-file progress line and an end-of-batch summary:

```
[setup] FLIR SDK loaded: ...\fnv\__init__.py
[setup] input:  D:\Recordings\modulated
[setup] output: E:\converted\modulated
[setup] 12 .ats files to convert  (unit=celsius, skip existing)

=== [1/12] Rec-000548.ats ===
  camera = X6900sc  serial = 00139
  5311 frames @ 1000.0 fps  512x640  payload (float32) ~ 6.96 GB
  writing temperature in celsius
    531/5311  (200 fr/s, elapsed 0.0 min, ETA 0.4 min)
    ...
  -> ok
...
[done] 12 files in 4.1 min  (ok=12, skipped=0, error=0)
```

## What's inside the TIFF

Every page of the BigTIFF is one camera frame, stored as `float32`. Each
value is the temperature computed by the FLIR Science File SDK with the
camera's factory Planck calibration applied (`Unit.TEMPERATURE_FACTORY`).
If `--unit celsius` was used (the default), 273.15 was subtracted so the
values are Celsius; with `--unit kelvin` the values are Kelvin.

The temperature in the TIFF therefore depends on the object parameters
recorded inside the `.ats` (emissivity, reflected temperature, distance,
atmospheric transmission, humidity, external optics). Those parameters
are captured in the JSON sidecar so you always know which set of
assumptions produced the temperatures in the TIFF. If you want to
recompute temperatures under a different set of assumptions, reopen the
source `.ats` through the SDK with adjusted `object_parameters` (the SDK
exposes those as writeable attributes) and convert in
`Unit.TEMPERATURE_USER`.

## Temperature calculation and emissivity

### 1. The full radiometric model

Inside the `.ats` the camera stores a per-pixel 14-bit ADC count.  The
temperature written into the TIFF is the result of the FLIR Science
File SDK inverting the camera's factory radiometric model.  The model
the SDK uses (the standard form for cooled mid-wave IR cameras) is

```
I_measured = ε · τ_atm · τ_optics · B(T_obj)            ← own emission
           + (1 − ε) · τ_atm · τ_optics · B(T_refl)     ← reflected
           + (1 − τ_atm) · τ_optics · B(T_atm)          ← path radiance
           + (1 − τ_optics) · B(T_window)               ← external optics
```

where

| Symbol | Meaning | `object_parameters` key |
|---|---|---|
| `ε` | surface emissivity | `emissivity` |
| `τ_atm` | atmospheric transmission | `atmospheric_transmission` |
| `τ_optics` | external optics transmission | `ext_optics_transmission` |
| `T_refl` | reflected (background) temperature | `reflected_temp` |
| `T_atm` | atmospheric temperature | `atmosphere_temp` |
| `T_window` | external optics temperature | `ext_optics_temp` |
| `B(T)` | Planck spectral radiance integrated over the camera's spectral response | (built into the factory calibration) |

The SDK solves the equation for `T_obj` given the measured radiance
and every other parameter.  The "Radiometric parameter inspection"
prompt at the start of a batch run displays the recorded values and
lets you override any of them.

### 2. The factory Planck approximation

`B(T)` is the spectral radiance integrated over the camera's filter
band.  FLIR fits this with a closed-form expression (the so-called
"Thermimage" approximation),

```
I_camera = R1 / ( R2 · (exp(B / T) − F) ) − O
```

with calibration constants `R1`, `R2`, `B`, `F`, `O` baked into the
camera at the factory.  The SDK inverts this for `T`:

```
T = B / ln( R1 / ( R2 · (I + O) ) + F )
```

before adding the corrections for emissivity, reflected radiation, and
atmospheric / optics losses.  This tool never sees the raw constants
— the SDK applies them internally and returns calibrated Kelvin.

### 3. Emissivity (and how this tool re-targets it)

Emissivity multiplies the object's own thermal emission.  Two
consequences:

- **Lower assumed ε → higher reported T.**  For the same measured
  radiance, the camera attributes the shortfall to the surface being
  a less efficient emitter, so it must be hotter than initially assumed.
- **Emissivity is the dominant uncertainty for hot, near-black-body
  scenes.**  A 10 % change in ε produces a few percent shift in
  reported T (often tens of °C at 1000 °C).

The Science File SDK 2026.1.2 reports
`can_change_object_parameters = False` for X6900sc ATS files and
`Unit.TEMPERATURE_USER` raises *"failed to set unit"*; assigning to
`f.object_parameters.emissivity` looks like it succeeds but the SDK's
temperature output does not change.  To recover an emissivity-
sensitivity capability this tool **inverts the Planck radiation law
exactly at one effective wavelength**.

The single-wavelength Planck radiance is

```
B(T) = (constant) / ( exp( C2 / (λ_eff · T) ) − 1 )    with C2 = 14388 µm·K
```

Equating the measured radiance under two emissivity assumptions (and
dropping the reflected term, see Section 4),

```
ε_assumed · B(T_assumed) = ε_new · B(T_new)
```

rearranges to a closed-form inversion with no high-temperature
approximation:

```
                   C2
T_new = ────────────────────────────────────────────────────────
        λ_eff · ln( 1 + (ε_new/ε_assumed) · (exp(C2/(λ_eff·T_assumed)) − 1) )
```

`T` is in Kelvin throughout.  This tool ships `λ_eff = 3.5 µm`, the
centre of the X6900sc cold filter pass-band (2-5 µm, MWIR) with the
ND 2.0 filter installed; for other camera / filter combinations edit
the `DEFAULT_LAMBDA_EFF_UM` constant at the top of
`flir_ats_batch.py`.  For very hot scenes (T > 1500 °C) the band's
effective wavelength shifts towards the short end of the pass-band
(Wien displacement law), so `λ_eff = 2.8 – 3.0 µm` may give a slightly
better match.

The same equation is applied in **batch mode** whenever an emissivity
override is supplied for a locked file (so the TIFF's pixel values are
already corrected) and in **test mode** when the user sweeps a list of
emissivity values on the hottest frame.

#### Earlier versions used a Wien approximation; what changed

Prior to the *exact-Planck* commit this tool used the Wien
high-temperature approximation

```
1/T_new ≈ 1/T_assumed − (λ_eff / C2) · ln(ε_assumed / ε_new)
```

which drops the "− 1" in the Planck denominator.  The Wien form is
fine at low T or small Δε but overestimates `T_new` by tens of
percent in the high-T / large-Δε corner that high-temperature users
actually live in (see the table that used to be here, now removed:
the gap at e.g. `T_assumed = 1500 °C, ε: 0.92 → 0.1` was a 35 000 °C
overestimate).  The exact form costs one extra `exp` and `log1p` per
pixel per requested emissivity and removes that error class
completely; numbers written by the new code differ from older runs
by up to several hundred degrees Celsius in the hot / low-ε corner.
The `*_meta.json` records `"emissivity_correction.method"` so old
versus new outputs are distinguishable.

### 4. Uncertainty and known limitations

Every temperature in the output TIFF carries error from several
independent sources.  In rough order of importance for high-T (>700 °C)
scenes recorded on the X6900sc:

1. **Emissivity uncertainty (usually dominant).**  At T ≈ 1000 °C a
   ±10 % change in assumed ε produces about ±50 °C of bias in the
   reported T.  At 1500 °C the same ±10 % is roughly ±100 °C.  Use
   test mode to bracket your scene's actual ε before committing to a
   batch run.

2. **Camera factory calibration.**  The X6900sc datasheet specifies an
   absolute accuracy of ±1 °C or ±1 % of reading (whichever is
   greater) over its calibrated range, plus an NETD < 20 mK
   per-pixel (1-σ).  This is the irreducible noise floor on the SDK's
   own output at the recorded object parameters.

3. **Preset clamping and out-of-range extrapolation.**  ATS files
   recorded on the 700-1500 °C preset clamp pixels below 700 °C to
   exactly 700 °C and above 1500 °C to exactly 1500 °C in the SDK's
   factory calibration.  Histogram "spikes" at those two values are
   saturation, not real pixel values.  When you re-target with a
   *lower* emissivity (in test mode or via override), the Planck
   inversion is mathematically defined above 1500 °C but the
   underlying SDK Kelvin numbers are already extrapolated outside the
   factory-calibrated range — so apparent temperatures > 1500 °C
   carry an additional 10-30 % calibration uncertainty on top of #1
   and #4.  To get trustworthy values above 1500 °C, record on a
   higher-range preset (e.g. 1000-2500 °C) before doing the
   emissivity sweep.

4. **Single-wavelength model error.**  This tool inverts Planck at
   one effective wavelength (`λ_eff = 3.5 µm` by default).  The real
   X6900sc integrates radiance over the 2-5 µm pass-band weighted by
   the detector quantum efficiency and the filter transmission, so a
   single λ is itself an approximation.  Practical impact:

   - For **moderate** emissivity changes (factor ≤ ~2) and T below
     ~1500 °C the single-λ assumption is within 1-2 % of a full
     band-integrated inversion.
   - For **very hot** scenes (T > 1700 °C) the Planck curve peaks
     well below 3.5 µm (Wien displacement: peak at 2900 µm·K / T) so
     the detector sees more short-wavelength signal than the single-λ
     model.  Lowering `DEFAULT_LAMBDA_EFF_UM` to 2.8 – 3.0 µm closes
     most of this gap.
   - The reflected-radiance term is omitted (see #5).

5. **Reflected-radiance term omitted in the post-correction.**  Valid
   when `T_scene` is much larger than `T_refl`.  Concretely: for a
   700 °C scene against a 20 °C room the reflected contribution is
   below 10⁻¹⁵ of the scene's own emission and is wholly negligible.
   For a 200 °C scene against a 100 °C oven it would be a several-
   percent contributor and the correction would have to be extended.

6. **Locked object-parameter overrides.**  When the SDK reports
   `can_change_object_parameters = False`, overrides on every
   parameter *other* than emissivity are silently ignored.  This tool
   prints a `[info]` line when that happens and records both the
   recorded value and the requested value in the JSON so the failure
   is transparent.

7. **Atmospheric / external-optics losses.**  Reasonable for the
   sub-meter target distances used on this project; the recorded
   `atmospheric_transmission = 0.97` corresponds to ~1 m of room air
   in the MWIR band.  At metre-scale distances the contribution to
   total uncertainty is well under 1 °C and is dwarfed by the
   emissivity term.

The `*_meta.json` next to every TIFF records, for full transparency:
the recorded object parameters, the applied overrides, the emissivity
used to compute the stored pixel values, the crop and flip applied
afterwards, and (in test mode) the Wien correction parameters
`λ_eff`, `C2`, and the source emissivity used as `ε_assumed`.

## How to read the output

### In Fiji / ImageJ

`File → Open…`, point at `*_eps*_temp_C.tif`. Fiji recognises the BigTIFF
and opens a 32-bit grayscale stack. Hover your cursor over any pixel and
the status bar shows the temperature directly. Use `Image → Adjust →
Brightness/Contrast` and `Auto` to stretch the very narrow real range so
spatial structure becomes visible — the actual numerical pixel values
are unchanged by the display adjustment.

### In Python

```python
import tifffile, json, numpy as np

stack = tifffile.imread("Rec-000548_eps0.92_temp_C.tif")
                                          # shape (5311, 410, 440), float32, °C
                                          # (post-crop + post-flip)
meta  = json.loads(open("Rec-000548_eps0.92_meta.json").read())

print(stack.shape, stack.dtype, stack.min(), stack.mean(), stack.max())
print("camera:", meta["source_info"]["camera"],
      "  fps:", meta["preset_info"][0]["frame_rate"])
print("object params:", meta["object_parameters"])
```

### Reading raw counts instead

If at any point you want the unconverted 14-bit ADC counts (e.g. you
want to apply your own Planck inversion or work outside the calibrated
range), reopen the source `.ats`:

```python
import fnv, fnv.file, numpy as np

f = fnv.file.ImagerFile("Rec-000548.ats")
f.unit = fnv.Unit.COUNTS
f.get_frame(0)
raw = np.asarray(f.final, dtype=np.uint16).reshape((f.height, f.width))
```

## What's in `*_meta.json`

```json
{
  "n_frames": 5311,
  "width": 640,
  "height": 512,
  "data_type_in_tiff": "float32",
  "pixel_unit_written": "celsius",
  "tiff_pixel_meaning": "Per-pixel calibrated temperature in celsius. ...",
  "preview": { "frame_index": 536, "frame_mean_temperature": 410.5,
               "frame_mean_temperature_unit": "celsius",
               "selection_rule": "frame with the highest mean temperature ..." },
  "source_info": {
    "camera": "X6900sc", "camera_serial": "00139",
    "lens": "50 mm Macro", "filter": "2. ND 2.0, 2000-5000nm",
    "ad_bits": 14, "image_width": 640, "image_height": 512, ... },
  "object_parameters": {
    "emissivity": 0.92, "distance": 1.0,
    "reflected_temp": 293.15, "atmosphere_temp": 293.15,
    "relative_humidity": 0.3, "atmospheric_transmission": 0.973, ... },
  "preset_info": [
    { "frame_rate": 1000.0, "int_time": 0.01333,
      "min_temp": 973.15, "max_temp": 1773.15, "calibrated": true, ... } ]
}
```

## Notes on the preview frame

Picking "the hottest frame" requires one mean per frame. On 512×640
float32 it costs ~1 ms per frame and one extra ~1.3 MB array kept in
memory. The overall overhead is well below 5 % of the disk write and
the memory is negligible.

If you ever want a deterministic preview instead (e.g. for reproducible
side-by-side comparisons), set `FALLBACK_PREVIEW_FRAME = 700` near the
top of the script, comment out the max-mean tracking inside the loop,
and the fallback path takes over.

## Notes on file size and compression

`float32` would naively be 4 × the size of the camera's raw uint16
output, but real thermal recordings have very high spatial homogeneity
(most pixels at most times sit in a narrow temperature range), and
zlib at level 5 compresses them down dramatically. For a 5311-frame
640 × 512 recording the float32 BigTIFF typically ends up between
10 MB and 1 GB depending on how much of the frame's range the scene
actually exercises.

## Troubleshooting

**`ERROR: the FLIR Science File SDK Python bindings are not installed`**
You either skipped the MSI step, installed the wheel into a different
Python than the one running the script, or installed the wrong cp
version. `python --version` and re-pick the wheel.

**The TIFF opens but every pixel looks like the same value**
The scene's temperature range is below the camera's calibration limit
for the active preset, so the SDK is reporting the clamp value
everywhere. Check `preset_info.min_temp` / `max_temp` in the JSON.
Switching the camera to a different preset before recording, or using
`Unit.TEMPERATURE_USER` with adjusted object parameters, gives the SDK
room to report below-preset values.

**Conversion runs but the preview frame is at frame 0 every time**
`f.final` is returning the same buffer reference every iteration —
make sure you `.copy()` the array when you record a new best. The
supplied script does this.

**RAM is tight on a low-memory machine**
The script streams; it never holds more than two frames in RAM (the
current one and the running best). On a 640 × 512 camera that is
~2.6 MB peak.

## Files in this repo

```
flir_ats_batch.py                    the batch converter
installers/
    FLIRScienceFileSDK-2026.1.2+10-Windows-x64.msi   FLIR Science File SDK
README.md                            this file
.gitignore                           keeps generated outputs out of git
```

## Licence

The Python script in this repository is offered without warranty;
treat it as a starting point and adapt it to your data.

The bundled `FLIRScienceFileSDK-2026.1.2+10-Windows-x64.msi` is the
property of Teledyne FLIR LLC and is redistributed here purely for
convenience. Its end-user licence agreement applies in the same way as
if you had downloaded it directly from FLIR. By installing the MSI you
accept that EULA.
