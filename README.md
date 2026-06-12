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
| `Rec-NNNNNN_temp_C.tif` (or `_temp_K.tif`) | Multi-page BigTIFF, every page is one camera frame stored as `float32` Celsius (default) or Kelvin. Bit-for-bit the SDK's `Unit.TEMPERATURE_FACTORY` output (optionally shifted by 273.15). zlib level 5. |
| `Rec-NNNNNN_meta.json` | Camera (model, serial, lens, filter), recording (frame rate, integration time, temperature range), and object parameters (emissivity, distance, reflected / atmospheric temperature, humidity). |
| `Rec-NNNNNN_preview.png` | One 8-bit grayscale preview frame, selected as the frame with the highest mean temperature across the whole recording (= the overall hottest moment). The frame index is recorded in the JSON. |

### Why float32 temperature instead of raw counts?

Because temperature is what you actually want to look at: hover any pixel
in Fiji and the status bar shows the value directly, no Planck inversion
required. Storing the SDK's float32 output preserves it bit-for-bit, so
the temperature you read back from the TIFF is exactly the temperature
the SDK computes from the source `.ats` (plus the optional 273.15
shift). The raw 14-bit ADC counts remain recoverable at any time by
reopening the source `.ats` with `f.unit = fnv.Unit.COUNTS`; this tool
never modifies the source files.

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
   pip install numpy tifffile pillow
   ```

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

The script then writes one full conversion (TIFF + JSON + PNG) per
emissivity value, with the value embedded in the filename:

```
Rec-000548_eps0.300_temp_C.tif
Rec-000548_eps0.300_meta.json
Rec-000548_eps0.300_preview.png
Rec-000548_eps0.400_temp_C.tif
...
```

After the sweep finishes you're asked whether to run another sweep
(pick another file and / or other emissivity values), hand off to
batch mode, or quit.

**Batch mode** is the original whole-folder converter that uses one
shared set of parameters for every file. Batch mode never loops back —
when it finishes the program exits.

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

## How to read the output

### In Fiji / ImageJ

`File → Open…`, point at `*_temp_C.tif`. Fiji recognises the BigTIFF
and opens a 32-bit grayscale stack. Hover your cursor over any pixel and
the status bar shows the temperature directly. Use `Image → Adjust →
Brightness/Contrast` and `Auto` to stretch the very narrow real range so
spatial structure becomes visible — the actual numerical pixel values
are unchanged by the display adjustment.

### In Python

```python
import tifffile, json, numpy as np

stack = tifffile.imread("Rec-000548_temp_C.tif")
                                          # shape (5311, 512, 640), float32, °C
meta  = json.loads(open("Rec-000548_meta.json").read())

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
