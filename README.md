# FLIR ATS → 16-bit BigTIFF

A batch converter that turns FLIR `.ats` recordings (from cameras like the
X6900sc, A6750sc, etc.) into **lossless 16-bit BigTIFF stacks** plus a
metadata sidecar and a single preview image, per file.

The TIFFs open straight into Fiji / ImageJ as a 16-bit grayscale stack
and load into Python as a `(n_frames, H, W) uint16` ndarray via
`tifffile.imread`.

Per input `Rec-NNNNNN.ats` the tool writes three files into your output
folder:

| File | Content |
|---|---|
| `Rec-NNNNNN_raw.tif` | Multi-page BigTIFF, every page is one camera frame stored as `uint16` (raw 14-bit ADC counts in a 16-bit container), zlib level 5. |
| `Rec-NNNNNN_meta.json` | Camera (model, serial, lens, filter), recording (frame rate, integration time, temperature range), and object parameters (emissivity, distance, reflected / atmospheric temperature, humidity). Enough to reconstruct calibrated temperatures with the FLIR SDK. |
| `Rec-NNNNNN_preview.png` | One 8-bit grayscale preview frame, selected as the frame with the highest mean ADC count across the whole recording (= the overall hottest moment). The frame index is recorded in the JSON. |

## What you need to download

1. **This repository.** Clone or download as a ZIP.
2. **FLIR Science File SDK** – the Windows MSI is bundled in
   [`installers/`](installers/) for convenience
   (`FLIRScienceFileSDK-2026.1.2+10-Windows-x64.msi`). If you would
   rather get the latest version directly from FLIR, you can register a
   free account at <https://flir.custhelp.com/app/account/fl_login> and
   download from the Software downloads section.
3. **Python 3.10, 3.11, 3.12, 3.13, or 3.14** for Windows x64 – any
   recent CPython will do.
4. **Python packages** – `numpy`, `tifffile`, `pillow` (see
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

   Wheels for cp310, cp311, cp312, cp313, cp314 are all in that folder –
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
--no-recurse       only scan the top level of --input (default scans subfolders)
--overwrite        re-convert files whose outputs already exist
```

The tool reports a per-file progress line and an end-of-batch summary:

```
[setup] FLIR SDK loaded: ...\fnv\__init__.py
[setup] input:  D:\Recordings\modulated
[setup] output: E:\converted\modulated
[setup] 12 .ats files to convert (skip existing)

=== [1/12] Rec-000548.ats ===
  camera = X6900sc  serial = 00139
  5311 frames @ 1000.0 fps  512x640  raw payload ~ 3.48 GB
    531/5311  (87 fr/s, elapsed 0.1 min, ETA 0.9 min)
    ...
  -> ok
...
[done] 12 files in 14.3 min  (ok=12, skipped=0, error=0)
```

## How the pixel data is stored, and how to get temperature

### What is in the TIFF

Every page of the BigTIFF is a single camera frame stored as the
camera's **raw 14-bit ADC counts** (range 0–16383) in a `uint16`
container. No gain, offset, NUC, or temperature conversion has been
applied; this is the most faithful representation possible and is
exactly what `Unit.COUNTS` returns from the FLIR SDK.

We *do not* store an RGB / colourised version on purpose – that would
throw away ~6 bits of dynamic range and the temperature information
encoded in the values themselves.

### Converting counts to temperature (Planck inversion)

FLIR cameras are calibrated against a blackbody at the factory. The
documented inversion is

```
T_kelvin  = PB / ln( PR1 / (PR2 · (raw + PO)) + PF )
T_celsius = T_kelvin − 273.15
```

with five camera-specific Planck constants (PR1, PR2, PB, PF, PO) plus
the object parameters (emissivity, reflected temperature, distance,
atmospheric temperature, humidity, external optics) that further
correct for the recording conditions.

The Planck constants are **not** exposed by the modern Science File SDK
as plain attributes – they live inside the camera's calibration tables
that the SDK applies internally. The lossless storage strategy of this
tool sidesteps that: as long as you keep the original `.ats` file (or
just re-open it with the SDK), you can produce temperature at any time:

```python
import fnv, fnv.file, numpy as np
f = fnv.file.ImagerFile("Rec-000548.ats")
f.unit = fnv.Unit.TEMPERATURE_FACTORY     # or TEMPERATURE_USER if you have
                                          # custom object parameters
f.get_frame(0)
T_celsius = np.asarray(f.final).reshape((f.height, f.width)) - 273.15
```

If you only have the TIFF + JSON (no original `.ats`), you can still get
**relative** temperatures from the raw counts via a linear model fit to
the camera's `min_temp / max_temp` range stored in the JSON, but
**absolute** temperatures require the original `.ats` (the calibration
tables) and the SDK.

### What is in `*_meta.json`

The JSON sidecar is the minimum you would carry to a colleague who only
has the TIFF:

```json
{
  "n_frames": 5311,
  "width": 640,
  "height": 512,
  "pixel_unit_written": "COUNTS (raw 14-bit ADC in uint16)",
  "preview": { "frame_index": 4231, "frame_mean_count": 736.4,
               "selection_rule": "frame with the highest mean ADC count ..." },
  "source_info": {
      "camera": "X6900sc", "camera_serial": "00139",
      "lens": "50 mm Macro", "filter": "2. ND 2.0, 2000-5000nm",
      "ad_bits": 14, "frame_width": 640, "frame_height": 513,
      "image_width": 640, "image_height": 512,
      "num_presets": 4, ... },
  "object_parameters": {
      "emissivity": 0.92, "distance": 1.0,
      "reflected_temp": 293.15, "atmosphere_temp": 293.15,
      "relative_humidity": 0.3,
      "atmospheric_transmission": 0.973, ... },
  "preset_info": [
      { "frame_rate": 1000.0, "int_time": 0.01333,
        "min_temp": 973.15, "max_temp": 1773.15,
        "calibrated": true, ... }, ...
  ]
}
```

## How to read the output

### In Fiji / ImageJ

`File → Open…`, point at `*_raw.tif`. Fiji recognises BigTIFF and opens
a 16-bit grayscale stack. Use `Image → Adjust → Brightness/Contrast` to
auto-stretch the very narrow real value range (the full uint16 range is
not used).

### In Python

```python
import tifffile, json, numpy as np

stack = tifffile.imread("Rec-000548_raw.tif")     # shape (5311, 512, 640), uint16
meta  = json.loads(open("Rec-000548_meta.json").read())

print(stack.shape, stack.dtype, stack.min(), stack.max())
print("recording:", meta["source_info"]["camera"],
      meta["preset_info"][0]["frame_rate"], "fps")
```

To get calibrated temperature you reopen the original .ats with the SDK
(see [Converting counts to temperature](#converting-counts-to-temperature-planck-inversion)).

## Notes on the preview frame

Picking "the hottest frame" requires one mean per frame. On 512×640
uint16 that costs ~0.5 ms per frame, ~3 % above the cost of the disk
write itself, and one extra ~640 kB array kept in memory. Both are well
inside any reasonable budget and are always on.

If you ever want a deterministic preview instead (e.g. for reproducible
side-by-side comparisons), set `FALLBACK_PREVIEW_FRAME = 700` near the
top of the script, comment out the max-mean tracking inside the loop,
and the fallback path takes over.

## Troubleshooting

**`ERROR: the FLIR Science File SDK Python bindings are not installed`**
You either skipped the MSI step, installed the wheel into a different
Python than the one running the script, or installed the wrong cp
version. `python --version` and re-pick the wheel.

**`IndexError: tuple index out of range` while writing TIFF**
The frame buffer needs reshaping to `(H, W)` – the script already does
this; if you have hacked it, make sure `np.asarray(f.final).reshape((H, W))`
is in your code.

**Conversion runs but the preview frame is at frame 0 every time**
`f.final` is returning the same buffer reference every iteration – make
sure you `.copy()` the array when you record a new best. The supplied
script does this.

**RAM is tight on a low-memory machine**
The script streams; it never holds more than two frames in RAM (the
current one and the running best). On a 640×512 camera that is
~1.3 MB peak.

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
