r"""Batch lossless conversion of FLIR ATS-US recordings to 32-bit-float TIFF
where each pixel is a calibrated temperature.

Each input ``Rec-NNNNNN.ats`` produces three files in the output folder:

    Rec-NNNNNN_temp_C.tif    multi-page BigTIFF, float32, Celsius (default)
        or
    Rec-NNNNNN_temp_K.tif    multi-page BigTIFF, float32, Kelvin

    Rec-NNNNNN_meta.json     camera + recording + radiometric parameters
                             that were used during the temperature inversion

    Rec-NNNNNN_preview.png   single 8-bit grayscale preview frame, picked
                             as the overall hottest frame in the recording

The pixel values are bit-for-bit identical to what the FLIR Science File
SDK returns in ``Unit.TEMPERATURE_FACTORY``; the only post-processing is
the optional Kelvin to Celsius subtraction.  Stored in float32 so no
quantisation step is introduced.

To recover the original raw ADC counts you can always reopen the source
``.ats`` with ``f.unit = fnv.Unit.COUNTS`` (the .ats files are unchanged
by this tool).

Preview frame selection:
    During the conversion pass each frame's mean temperature is tracked
    in O(1) memory.  The frame with the highest mean is kept and written
    as the preview PNG so a quick look at the file shows the hottest
    moment of the recording.

Usage:
    python flir_ats_batch.py
        (then enter input and output folders at the prompts)

    python flir_ats_batch.py --input D:\path\to\ATS --output E:\path\to\TIFF
        (non-interactive)

    python flir_ats_batch.py --unit kelvin
        (write Kelvin instead of the default Celsius)

    python flir_ats_batch.py --no-recurse
        (only scan the top level of the input folder, default is recursive)

    python flir_ats_batch.py --overwrite
        (re-encode files whose outputs already exist; default skips them)

    python flir_ats_batch.py --no-confirm
        (skip both the interactive file-selection prompt and the
        radiometric-parameter inspection / override prompt; just process
        every file found, with the values recorded inside each .ats
        verbatim)

    python flir_ats_batch.py --files "1-10"
        (process only files 1 through 10 from the sorted .ats listing;
        also accepts e.g. "1 3 5", "1,3,5", or "1-5 10 15-20")

    python flir_ats_batch.py --mode test
        (skip the mode prompt and go straight to test mode -- a sweep of
        emissivity values applied to a single .ats file the user picks;
        each output is labelled with the emissivity it was made with)

Modes:
    On start the script asks whether you want TEST mode or BATCH mode.
    Test mode is for parameter-sensitivity studies: pick ONE .ats file,
    then type the emissivity values to try (either a range '[start, end,
    step]' or a discrete list '0.3 0.5 0.7').  The script then locates
    the hottest frame in that .ats (one fast pass over the recording),
    re-decodes ONLY that frame for every emissivity value, and writes a
    single multi-page float32 TIFF where each page is one emissivity
    result.  Each page has a "emissivity = X.XXX" label burnt as white
    text on a dark strip into the bottom edge so you can scroll-wheel
    through the pages in Fiji and read the value at a glance while
    comparing pixel temperatures.  After each sweep you can run another,
    hand off to batch mode, or quit.  Batch mode is the original
    whole-folder converter; it never loops back.

File-selection prompt (default behaviour):
    After finding the .ats files under --input the script prints a
    numbered listing (1-based) and asks you to enter "all" / a range /
    a list of individual indices / or a mix.  See --files above for the
    exact syntax.

Object-parameter inspection (default behaviour):
    Before the batch starts the script opens the FIRST .ats file in the
    input folder and prints its radiometric inversion parameters --
    emissivity, reflected temperature, distance, atmospheric temperature,
    relative humidity, atmospheric transmission, external optics
    temperature and transmission.  You can then optionally override any
    of them by typing a new number at the per-parameter prompts; any
    overrides are applied to EVERY file in the batch before the SDK
    computes temperatures.  The original .ats files are never modified;
    the recorded values and the applied overrides are both written into
    each *_meta.json so the run is reproducible.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------- FLIR SDK guard --------------------------------------------------
try:
    import fnv
    import fnv.file
    import fnv.reduce  # noqa: F401  (registers Unit, DataType enums)
except ImportError as exc:
    print(
        "ERROR: the FLIR Science File SDK Python bindings are not installed "
        "in this Python.\n"
        f"  caught:  {exc!r}\n"
        f"  python:  {sys.executable}\n\n"
        "Install: run the FLIR Science File SDK MSI bundled in installers/, "
        "then\n"
        '  pip install "C:/Program Files/FLIR Systems/sdks/file/python/dist/'
        'FileSDK-<version>-cp<XY>-cp<XY>-win_amd64.whl"\n'
        "Pick the wheel whose cpXY matches your Python (cp312 for Python 3.12, "
        "etc).  See README.md.",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    import numpy as np
    import tifffile
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print(
        f"ERROR: missing scientific package ({exc.name}).\n"
        "  Install with:  pip install numpy tifffile pillow",
        file=sys.stderr,
    )
    sys.exit(2)

# Matplotlib is a soft dependency used to draw the post-sweep summary
# PNGs (T-min / T-max curves vs the swept parameters).  If it isn't
# installed the conversion still runs end-to-end -- we just skip the
# plot and print a one-line note.
try:
    import matplotlib
    matplotlib.use("Agg")    # headless backend; no DISPLAY required
    import matplotlib.pyplot as plt
    _HAVE_MATPLOTLIB = True
except ImportError:
    _HAVE_MATPLOTLIB = False


# ---------- Tunable defaults -----------------------------------------------
PREVIEW_PCT_LO          = 1       # percentile for 16-bit -> 8-bit preview
PREVIEW_PCT_HI          = 99
FALLBACK_PREVIEW_FRAME  = 700
ZLIB_LEVEL              = 5       # tiff compression level (1=fast, 9=best)
KELVIN_OFFSET           = 273.15

# Persisted user-state file: every interactive prompt remembers the value
# the user entered last time and offers it as the press-Enter default.
STATE_FILE = Path.home() / ".flir_ats_batch_state.json"


def load_state() -> dict:
    """Read the saved last-run state from disk; empty dict on any error."""
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    """Best-effort write; failures are not fatal."""
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    except Exception:
        pass


# ---------- Crop + flip helpers (applied per-frame in pre-flip order) -----
# Crop is expressed as (x0, x1, y0, y1) in 0-based half-open Python slice
# coordinates over the PRE-FLIP frame (x = width axis, y = height axis).
# Flip mode is one of "none", "hflip", "vflip", "rot180" -- 180-degree
# rotation is equivalent to horizontal + vertical flip and is the default.

FLIP_MODES = ("none", "hflip", "vflip", "rot180")
DEFAULT_FLIP_MODE = "rot180"
CROP_PREVIEW_NAME = "_crop_preview.png"


def apply_flip(img: np.ndarray, mode: str) -> np.ndarray:
    """Return a flipped view of a 2D array.  Unknown / empty mode is a
    no-op.  Output is contiguous so downstream tifffile writes happy."""
    if not mode or mode == "none":
        return img
    if mode == "hflip":
        return np.ascontiguousarray(img[:, ::-1])
    if mode == "vflip":
        return np.ascontiguousarray(img[::-1, :])
    if mode == "rot180":
        return np.ascontiguousarray(img[::-1, ::-1])
    raise ValueError(
        f"unknown flip mode {mode!r}; choose one of {FLIP_MODES}")


def apply_crop(img: np.ndarray,
               crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Apply a `(x0, x1, y0, y1)` half-open crop to a 2D `(H, W)` array.
    Out-of-range coordinates are clipped silently; an empty intersection
    raises ValueError so the caller can warn before writing nothing."""
    if crop is None:
        return img
    x0, x1, y0, y1 = crop
    H, W = img.shape[:2]
    x0c, x1c = max(0, x0), min(W, x1)
    y0c, y1c = max(0, y0), min(H, y1)
    if x0c >= x1c or y0c >= y1c:
        raise ValueError(
            f"crop {crop} produces an empty region on {H}x{W} image")
    return img[y0c:y1c, x0c:x1c]


def parse_crop_spec(raw: str, W: int, H: int
                    ) -> tuple[int, int, int, int] | None:
    """Parse a 1-based inclusive crop spec like '100-540 50-460' and
    return a 0-based half-open `(x0, x1, y0, y1)` tuple.  Empty / 'none'
    / 'full' / 'all' / 'skip' -> None (no crop).  `W` and `H` are the
    sample frame's width and height; the spec is bounds-checked
    against them.  x is the width axis, y is the height axis."""
    raw = raw.strip().lower()
    if not raw or raw in ("none", "full", "all", "skip"):
        return None
    parts = raw.replace(",", " ").split()
    if len(parts) != 2 or "-" not in parts[0] or "-" not in parts[1]:
        raise ValueError(
            "crop spec must look like 'x_min-x_max y_min-y_max' "
            "(1-based inclusive), e.g. '100-540 50-460'")
    try:
        x_lo, x_hi = (int(s) for s in parts[0].split("-", 1))
        y_lo, y_hi = (int(s) for s in parts[1].split("-", 1))
    except ValueError as exc:
        raise ValueError(f"crop spec has non-integer values: {exc}") from exc
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    if y_lo > y_hi:
        y_lo, y_hi = y_hi, y_lo
    if not (1 <= x_lo <= W and 1 <= x_hi <= W
            and 1 <= y_lo <= H and 1 <= y_hi <= H):
        raise ValueError(
            f"crop x:{x_lo}-{x_hi} y:{y_lo}-{y_hi} is outside image "
            f"bounds x:1-{W} y:1-{H}")
    return (x_lo - 1, x_hi, y_lo - 1, y_hi)


def format_crop_spec(crop: tuple[int, int, int, int]) -> str:
    """Inverse of `parse_crop_spec`, used for round-trip persistence."""
    x0, x1, y0, y1 = crop
    return f"{x0 + 1}-{x1} {y0 + 1}-{y1}"


def _write_crop_preview(sample: np.ndarray,
                        crop: tuple[int, int, int, int],
                        out_path: Path) -> None:
    """Stretch a 2D sample frame to 8-bit grayscale and draw a red
    rectangle around the proposed crop region; save as PNG."""
    lo, hi = np.percentile(sample.astype(np.float32),
                           [PREVIEW_PCT_LO, PREVIEW_PCT_HI])
    u8 = np.clip(
        (sample.astype(np.float32) - lo) / max(hi - lo, 1e-9) * 255.0,
        0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(pil)
    x0, x1, y0, y1 = crop
    # Pillow rectangle is inclusive on both corners.  Use x1-1 / y1-1.
    draw.rectangle([(x0, y0), (x1 - 1, y1 - 1)],
                   outline=(255, 0, 0), width=2)
    pil.save(out_path)


def prompt_flip_mode(default: str | None = None) -> str:
    """Ask the user which post-crop flip to apply.  Empty input accepts
    `default` (or `DEFAULT_FLIP_MODE` if `default` is None / unknown)."""
    effective = default if default in FLIP_MODES else DEFAULT_FLIP_MODE
    print("\nFlip the output image?")
    print("  [1] none     -- no flip")
    print("  [2] hflip    -- horizontal flip (left <-> right)")
    print("  [3] vflip    -- vertical flip (top <-> bottom)")
    print("  [4] rot180   -- 180-degree rotation (= hflip + vflip)")
    label = {"none": "1=none", "hflip": "2=hflip",
             "vflip": "3=vflip", "rot180": "4=rot180"}[effective]
    tag = f"  [Enter = {label}]"
    while True:
        ans = input(f"Choice [1/2/3/4]{tag}: ").strip().lower()
        if not ans:
            return effective
        if ans in ("1", "n", "none"):
            return "none"
        if ans in ("2", "h", "hflip"):
            return "hflip"
        if ans in ("3", "v", "vflip"):
            return "vflip"
        if ans in ("4", "r", "rot180", "180"):
            return "rot180"
        print("  please type 1, 2, 3, or 4")


def prompt_crop_range(
        sample_frame: np.ndarray,
        out_dir: Path,
        default_spec: str | None = None,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Ask the user for a pre-flip crop region.  Each round writes a
    grayscale PNG of the sample frame with a red rectangle around the
    proposed crop (to `out_dir/_crop_preview.png`) and waits for the
    user to confirm / reset / cancel.

    Returns `(crop, accepted_spec)`:
      - crop = `(x0, x1, y0, y1)` half-open or None (no crop)
      - accepted_spec = the spec string suitable for re-offering next
        run; empty string means 'no crop' (so Enter next time keeps it)
    """
    H, W = int(sample_frame.shape[0]), int(sample_frame.shape[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_path = out_dir / CROP_PREVIEW_NAME

    print("\nCrop region (pre-flip coordinates)")
    print(f"  image size: {W} x {H}  (W x H)")
    print("  syntax:   x_min-x_max y_min-y_max  (1-based inclusive)")
    print("  example:  100-540 50-460")
    print("  press Enter (default below), or type 'none' / 'full' to skip crop")

    while True:
        if default_spec:
            tag = f"  [Enter = last: {default_spec}]"
        else:
            tag = "  [Enter = no crop]"
        raw = input(f"Crop{tag}: ").strip()
        if not raw:
            raw = default_spec or "none"
            print(f"  -> using default: {raw!r}")
        try:
            crop = parse_crop_spec(raw, W, H)
        except ValueError as exc:
            print(f"  invalid: {exc}, try again")
            continue
        if crop is None:
            print("  -> no crop will be applied")
            return None, ""

        _write_crop_preview(sample_frame, crop, preview_path)
        accepted_spec = format_crop_spec(crop)
        cw = crop[1] - crop[0]
        ch = crop[3] - crop[2]
        print(f"  -> crop x:{crop[0] + 1}-{crop[1]} "
              f"y:{crop[2] + 1}-{crop[3]}  ({cw} x {ch} px)")
        print(f"  preview PNG written: {preview_path}")
        print(f"    open it now -- the red rectangle shows what will be kept")
        ans = input(
            "Proceed? [y=confirm / r=reset / c=cancel crop]: "
        ).strip().lower()
        if ans in ("y", "yes"):
            return crop, accepted_spec
        if ans in ("c", "cancel", "n", "no"):
            print("  -> crop cancelled, no crop will be applied")
            return None, ""
        print("  resetting, re-entering")


def _read_hottest_frame_counts(ats_path: Path) -> tuple[int, np.ndarray]:
    """One scan of `ats_path` in Unit.COUNTS.  Returns
    `(hottest_idx, hottest_frame_uint16)`.  Cheap relative to the full
    float32 temperature decode -- used to fuel the crop-preview PNG and
    to skip the redundant scan inside the subsequent test-mode sweep."""
    f = fnv.file.ImagerFile(str(ats_path))
    f.unit = fnv.Unit.COUNTS
    n = int(f.num_frames)
    H, W = int(f.height), int(f.width)
    best_idx, best_mean = -1, -1.0
    best_frame: np.ndarray | None = None
    report_step = max(1, n // 5)
    for i in range(n):
        f.get_frame(i)
        page = np.asarray(f.final, dtype=np.uint16).reshape((H, W))
        m = float(page.mean())
        if m > best_mean:
            best_mean = m
            best_idx = i
            best_frame = page.copy()
        if (i + 1) % report_step == 0 or (i + 1) == n:
            print(f"    scan {i + 1}/{n}", flush=True)
    if best_frame is None:
        raise RuntimeError(f"no frames in {ats_path}")
    return best_idx, best_frame


# ---------- JSON helpers ---------------------------------------------------
def _jsonable(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if hasattr(x, "name") and hasattr(x, "value"):
        return f"{x.name}({x.value})"
    if isinstance(x, bytes):
        return x.hex()
    out = {}
    for a in dir(x):
        if a.startswith("_"):
            continue
        try:
            v = getattr(x, a)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            out[a] = _jsonable(v)
        except Exception:
            out[a] = repr(v)
    return out


def collect_metadata(f, *, n_frames, preview_idx, preview_mean, unit_label,
                     recorded_object_params=None, applied_overrides=None):
    return {
        "n_frames": int(n_frames),
        "width": int(f.width),
        "height": int(f.height),
        "data_type_in_tiff": "float32",
        "pixel_unit_written": unit_label,
        "tiff_pixel_meaning": (
            f"Per-pixel calibrated temperature in {unit_label}.  Values are "
            "the FLIR Science File SDK output in Unit.TEMPERATURE_FACTORY, "
            "minus 273.15 if the configured unit was Celsius.  The SDK "
            "applies the camera's factory Planck calibration plus the "
            "object parameters listed below."
        ),
        "preview": {
            "frame_index": int(preview_idx),
            "frame_mean_temperature": float(preview_mean),
            "frame_mean_temperature_unit": unit_label,
            "selection_rule": (
                "frame with the highest mean temperature across the whole "
                "recording (= overall hottest frame)"
            ),
        },
        "object_parameters_recorded": recorded_object_params,
        "object_parameters_applied": _jsonable(f.object_parameters),
        "object_parameter_overrides": applied_overrides or {},
        "raw_counts_recovery": (
            "Reopen the source .ats through the FLIR Science File SDK and "
            "set f.unit = fnv.Unit.COUNTS to get the unchanged 14-bit ADC "
            "counts back.  This tool does not modify the source .ats files."
        ),
        "source_info": _jsonable(f.source_info),
        "current_preset_index": int(f.preset),
        "preset_info": _jsonable(list(f.source_info.preset_info)),
    }


# ---------- Object-parameter inspection / override -------------------------
# Names of the radiometric inversion knobs we expose to the user.  Each
# entry is (sdk attribute, display unit, short description).  Loaded from
# the SDK's ObjectParameters class.
OBJECT_PARAM_FIELDS = [
    ("emissivity",               "0-1 (-)",   "surface emissivity"),
    ("reflected_temp",           "Kelvin",    "reflected (background) temperature"),
    ("distance",                 "metres",    "target distance"),
    ("atmosphere_temp",          "Kelvin",    "atmospheric temperature"),
    ("relative_humidity",        "0-1 (-)",   "relative humidity"),
    ("atmospheric_transmission", "0-1 (-)",   "atmospheric transmission"),
    ("ext_optics_temp",          "Kelvin",    "external optics temperature"),
    ("ext_optics_transmission",  "0-1 (-)",   "external optics transmission"),
]


def read_object_params(f) -> dict[str, float]:
    """Pull the current numeric value of every field we care about."""
    out = {}
    for name, _, _ in OBJECT_PARAM_FIELDS:
        try:
            out[name] = float(getattr(f.object_parameters, name))
        except Exception:
            out[name] = float("nan")
    return out


def print_object_params(params: dict[str, float], heading: str) -> None:
    print(f"\n  {heading}")
    print(f"  {'name':<28s}  {'value':>12s}  unit         description")
    print(f"  {'-'*28}  {'-'*12}  {'-'*12} {'-'*40}")
    for name, unit, desc in OBJECT_PARAM_FIELDS:
        v = params.get(name, float("nan"))
        print(f"  {name:<28s}  {v:>12.4f}  {unit:<12s} {desc}")


def prompt_object_param_overrides(initial: dict[str, float],
                                  last_overrides: dict[str, float] | None = None
                                  ) -> dict[str, float]:
    """Display the recorded parameters and let the user override any of
    them.  Returns the final dict (initial + overrides).  Loops until the
    user confirms.

    `last_overrides` (saved from a previous run) is shown as the
    suggested default at each per-parameter prompt.  Empty input keeps
    that suggestion (which falls back to the recorded value if the user
    did not override that parameter previously)."""
    last_overrides = dict(last_overrides or {})
    while True:
        print_object_params(initial, "Radiometric inversion parameters "
                            "(recorded inside the first .ats):")
        if last_overrides:
            print(f"\n  Last run overrode: " +
                  ", ".join(f"{k}={v}" for k, v in last_overrides.items()))
        print("\n  Press Enter at each prompt to keep the suggested value, "
              "or type a new number.")
        modify = input("\n  Do you want to override any of them? [y/N]: "
                       ).strip().lower()
        if modify not in ("y", "yes"):
            # Re-apply last_overrides if the user just presses through.
            out = dict(initial)
            out.update(last_overrides)
            return out

        new_vals = dict(initial)
        new_vals.update(last_overrides)
        for name, unit, desc in OBJECT_PARAM_FIELDS:
            cur = new_vals[name]
            note = " (last)" if name in last_overrides else ""
            raw = input(f"    {name} [{cur:.4f} {unit}{note}]: ").strip()
            if not raw:
                continue
            try:
                new_vals[name] = float(raw)
            except ValueError:
                print(f"      not a number, keeping {cur:.4f}")

        if new_vals == initial:
            print("  (no values changed)")
            return new_vals

        print_object_params(new_vals,
                            "Parameters that WILL be applied to every file:")
        confirm = input("\n  Proceed with these values? [y/N/edit]: "
                        ).strip().lower()
        if confirm in ("y", "yes"):
            return new_vals
        if confirm in ("e", "edit"):
            initial = new_vals    # iterate again on top of the edited copy
            continue
        print("  Cancelled; you can edit again.\n")
        initial = new_vals
        continue


def apply_overrides(f, overrides: dict[str, float]) -> None:
    """Write each override into f.object_parameters before frame reads.

    Note: many ATS files lock object_parameters at the SDK level
    (f.can_change_object_parameters is False) -- the setattr calls then
    silently succeed but the SDK ignores them when computing
    temperatures.  Callers that depend on emissivity changes taking
    effect should apply the exact-Planck post-correction in
    emissivity_correct_kelvin() instead."""
    op = f.object_parameters
    for name, val in overrides.items():
        try:
            setattr(op, name, float(val))
        except Exception as exc:
            print(f"    [warn] could not set object_parameters.{name} = "
                  f"{val!r}: {exc!r}", flush=True)


# ---------- Per-file conversion -------------------------------------------
def convert_one(ats_path: Path, out_dir: Path,
                *, unit: str = "celsius",
                overrides: dict[str, float] | None = None,
                overwrite: bool = False,
                stem_suffix: str = "",
                crop: tuple[int, int, int, int] | None = None,
                flip_mode: str = "none") -> dict:
    """Convert one .ats into _eps{E:.2f}_temp_{C|K}.tif + _meta.json + _preview.png.

    `stem_suffix` is appended to the output stem (used by test mode to
    label files with the emissivity value they were generated with).
    `crop` is a half-open `(x0, x1, y0, y1)` over the PRE-FLIP frame;
    `flip_mode` is one of `FLIP_MODES` and is applied AFTER the crop."""
    if unit not in ("celsius", "kelvin"):
        raise ValueError(f"unit must be 'celsius' or 'kelvin', got {unit!r}")
    unit_suffix = "C" if unit == "celsius" else "K"
    unit_label  = "celsius" if unit == "celsius" else "kelvin"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Open early so we know each file's recorded emissivity before we
    # build the output paths: the filename now records the actually-used
    # emissivity (per-file recorded value, or the override if any).
    t0 = time.time()
    f = fnv.file.ImagerFile(str(ats_path))

    recorded = read_object_params(f)
    eps_recorded = float(recorded.get("emissivity", 1.0))
    sdk_can_change = bool(getattr(f, "can_change_object_parameters", False))

    eps_used = (float(overrides["emissivity"])
                if overrides and "emissivity" in overrides
                else eps_recorded)
    eps_tag = f"_eps{eps_used:.2f}"

    stem      = ats_path.stem + stem_suffix + eps_tag
    out_tif   = out_dir / f"{stem}_temp_{unit_suffix}.tif"
    out_json  = out_dir / f"{stem}_meta.json"
    out_png   = out_dir / f"{stem}_preview.png"

    if (not overwrite
        and out_tif.exists() and out_json.exists() and out_png.exists()):
        return {"status": "skipped",
                "tif_gb": out_tif.stat().st_size / 1e9}

    # Apply user-supplied object-parameter overrides BEFORE switching
    # to a temperature unit.  Two cases:
    #
    #  - If the SDK actually honours live object_parameter edits, the
    #    new values flow through its own band-integrated radiometric
    #    inversion -- nothing else to do, leave eps_override = None.
    #
    #  - If the SDK silently ignores the setattr (some science-camera
    #    ATS files do this even though they advertise
    #    can_change_object_parameters = True), we have to post-correct
    #    in Python with the exact single-wavelength Planck inversion.
    #    Other-parameter overrides cannot be recovered this way and
    #    are reported as ignored.
    #
    # We don't trust the advertised flag -- we probe.  The probe uses
    # frame 0 (cheap, and only used to detect SDK responsiveness; the
    # actual per-frame conversion still walks every frame below).
    eps_override = None
    if overrides:
        apply_overrides(f, overrides)
        if "emissivity" in overrides \
                and abs(float(overrides["emissivity"]) - eps_recorded) > 1e-9:
            f.unit = fnv.Unit.TEMPERATURE_FACTORY
            sdk_responds, probe_delta = sdk_emissivity_actually_responds(
                f, best_idx=0, eps_recorded=eps_recorded,
                H=int(f.height), W=int(f.width),
            )
            if not sdk_responds:
                eps_override = float(overrides["emissivity"])
                print(f"  [info] SDK setattr on object_parameters.emissivity "
                      f"produced only {probe_delta:.3g} K change on the "
                      f"probe frame "
                      f"(claimed can_change={sdk_can_change}) -- "
                      f"falling back to exact-Planck post-correction "
                      f"(lambda_eff = {DEFAULT_LAMBDA_EFF_UM} um) to "
                      f"retarget emissivity {eps_recorded:.4f} -> "
                      f"{eps_override:.4f}",
                      flush=True)
                # Restore the recorded eps so the SDK keeps decoding at
                # its baseline -- the Wien/Planck post-correction
                # handles the retargeting.
                try:
                    f.object_parameters.emissivity = float(eps_recorded)
                except Exception:
                    pass
            # Re-apply any *other* overrides; restoring eps above may
            # have wiped them out on some SDK builds.
            apply_overrides(f, {k: v for k, v in overrides.items()
                                if k != "emissivity"})

    f.unit = fnv.Unit.TEMPERATURE_FACTORY

    n = int(f.num_frames)
    H, W = int(f.height), int(f.width)

    # Probe the post-crop+flip shape on a dummy frame so the progress
    # line and the meta JSON record the actual stored size.
    dummy = np.zeros((H, W), dtype=np.float32)
    try:
        sample_out = apply_flip(apply_crop(dummy, crop), flip_mode)
    except ValueError as exc:
        raise RuntimeError(
            f"crop {crop} does not fit this file's {H}x{W} frame: {exc}"
        ) from exc
    Hout, Wout = sample_out.shape[:2]

    print(f"  camera = {f.source_info.camera}  "
          f"serial = {f.source_info.camera_serial}", flush=True)
    print(f"  {n} frames @ {f.source_info.preset_info[0].frame_rate:.1f} fps  "
          f"src {H}x{W} -> out {Hout}x{Wout}  "
          f"crop={crop}  flip={flip_mode}", flush=True)
    print(f"  emissivity used = {eps_used:.4f}  "
          f"(recorded {eps_recorded:.4f})", flush=True)
    print(f"  payload (float32) ~ {n*Hout*Wout*4/1e9:.2f} GB  "
          f"writing temperature in {unit_label}", flush=True)

    # ---- Track the hottest frame (max mean temperature) -----------------
    best_mean: float = -float("inf")
    best_idx:  int   = -1
    best_frame: np.ndarray | None = None

    offset = -KELVIN_OFFSET if unit == "celsius" else 0.0
    report_step = max(1, n // 10)

    with tifffile.TiffWriter(str(out_tif), bigtiff=True) as tw:
        for i in range(n):
            f.get_frame(i)
            page_K = (np.asarray(f.final, dtype=np.float32)
                      .reshape((H, W)))
            if eps_override is not None:
                page_K = emissivity_correct_kelvin(
                    page_K, eps_recorded, eps_override,
                    lambda_eff_um=DEFAULT_LAMBDA_EFF_UM,
                )
            page = page_K + offset
            page = apply_flip(apply_crop(page, crop), flip_mode)
            tw.write(
                page, photometric="minisblack",
                compression="zlib", compressionargs={"level": ZLIB_LEVEL},
                contiguous=False,
            )
            m = float(page.mean())
            if m > best_mean:
                best_mean = m
                best_idx  = i
                best_frame = page.copy()
            if (i + 1) % report_step == 0 or (i + 1) == n:
                dt = time.time() - t0
                rate = (i + 1) / max(dt, 1e-3)
                eta  = (n - i - 1) / max(rate, 1e-3)
                print(f"    {i+1}/{n}  ({rate:.0f} fr/s, "
                      f"elapsed {dt/60:.1f} min, ETA {eta/60:.1f} min)",
                      flush=True)

    if best_frame is None:
        idx = FALLBACK_PREVIEW_FRAME if n > FALLBACK_PREVIEW_FRAME else n // 2
        f.get_frame(idx)
        raw_K = (np.asarray(f.final, dtype=np.float32).reshape((H, W)))
        fallback = apply_flip(apply_crop(raw_K + offset, crop), flip_mode)
        best_frame = fallback
        best_idx, best_mean = idx, float(best_frame.mean())

    # ---- Preview PNG (1-99 % percentile stretch to 8-bit) ---------------
    lo, hi = np.percentile(best_frame, [PREVIEW_PCT_LO, PREVIEW_PCT_HI])
    u8 = np.clip((best_frame - lo) / max(hi - lo, 1e-9) * 255.0,
                 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_png)

    # ---- Metadata JSON --------------------------------------------------
    meta = collect_metadata(
        f, n_frames=n, preview_idx=best_idx,
        preview_mean=best_mean, unit_label=unit_label,
        recorded_object_params=recorded,
        applied_overrides=overrides or {},
    )
    meta["source_file"] = str(ats_path)
    meta["emissivity_used"] = float(eps_used)
    meta["source_height"] = int(H)
    meta["source_width"]  = int(W)
    meta["output_height"] = int(Hout)
    meta["output_width"]  = int(Wout)
    meta["crop_x0_x1_y0_y1_zero_based_half_open"] = (
        list(crop) if crop is not None else None)
    meta["flip_mode_after_crop"] = flip_mode
    out_json.write_text(json.dumps(meta, indent=2, default=str))

    return {
        "status": "ok",
        "n_frames": n,
        "preview_frame": best_idx,
        "preview_mean_temp": best_mean,
        "unit": unit_label,
        "elapsed_s": time.time() - t0,
        "tif_gb": out_tif.stat().st_size / 1e9,
    }


# ---------- File-selection helpers ----------------------------------------
def parse_file_selection(raw: str, n_total: int) -> list[int]:
    """Parse '', 'all', '1-10', '1 3 5', '1,3,5', '1-5 10 15-20' (or any
    combination) into a sorted, deduped, 1-based list of indices clipped
    to [1, n_total].  Empty string or 'all' (case-insensitive) -> every
    index 1..n_total.  Raises ValueError on garbage tokens."""
    raw = raw.strip().lower()
    if not raw or raw == "all":
        return list(range(1, n_total + 1))
    selected: set[int] = set()
    for tok in raw.replace(",", " ").split():
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            try:
                lo_i, hi_i = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"invalid range {tok!r}") from exc
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            for i in range(lo_i, hi_i + 1):
                if 1 <= i <= n_total:
                    selected.add(i)
        else:
            try:
                i = int(tok)
            except ValueError as exc:
                raise ValueError(f"not a number {tok!r}") from exc
            if 1 <= i <= n_total:
                selected.add(i)
    return sorted(selected)


def _compress_indices(idxs: list[int]) -> str:
    """Collapse a sorted 1-based index list into the most compact spec
    string the inverse of `parse_file_selection` would accept, e.g.
    [1,2,3,5,7,8,9] -> '1-3 5 7-9'.  Used when persisting the user's
    last batch-mode file selection."""
    if not idxs:
        return ""
    parts: list[str] = []
    run_lo = run_hi = idxs[0]
    for i in idxs[1:]:
        if i == run_hi + 1:
            run_hi = i
            continue
        parts.append(str(run_lo) if run_lo == run_hi
                     else f"{run_lo}-{run_hi}")
        run_lo = run_hi = i
    parts.append(str(run_lo) if run_lo == run_hi else f"{run_lo}-{run_hi}")
    return " ".join(parts)


def _display_name(p: Path, input_dir: Path) -> str:
    try:
        return str(p.relative_to(input_dir))
    except ValueError:
        return p.name


def prompt_file_selection(ats_files: list[Path],
                          input_dir: Path,
                          default_selection: str | None = None) -> list[Path]:
    """List the .ats files with 1-based indices and let the user pick a
    subset.  Re-prompts on invalid input or rejected confirmation.
    Empty input accepts `default_selection` if given (typically the
    string the user typed last time, like '1-10')."""
    n = len(ats_files)
    width = len(str(n))
    while True:
        print(f"\nFound {n} .ats files under {input_dir}:")
        for i, p in enumerate(ats_files, 1):
            sz_gb = p.stat().st_size / 1e9
            print(f"  {i:>{width}}  {_display_name(p, input_dir):<40s}  "
                  f"({sz_gb:5.2f} GB)")
        print("\nWhich files do you want to convert?")
        print("  press Enter (or 'all') -> every file")
        print("  '1-10'                 -> a range (inclusive)")
        print("  '1 3 5' or '1,3,5'     -> individual files")
        print("  '1-5 10 15-20'         -> mix of ranges and individuals")
        tag = f"  [Enter = last: {default_selection}]" if default_selection else ""
        raw = input(f"Selection{tag}: ").strip()
        if not raw and default_selection:
            raw = default_selection
            print(f"  -> using last: {raw}")
        try:
            idxs = parse_file_selection(raw, n)
        except ValueError as exc:
            print(f"  invalid selection: {exc}, try again")
            continue
        if not idxs:
            print("  no files selected, try again")
            continue
        picked = [ats_files[i - 1] for i in idxs]
        print(f"\n  Selected {len(picked)} of {n} file(s):")
        for p in picked:
            print(f"    - {_display_name(p, input_dir)}")
        ans = input("Proceed with this selection? [y/N/edit]: "
                    ).strip().lower()
        if ans in ("y", "yes"):
            return picked
        if ans in ("e", "edit"):
            continue
        print("  cancelled, re-listing")


def prompt_single_file(ats_files: list[Path], input_dir: Path,
                       default_name: str | None = None) -> Path:
    """List the files with 1-based indices and let the user pick exactly
    one for the test-mode emissivity sweep.  `default_name` is the
    relative filename the user picked last time; if it still appears in
    the listing, Enter picks it again."""
    n = len(ats_files)
    width = len(str(n))
    default_idx = None
    if default_name:
        for i, p in enumerate(ats_files, 1):
            if _display_name(p, input_dir) == default_name:
                default_idx = i
                break
    while True:
        print(f"\nFound {n} .ats files under {input_dir}:")
        for i, p in enumerate(ats_files, 1):
            sz_gb = p.stat().st_size / 1e9
            mark = " <- last" if i == default_idx else ""
            print(f"  {i:>{width}}  {_display_name(p, input_dir):<40s}  "
                  f"({sz_gb:5.2f} GB){mark}")
        tag = (f"  [Enter = last: {default_idx} ({default_name})]"
               if default_idx else "")
        raw = input(f"Pick ONE file by its number{tag}: ").strip()
        if not raw and default_idx:
            raw = str(default_idx)
        try:
            idx = int(raw)
        except ValueError:
            print("  not a number, try again")
            continue
        if not (1 <= idx <= n):
            print(f"  out of range [1, {n}], try again")
            continue
        picked = ats_files[idx - 1]
        print(f"  -> {_display_name(picked, input_dir)}")
        return picked


# ---------- Emissivity selection (test mode) ------------------------------
def parse_emissivity_values(raw: str) -> list[float]:
    """Parse either '[start, end, step]' (inclusive range) or a
    space/comma-separated list of individual values.  Returns a list of
    floats clamped to (0, 1] with duplicates kept, in user order."""
    raw = raw.strip()
    if not raw:
        raise ValueError("empty input")

    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].replace(",", " ").split()
        if len(inner) != 3:
            raise ValueError(
                f"range form needs exactly 3 numbers "
                f"[start, end, step], got {len(inner)}"
            )
        start, end, step = (float(x) for x in inner)
        if step <= 0:
            raise ValueError("step must be > 0")
        if end < start:
            raise ValueError("end must be >= start")
        vals: list[float] = []
        x = start
        # tiny tolerance so e.g. start=0.1, end=0.9, step=0.1 yields 0.9
        while x <= end + step * 1e-6:
            vals.append(round(x, 6))
            x += step
    else:
        toks = raw.replace(",", " ").split()
        try:
            vals = [float(t) for t in toks]
        except ValueError as exc:
            raise ValueError(f"not all tokens are numbers: {exc}") from exc

    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"emissivity must be in (0, 1], got {v}"
            )
    if not vals:
        raise ValueError("no values parsed")
    return vals


DEFAULT_EMISSIVITY_SPEC = "[0.1, 0.95, 0.05]"


def prompt_emissivity_values(
        default: str | None = None,
) -> tuple[list[float], str]:
    """Get the list of emissivity values to test from the user.  Repeats
    on parse error until accepted.  An empty line accepts `default`
    (or `DEFAULT_EMISSIVITY_SPEC` if `default` is None).

    Returns (vals, accepted_spec) where `accepted_spec` is the exact
    string the user typed (or the default they accepted), suitable for
    re-offering verbatim next time."""
    effective_default = default or DEFAULT_EMISSIVITY_SPEC
    print("\nEmissivity values to test")
    print("  range form:  [start, end, step]   e.g.  [0.3, 0.9, 0.1]")
    print("  list form:   0.3 0.5 0.7   or   0.3,0.5,0.7")
    print(f"  default (just press Enter): {effective_default}"
          + (f"  [remembered from last run]" if default else ""))
    while True:
        raw = input("Emissivities: ").strip()
        if not raw:
            raw = effective_default
            print(f"  -> using default: {raw}")
        try:
            vals = parse_emissivity_values(raw)
        except ValueError as exc:
            print(f"  invalid input: {exc}, try again")
            continue
        print(f"  -> {len(vals)} value(s): "
              f"{', '.join(f'{v:.3f}' for v in vals)}")
        ans = input("Proceed? [y/N/edit]: ").strip().lower()
        if ans in ("y", "yes"):
            return vals, raw
        if ans in ("e", "edit"):
            continue


# ---------- Lambda-eff selection (test mode, exact-Planck path) ----------
LAMBDA_EFF_BOUNDS_UM = (0.5, 15.0)
DEFAULT_LAMBDA_EFF_SPEC = "[2.5, 4.5, 0.5]"


def parse_lambda_eff_values(raw: str) -> list[float]:
    """Like `parse_emissivity_values` but for the effective wavelength of
    the camera+filter pass-band, in micrometres.  Bounds:
    `LAMBDA_EFF_BOUNDS_UM` = (0.5, 15.0).  Accepts either
    `[start, end, step]` or a space/comma-separated list."""
    raw = raw.strip()
    if not raw:
        raise ValueError("empty input")
    lo, hi = LAMBDA_EFF_BOUNDS_UM
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].replace(",", " ").split()
        if len(inner) != 3:
            raise ValueError(
                f"range form needs exactly 3 numbers "
                f"[start, end, step], got {len(inner)}"
            )
        start, end, step = (float(x) for x in inner)
        if step <= 0:
            raise ValueError("step must be > 0")
        if end < start:
            raise ValueError("end must be >= start")
        vals: list[float] = []
        x = start
        while x <= end + step * 1e-6:
            vals.append(round(x, 6))
            x += step
    else:
        toks = raw.replace(",", " ").split()
        try:
            vals = [float(t) for t in toks]
        except ValueError as exc:
            raise ValueError(f"not all tokens are numbers: {exc}") from exc

    for v in vals:
        if not (lo <= v <= hi):
            raise ValueError(
                f"lambda_eff must be in [{lo}, {hi}] um, got {v}"
            )
    if not vals:
        raise ValueError("no values parsed")
    return vals


def prompt_lambda_eff_values(
        default: str | None = None,
) -> tuple[list[float], str]:
    """Get the list of effective wavelengths to test (micrometres).  Same
    UX as `prompt_emissivity_values`.  An empty line accepts `default`
    (or `DEFAULT_LAMBDA_EFF_SPEC` if `default` is None).

    Returns (vals, accepted_spec) for round-trip persistence.  When the
    chosen correction method is `sdk_native`, lambda_eff has no effect
    on the output -- callers should still collect it for completeness
    but expect the SDK-native page builder to collapse the lambda list
    to a single value with a console warning."""
    effective_default = default or DEFAULT_LAMBDA_EFF_SPEC
    lo, hi = LAMBDA_EFF_BOUNDS_UM
    print("\nEffective wavelength (lambda_eff) values to test, in um")
    print("  range form:  [start, end, step]   e.g.  [2.5, 4.5, 0.5]")
    print("  list form:   2.5 3.0 3.5   or   2.5,3.0,3.5")
    print(f"  bounds:      [{lo}, {hi}] um")
    print(f"  default (just press Enter): {effective_default}"
          + (f"  [remembered from last run]" if default else ""))
    print("  Only used when the SDK refuses live eps edits and the tool "
          "post-corrects in Python (exact-Planck path).  See README.")
    while True:
        raw = input("lambda_eff (um): ").strip()
        if not raw:
            raw = effective_default
            print(f"  -> using default: {raw}")
        try:
            vals = parse_lambda_eff_values(raw)
        except ValueError as exc:
            print(f"  invalid input: {exc}, try again")
            continue
        print(f"  -> {len(vals)} value(s): "
              f"{', '.join(f'{v:.3f}' for v in vals)}")
        ans = input("Proceed? [y/N/edit]: ").strip().lower()
        if ans in ("y", "yes"):
            return vals, raw
        if ans in ("e", "edit"):
            continue
        print("  cancelled, re-entering")


# ---------- Exact Planck emissivity correction ----------------------------
# This SDK release locks `f.object_parameters` for ATS files written by
# many science cameras: `can_change_object_parameters` is False and the
# only TEMPERATURE_* unit on the file's `supported_units` is
# TEMPERATURE_FACTORY (trying to set f.unit = TEMPERATURE_USER raises
# 'failed to set unit').  That means setting f.object_parameters.* has
# NO EFFECT on the SDK's temperature output -- changing emissivity in
# Python and re-reading the frame gives exactly the same numbers back.
#
# To still allow an emissivity-sensitivity study we do the inversion
# ourselves with the EXACT Planck radiance ratio.  At a single
# effective wavelength `lambda_eff_um`,
#
#     B(T) = const / (exp(C2 / (lambda_eff * T)) - 1)        (Planck)
#
# Equating measured radiances under two emissivity assumptions and
# ignoring the reflected-radiance term (negligible when scene T is much
# larger than the reflected-environment T):
#
#     eps_assumed * B(T_assumed)  =  eps_new * B(T_new)
#
# rearranges to a closed-form inversion:
#
#     exp(C2/(lambda_eff*T_new)) - 1
#         = (eps_new/eps_assumed) * (exp(C2/(lambda_eff*T_assumed)) - 1)
#
#                            C2
#     T_new = -----------------------------------------------------
#             lambda_eff * ln(1 + (eps_new/eps_assumed)
#                                  * (exp(C2/(lambda_eff*T_assumed)) - 1))
#
# This is the same as the Wien high-T approximation when the "-1"
# terms are negligible (i.e. C2/(lambda*T) >> 1), but stays accurate
# all the way up to and beyond 2000 C, where the Wien form would
# overestimate T by tens of percent.  No extra parameters; the cost
# is one numpy exp + log per pixel per requested emissivity.
#
# `lambda_eff_um` is the camera+filter band's effective wavelength;
# 3.5 micrometres is a sensible default for a mid-wave InSb camera
# (X6900sc, A6750sc etc.) with a 2-5 micrometre filter.

PLANCK_C2_UM_K = 14388.0      # second radiation constant, micrometre*Kelvin
WIEN_C2_UM_K = PLANCK_C2_UM_K # back-compat alias for any older callers
DEFAULT_LAMBDA_EFF_UM = 3.5
EMISSIVITY_CORRECTION_METHOD = "exact_planck_single_wavelength"


def emissivity_correct_kelvin(T_kelvin: np.ndarray,
                              eps_assumed: float,
                              eps_new: float,
                              lambda_eff_um: float = DEFAULT_LAMBDA_EFF_UM
                              ) -> np.ndarray:
    """Exact single-wavelength Planck post-correction from one assumed
    emissivity to another.  `T_kelvin` is the per-pixel temperature the
    FLIR SDK computed under `eps_assumed`; the returned array is the
    per-pixel temperature that the same measured radiance would imply
    if the actual emissivity were `eps_new`.  Reflection ignored.

    For ratios `eps_new/eps_assumed` close to 1 this is numerically
    equivalent to the Wien high-T approximation; for the corners that
    matter to high-T users (large ratio + T > 1500 C) it removes the
    Wien overestimate, which can reach tens of percent."""
    if eps_new <= 0.0 or eps_assumed <= 0.0:
        raise ValueError("emissivity must be > 0")
    T_a = np.asarray(T_kelvin, dtype=np.float64)
    # exp(C2 / (lambda * T)) - 1  is the inverse Planck radiance factor
    # at T_assumed; multiply by (eps_new/eps_assumed) to get the same
    # quantity at T_new.
    inv_band = PLANCK_C2_UM_K / (lambda_eff_um * T_a)
    rhs = (eps_new / eps_assumed) * (np.exp(inv_band) - 1.0)
    # 1 + rhs is strictly positive, log is well-defined.
    inv_band_new = np.log1p(rhs)
    T_new = PLANCK_C2_UM_K / (lambda_eff_um * inv_band_new)
    return T_new.astype(np.float32)


SDK_NATIVE_PROBE_TOLERANCE_K = 0.5


def sdk_emissivity_actually_responds(f, *, best_idx: int,
                                     eps_recorded: float,
                                     H: int, W: int) -> tuple[bool, float]:
    """Detect whether writing to `f.object_parameters.emissivity` and
    re-reading the frame actually produces a different decoded
    temperature -- regardless of what `f.can_change_object_parameters`
    advertises.

    Many science-camera ATS files (X6900sc, A6750sc, ...) report
    `can_change_object_parameters = True` *and* silently no-op the
    setattr, so the only reliable test is empirical: write a sharply
    different emissivity, re-decode the same frame, and check whether
    the per-pixel temperature actually moved.

    Returns `(responds, max_delta_kelvin)`.  `responds = True` iff the
    probe Δ is larger than `SDK_NATIVE_PROBE_TOLERANCE_K` (0.5 K).  The
    emissivity is restored to `eps_recorded` and the cached frame
    decode is left in the recorded-emissivity state."""
    # Baseline at recorded eps.
    f.get_frame(best_idx)
    T_baseline = (np.asarray(f.final, dtype=np.float32)
                  .reshape((H, W)).copy())

    # Pick a probe eps at the opposite end of the (0, 1) range so the
    # decoded temperature MUST change by a large margin if the SDK
    # honours the edit.
    probe_eps = 0.10 if eps_recorded > 0.5 else 0.95

    delta = 0.0
    try:
        f.object_parameters.emissivity = float(probe_eps)
        f.get_frame(best_idx)
        T_probe = (np.asarray(f.final, dtype=np.float32)
                   .reshape((H, W)))
        delta = float(np.abs(T_probe - T_baseline).max())
    except Exception as exc:
        # If the setattr itself raised, the SDK definitely refuses.
        print(f"  [probe] setattr / re-read raised "
              f"{type(exc).__name__}: {exc!r}", flush=True)
        delta = 0.0
    finally:
        # Restore recorded eps and re-decode so callers see a clean
        # recorded-eps frame in the cached buffer.
        try:
            f.object_parameters.emissivity = float(eps_recorded)
        except Exception:
            pass
        f.get_frame(best_idx)

    return (delta > SDK_NATIVE_PROBE_TOLERANCE_K, delta)


# ---------- Test-mode helpers (single-frame multi-emissivity stack) -------
def _find_font(size: int):
    """Best-effort TrueType font; fall back to PIL's tiny bitmap."""
    for cand in ("arial.ttf",
                 r"C:\Windows\Fonts\arial.ttf",
                 r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_label_strip(text: str, width: int, height: int,
                       bg_value: float, fg_value: float) -> np.ndarray:
    """Return a (height, width) float32 strip with `text` rendered as
    `fg_value` pixels on a `bg_value` background, centred."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    font = _find_font(max(10, int(height * 0.65)))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width  - tw) // 2 - bbox[0]
    y = (height - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    mask = np.asarray(img, dtype=np.uint8)
    strip = np.full((height, width), bg_value, dtype=np.float32)
    strip[mask > 128] = fg_value
    return strip


def find_hottest_frame(ats_path: Path) -> tuple[int, float]:
    """Scan the file in Unit.COUNTS (fast uint16 reads) and return
    (frame_index, mean_count) of the page with the highest mean ADC
    count.  Because the count -> temperature mapping is monotonic per
    pixel, the hottest mean-count frame is also the hottest mean-
    temperature frame."""
    f = fnv.file.ImagerFile(str(ats_path))
    f.unit = fnv.Unit.COUNTS
    n = int(f.num_frames)
    H, W = int(f.height), int(f.width)
    best_idx, best_mean = -1, -1.0
    report_step = max(1, n // 5)
    for i in range(n):
        f.get_frame(i)
        m = float(np.asarray(f.final, dtype=np.uint16).reshape((H, W)).mean())
        if m > best_mean:
            best_mean = m
            best_idx = i
        if (i + 1) % report_step == 0 or (i + 1) == n:
            print(f"    scan {i+1}/{n}", flush=True)
    return best_idx, best_mean


# ---------- Sweep summary plots -------------------------------------------
def plot_sweep_results(page_stats: list[dict],
                       lambda_eff_values: list[float],
                       emissivities: list[float],
                       out_dir: Path,
                       stem: str,
                       unit_label: str) -> list[Path]:
    """Draw summary PNGs of per-page T_min / T_max against the swept
    parameter(s) and save them next to the sweep TIFF.

    Layout depends on how many values were swept on each axis:
      - 1 lambda, n eps (or 1 eps, n lambda):  one line plot with two
        curves (T_min, T_max) against the varied parameter.
      - m lambda, n eps (both > 1):  four PNGs are written:
          * stacked subplots, x = eps, one subplot per lambda
          * stacked subplots, x = lambda, one subplot per eps
          * 2-D heatmap of T_max with lambda on the y-axis and eps on
            the x-axis (or vice versa, whichever fits the data better)
          * 2-D heatmap of T_min, same axes
      - 1 lambda, 1 eps: nothing drawn (one number says it all).

    Returns the list of paths actually written.  Returns an empty list
    silently if matplotlib is not installed.
    """
    if not _HAVE_MATPLOTLIB:
        print(f"  [plot] matplotlib not installed -- skipping summary "
              f"plots. `pip install matplotlib` to enable.", flush=True)
        return []

    n_lam = len(lambda_eff_values)
    n_eps = len(emissivities)
    if n_lam == 1 and n_eps == 1:
        return []

    # Build 2-D matrices indexed [i_lambda, j_eps].
    T_min = np.full((n_lam, n_eps), np.nan, dtype=np.float64)
    T_max = np.full((n_lam, n_eps), np.nan, dtype=np.float64)
    for s in page_stats:
        i = lambda_eff_values.index(s["lambda_eff_um"])
        j = emissivities.index(s["emissivity"])
        T_min[i, j] = s["min"]
        T_max[i, j] = s["max"]

    unit_sym = "C" if unit_label == "celsius" else "K"
    written: list[Path] = []

    def _save(fig, suffix: str) -> Path:
        p = out_dir / f"{stem}_plot_{suffix}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
        return p

    # ---- One-dimensional case --------------------------------------
    if n_lam == 1 or n_eps == 1:
        if n_lam == 1:
            xs = emissivities
            x_label = "emissivity"
            y_min = T_min[0]
            y_max = T_max[0]
            other = f"lambda_eff = {lambda_eff_values[0]:.3f} um"
            suffix = "vs_eps"
        else:
            xs = lambda_eff_values
            x_label = "lambda_eff (um)"
            y_min = T_min[:, 0]
            y_max = T_max[:, 0]
            other = f"emissivity = {emissivities[0]:.3f}"
            suffix = "vs_lambda"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, y_min, "o-", color="C0", label="T_min")
        ax.plot(xs, y_max, "o-", color="C1", label="T_max")
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"temperature (deg {unit_sym})")
        ax.set_title(f"{stem}: hottest-frame T vs {x_label}  ({other})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, suffix)
        return written

    # ---- Two-dimensional case --------------------------------------
    # `import` here so the module still loads cleanly when matplotlib
    # is missing; we only reach this branch when _HAVE_MATPLOTLIB.
    import matplotlib.patheffects as path_effects

    def _label_inside(ax, text: str) -> None:
        """Anchor a small param label in the subplot's upper-right
        corner with a translucent white halo so it sits above grid
        lines and any plotted curves."""
        ax.text(
            0.985, 0.94, text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.85,
                      edgecolor="0.6", linewidth=0.5),
        )

    # Stacked-subplot 1: one subplot per lambda, x = eps.
    fig, axes = plt.subplots(
        n_lam, 1,
        figsize=(8, max(3.5, min(28.0, 1.6 * n_lam + 1.5))),
        sharex=True,
    )
    if n_lam == 1:
        axes = [axes]
    for i, lam in enumerate(lambda_eff_values):
        ax = axes[i]
        ax.plot(emissivities, T_min[i], "o-", color="C0", label="T_min")
        ax.plot(emissivities, T_max[i], "o-", color="C1", label="T_max")
        ax.set_ylabel(f"T (deg {unit_sym})")
        ax.grid(True, alpha=0.3)
        # Show x-tick numeric labels on every subplot, not only the
        # bottom one (overrides sharex's default of hiding them).
        ax.tick_params(labelbottom=True)
        _label_inside(ax, f"lambda_eff = {lam:.3f} um")
        if i == 0:
            ax.legend(loc="upper left")
    axes[-1].set_xlabel("emissivity")
    fig.suptitle(
        f"{stem}: hottest-frame T vs emissivity, stacked by lambda_eff",
        fontsize=12,
    )
    _save(fig, "by_eps_stacked")

    # Stacked-subplot 2: one subplot per eps, x = lambda.
    fig, axes = plt.subplots(
        n_eps, 1,
        figsize=(8, max(3.5, min(40.0, 1.4 * n_eps + 1.5))),
        sharex=True,
    )
    if n_eps == 1:
        axes = [axes]
    for j, eps in enumerate(emissivities):
        ax = axes[j]
        ax.plot(lambda_eff_values, T_min[:, j], "o-", color="C0",
                label="T_min")
        ax.plot(lambda_eff_values, T_max[:, j], "o-", color="C1",
                label="T_max")
        ax.set_ylabel(f"T (deg {unit_sym})")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=True)
        _label_inside(ax, f"eps = {eps:.3f}")
        if j == 0:
            ax.legend(loc="upper left")
    axes[-1].set_xlabel("lambda_eff (um)")
    fig.suptitle(
        f"{stem}: hottest-frame T vs lambda_eff, stacked by emissivity",
        fontsize=12,
    )
    _save(fig, "by_lambda_stacked")

    # Heatmaps -- one per T_min and T_max.  x = eps, y = lambda.
    eps_arr = np.asarray(emissivities, dtype=np.float64)
    lam_arr = np.asarray(lambda_eff_values, dtype=np.float64)
    # Build cell edges from the value midpoints so each value sits at
    # the centre of its cell.
    def _edges(v: np.ndarray) -> np.ndarray:
        if len(v) == 1:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        mids = 0.5 * (v[:-1] + v[1:])
        return np.concatenate(
            ([v[0] - (mids[0] - v[0])], mids, [v[-1] + (v[-1] - mids[-1])])
        )
    x_edges = _edges(eps_arr)
    y_edges = _edges(lam_arr)
    # Scale the per-cell number font down a bit when there are lots of
    # cells so they don't visually collide.
    cell_font_size = max(5.5, min(10.0, 60.0 / max(n_eps, 1)))
    for matrix, kind, cmap in (
        (T_max, "T_max", "inferno"),
        (T_min, "T_min", "viridis"),
    ):
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * n_lam + 3)))
        pcm = ax.pcolormesh(x_edges, y_edges, matrix,
                            cmap=cmap, shading="flat")
        ax.set_xlabel("emissivity")
        ax.set_ylabel("lambda_eff (um)")
        ax.set_title(f"{stem}: {kind} (deg {unit_sym}) across the sweep")
        ax.invert_yaxis()  # smaller lambda at the top reads more naturally
        fig.colorbar(pcm, ax=ax, label=f"{kind} (deg {unit_sym})")
        # Overlay the numeric temperature at the centre of every cell.
        # White text with a thin black stroke stays legible on any
        # colormap value -- no need to flip colour per cell.
        for i in range(n_lam):
            for j in range(n_eps):
                v = matrix[i, j]
                if not np.isfinite(v):
                    continue
                txt = ax.text(
                    eps_arr[j], lam_arr[i], f"{v:.0f}",
                    ha="center", va="center",
                    fontsize=cell_font_size,
                    color="white",
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=1.4, foreground="black"),
                    path_effects.Normal(),
                ])
        _save(fig, f"heatmap_{kind.lower()}")

    return written


def test_sweep_one_file(ats_path: Path, out_dir: Path,
                        emissivities: list[float], *,
                        lambda_eff_values: list[float] | None = None,
                        unit: str = "celsius",
                        overwrite: bool = False,
                        crop: tuple[int, int, int, int] | None = None,
                        flip_mode: str = "none",
                        hottest_idx: int | None = None) -> dict:
    """For ONE .ats file:
      1.  Find the hottest frame (mean ADC count) -- one pass over the
          stack -- unless `hottest_idx` was supplied by an earlier scan.
      2.  Open the file at Unit.TEMPERATURE_FACTORY, read the recorded
          emissivity, and runtime-probe whether the SDK actually honours
          live `object_parameters.emissivity` edits (the advertised
          flag is unreliable on X6900sc files -- see README).
      3.  Build the full set of (lambda_eff, emissivity) combinations
          (`product(lambda_eff_values, emissivities)` -- outer loop is
          lambda, inner loop is eps) and for each combination compute
          that page's temperature map using the best available method:
            - sdk_native (probe Δ > 0.5 K): set
              `f.object_parameters.emissivity = eps`, re-read the
              hottest frame, and let the SDK perform its full band-
              integrated radiometric inversion (gold standard).
              `lambda_eff` is not a parameter of this path; if the
              caller supplied a multi-value `lambda_eff_values` list,
              it is collapsed to a single value with a console warning.
            - exact_planck_single_wavelength: read the hottest frame
              once at the recorded eps and post-correct it for each
              (lambda_eff, eps) combination with the closed-form Planck
              inversion.
          The user-chosen crop and flip are applied to every page, and
          a dark label strip with "lambda=X.XX um  eps=Y.YYY" is
          prepended at the TOP (never occludes pixels).
      4.  Stack the resulting pages into a single multi-page float32
          TIFF.  Page index j corresponds to
          (lambda_eff_values[j // n_eps], emissivities[j % n_eps]).
      5.  When matplotlib is installed and the sweep covers more than
          one combination, draw summary plots of per-page T_min/T_max
          against the swept parameter(s) and write them next to the
          TIFF.

    Output (next to each other):
        <stem>_eps_sweep_temp_{C|K}.tif      one page per (lambda, eps)
        <stem>_eps_sweep_meta.json           hottest frame + per-page stats
                                             + which correction method
                                             was used
        <stem>_sweep_plot*.png               summary plot(s) of T vs param
    """
    if lambda_eff_values is None:
        lambda_eff_values = [DEFAULT_LAMBDA_EFF_UM]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ats_path.stem
    suff = "C" if unit == "celsius" else "K"
    out_tif  = out_dir / f"{stem}_eps_sweep_temp_{suff}.tif"
    out_json = out_dir / f"{stem}_eps_sweep_meta.json"
    if not overwrite and out_tif.exists() and out_json.exists():
        return {"status": "skipped",
                "tif_mb": out_tif.stat().st_size / 1e6}

    t0 = time.time()

    if hottest_idx is None:
        print(f"  scanning for hottest frame in {ats_path.name} ...",
              flush=True)
        best_idx, best_mean_count = find_hottest_frame(ats_path)
    else:
        best_idx, best_mean_count = int(hottest_idx), float("nan")
        print(f"  using pre-scanned hottest frame index = {best_idx}",
              flush=True)
    print(f"  hottest frame index = {best_idx}  "
          f"(mean ADC = {best_mean_count:.1f})", flush=True)

    # Open the file at TEMPERATURE_FACTORY and read the recorded
    # emissivity.  Two paths depending on whether the SDK accepts live
    # object_parameters edits on this file:
    #
    #   - can_change_object_parameters == True ("sdk_native" path):
    #     for each test emissivity we set f.object_parameters.emissivity
    #     to that value, re-read the hottest frame, and let the SDK
    #     perform its full band-integrated radiometric inversion using
    #     the camera's factory spectral response.  This is the gold
    #     standard -- no single-wavelength approximation, no lambda_eff
    #     guess -- and is what gets used for all "unlocked" ATS files.
    #
    #   - can_change_object_parameters == False ("exact_planck" path):
    #     the SDK refuses live emissivity edits and Unit.TEMPERATURE_USER
    #     is unavailable, so we read the hottest frame once at the
    #     recorded emissivity and post-correct it to every requested
    #     emissivity using the closed-form exact Planck inversion at
    #     DEFAULT_LAMBDA_EFF_UM.  Accurate to a few percent for moderate
    #     emissivity ratios; see README.
    f = fnv.file.ImagerFile(str(ats_path))
    f.unit = fnv.Unit.TEMPERATURE_FACTORY
    f.get_frame(best_idx)
    H, W = int(f.height), int(f.width)
    eps_recorded = float(f.object_parameters.emissivity)
    sdk_can_change_claimed = bool(f.can_change_object_parameters)

    # Trust nothing -- empirically probe whether setting a new
    # emissivity actually moves the decoded frame.  Some science-camera
    # ATS files (X6900sc, A6750sc, ...) advertise can_change=True yet
    # silently no-op the setattr.
    print(f"  recorded emissivity in .ats = {eps_recorded:.4f}  "
          f"(SDK can_change_object_parameters = {sdk_can_change_claimed})",
          flush=True)
    sdk_responds, probe_delta = sdk_emissivity_actually_responds(
        f, best_idx=best_idx, eps_recorded=eps_recorded, H=H, W=W
    ) if sdk_can_change_claimed else (False, 0.0)
    T_recorded_K = np.asarray(f.final, dtype=np.float32).reshape((H, W))

    if sdk_responds:
        correction_method = "sdk_native"
        print(f"  probe: setting eps to a sharply different value moved "
              f"the decoded frame by {probe_delta:.1f} K -- SDK live "
              f"edits work.  Using the camera's band-integrated "
              f"radiometric inversion (no single-wavelength "
              f"approximation, no lambda_eff guess).", flush=True)
    else:
        correction_method = EMISSIVITY_CORRECTION_METHOD
        if sdk_can_change_claimed:
            print(f"  probe: setting eps to a sharply different value moved "
                  f"the decoded frame by only {probe_delta:.3g} K "
                  f"(< {SDK_NATIVE_PROBE_TOLERANCE_K} K threshold) -- "
                  f"the SDK ADVERTISES can_change=True but is silently "
                  f"ignoring the setattr.  Falling back to exact-Planck "
                  f"post-correction at lambda_eff = "
                  f"{DEFAULT_LAMBDA_EFF_UM} um.", flush=True)
        else:
            print(f"  SDK does not allow live emissivity changes for "
                  f"this file -- using exact-Planck post-correction at "
                  f"lambda_eff = {DEFAULT_LAMBDA_EFF_UM} um", flush=True)

    # SDK-native path doesn't take lambda_eff -- collapse any multi-
    # value request to a single sentinel so the loop and the post-sweep
    # plotting don't generate identical pages.
    if correction_method == "sdk_native" and len(lambda_eff_values) > 1:
        print(f"  [info] SDK-native path is active -- lambda_eff is not a "
              f"parameter of the camera's own radiometric inversion, so "
              f"the {len(lambda_eff_values)} requested values would "
              f"produce identical pages.  Collapsing lambda_eff_values "
              f"to [{lambda_eff_values[0]:.3f}] for the sweep.",
              flush=True)
        lambda_eff_values = [lambda_eff_values[0]]

    n_lam = len(lambda_eff_values)
    n_eps = len(emissivities)
    n_pages = n_lam * n_eps

    pages, page_stats = [], []
    unit_sym = "deg C" if unit == "celsius" else "K"
    print(f"  building {n_pages} page(s) "
          f"(n_lambda={n_lam} x n_eps={n_eps}, "
          f"crop={crop}, flip={flip_mode}, "
          f"method={correction_method}) ...", flush=True)
    j = 0
    for lam in lambda_eff_values:
        for eps in emissivities:
            j += 1
            if abs(eps - eps_recorded) < 1e-9:
                # Emissivity unchanged -- lambda_eff is mathematically
                # irrelevant (the closed-form Planck formula collapses
                # to T_new = T_assumed regardless of lambda).
                T_new_K = T_recorded_K
            elif correction_method == "sdk_native":
                # Ask the SDK to recompute under the new emissivity
                # using its own factory band-integrated inversion.
                f.object_parameters.emissivity = float(eps)
                f.get_frame(best_idx)
                T_new_K = (np.asarray(f.final, dtype=np.float32)
                           .reshape((H, W)))
            else:
                T_new_K = emissivity_correct_kelvin(
                    T_recorded_K, eps_recorded, float(eps),
                    lambda_eff_um=float(lam),
                )
            page = (T_new_K
                    - (KELVIN_OFFSET if unit == "celsius" else 0.0)
                    ).astype(np.float32)
            # Crop first, then flip: same order the batch path uses, so
            # the frame the user sees in Fiji matches what batch-mode
            # TIFFs will look like for the same camera and the same
            # chosen settings.
            page = apply_flip(apply_crop(page, crop), flip_mode)
            pages.append(page)
            page_stats.append({
                "page_index": j - 1,
                "lambda_eff_um": float(lam),
                "emissivity": float(eps),
                "min": float(page.min()),
                "max": float(page.max()),
                "mean": float(page.mean()),
            })
            print(f"    [{j}/{n_pages}] lambda={lam:.3f} um "
                  f"eps={eps:.3f}  min/mean/max = "
                  f"{page.min():.1f} / {page.mean():.1f} / "
                  f"{page.max():.1f} {unit_sym}", flush=True)

    # Pick values that make the label strip a clear dark bar with bright
    # white text under whatever auto-contrast Fiji applies to the stack.
    gmin = float(min(p.min() for p in pages))
    gmax = float(max(p.max() for p in pages))
    span = max(gmax - gmin, 1.0)
    bg_val = gmin - 0.05 * span - 1.0
    fg_val = gmax + 0.05 * span + 1.0
    strip_h = 28
    out_H, out_W = pages[0].shape[:2]

    print(f"  writing {out_tif.name}  "
          f"({len(pages)} pages, {strip_h + out_H}x{out_W} float32, "
          f"label on top) ...", flush=True)
    with tifffile.TiffWriter(str(out_tif), bigtiff=False) as tw:
        for page, stat in zip(pages, page_stats):
            lam = stat["lambda_eff_um"]
            eps = stat["emissivity"]
            if n_lam > 1:
                label = f"lambda={lam:.2f} um  eps={eps:.3f}"
            else:
                label = f"emissivity = {eps:.3f}"
            strip = render_label_strip(label, out_W, strip_h,
                                       bg_val, fg_val)
            # Label strip goes ABOVE the picture so the burnt-in text
            # never occludes any pixel of the real scene.
            page_with_label = np.concatenate([strip, page], axis=0)
            tw.write(
                page_with_label, photometric="minisblack",
                compression="zlib", compressionargs={"level": ZLIB_LEVEL},
                contiguous=False,
            )

    # Sidecar JSON
    probe = fnv.file.ImagerFile(str(ats_path))
    meta = {
        "source_file": str(ats_path),
        "hottest_frame_index": int(best_idx),
        "hottest_frame_mean_adc_count": float(best_mean_count),
        "test_emissivity_values": [float(e) for e in emissivities],
        "test_lambda_eff_um_values": [float(l) for l in lambda_eff_values],
        "page_order": (
            "outer = lambda_eff_um, inner = emissivity; "
            "page_index = i_lambda * n_eps + i_eps"
        ),
        "pixel_unit_written": "celsius" if unit == "celsius" else "kelvin",
        "data_type_in_tiff": "float32",
        "source_height": int(H),
        "source_width":  int(W),
        "output_height_per_page": int(out_H),
        "output_width_per_page":  int(out_W),
        "crop_x0_x1_y0_y1_zero_based_half_open": (
            list(crop) if crop is not None else None),
        "flip_mode_after_crop": flip_mode,
        "label_strip_height": int(strip_h),
        "label_strip_position": "top",
        "label_strip_bg_value": float(bg_val),
        "label_strip_fg_value": float(fg_val),
        "tiff_layout": (
            f"Each page is the hottest frame (index {best_idx}) of the "
            "source .ats, decoded in Unit.TEMPERATURE_FACTORY at the "
            "(lambda_eff_um, emissivity) combination recorded in "
            f"per_page_summary[page] (recorded eps "
            f"= {eps_recorded:.4f}; correction method = "
            f"{correction_method!r}), cropped to {crop} and flipped "
            f"({flip_mode}). All other object_parameters were left at "
            f"their recorded values.  A {strip_h}-row label strip is "
            "PREPENDED at the top of every page (white text on a dark "
            f"background); the real scene area is the bottom {out_H} "
            "rows."
        ),
        "emissivity_correction": (
            {
                "method": "sdk_native",
                "explanation": (
                    "FLIR Science File SDK 2026.1.2 reported "
                    "can_change_object_parameters = True for this file, "
                    "so every test page was produced by setting "
                    "f.object_parameters.emissivity = "
                    "test_emissivity_values[page] and re-reading the "
                    "hottest frame in Unit.TEMPERATURE_FACTORY.  The "
                    "SDK then performs its full band-integrated Planck "
                    "inversion using the camera's factory spectral "
                    "response curve -- no single-wavelength "
                    "approximation, no lambda_eff guess."
                ),
                "eps_recorded_in_ats": float(eps_recorded),
            }
            if correction_method == "sdk_native"
            else {
                "method": EMISSIVITY_CORRECTION_METHOD,
                "formula": (
                    "T_new = C2 / (lambda_eff * "
                    "ln(1 + (eps_new/eps_assumed) "
                    "* (exp(C2/(lambda_eff*T_assumed)) - 1)))"
                ),
                "C2_um_K": PLANCK_C2_UM_K,
                "lambda_eff_um_values": [float(l) for l in lambda_eff_values],
                "eps_assumed_at_decode_time": float(eps_recorded),
                "reflection_term": (
                    "omitted (valid when scene T >> reflected_T)"),
                "approximation_note": (
                    "Exact single-wavelength Planck inversion -- no "
                    "Wien high-T approximation, so accurate at "
                    "T > 1500 C where Wien would overestimate by "
                    "tens of percent."
                ),
                "sdk_constraint": (
                    "FLIR Science File SDK 2026.1.2 reports "
                    "can_change_object_parameters = False for this "
                    "file, and Unit.TEMPERATURE_USER raises 'failed "
                    "to set unit'; live SDK re-computation under a "
                    "different emissivity is therefore not "
                    "available.  This post-correction is the "
                    "practical alternative."),
            }
        ),
        "per_page_summary": page_stats,
        "shared_object_parameters_recorded": _jsonable(probe.object_parameters),
        "source_info": _jsonable(probe.source_info),
        "preset_info": _jsonable(list(probe.source_info.preset_info)),
    }
    out_json.write_text(json.dumps(meta, indent=2, default=str))

    # ---- Summary plots ----------------------------------------------------
    plot_paths = plot_sweep_results(
        page_stats=page_stats,
        lambda_eff_values=list(lambda_eff_values),
        emissivities=list(emissivities),
        out_dir=out_dir,
        stem=f"{stem}_eps_sweep",
        unit_label=("celsius" if unit == "celsius" else "kelvin"),
    )
    for p in plot_paths:
        print(f"  [plot] wrote {p.name}", flush=True)

    return {
        "status": "ok",
        "tif_mb": out_tif.stat().st_size / 1e6,
        "n_pages": len(pages),
        "n_plots": len(plot_paths),
        "hottest_frame": int(best_idx),
        "elapsed_s": time.time() - t0,
    }


# ---------- Mode prompt and dispatcher helpers ----------------------------
def prompt_mode(default: str | None = None) -> str:
    """Return 'test' or 'batch'.  Empty input accepts `default` if given."""
    print("\nWhat would you like to do?")
    print("  [1] Test mode  -- sweep emissivity values on ONE .ats file")
    print("  [2] Batch mode -- convert MANY .ats files with shared parameters")
    tag = f"  [Enter = last: {default}]" if default in ("test", "batch") else ""
    while True:
        ans = input(f"Choice [1/2]{tag}: ").strip().lower()
        if not ans and default in ("test", "batch"):
            return default
        if ans in ("1", "t", "test"):
            return "test"
        if ans in ("2", "b", "batch"):
            return "batch"
        print("  please type 1, 2, t, or b")


def prompt_post_test_action() -> str:
    """After a test sweep finishes: 'test', 'batch', or 'exit'."""
    print("\nWhat would you like to do next?")
    print("  [t] another test sweep (pick a file + emissivity values)")
    print("  [b] switch to batch mode and convert many files")
    print("  [e] exit the program")
    while True:
        ans = input("Choice [t/b/e]: ").strip().lower()
        if ans in ("t", "test", "1"):
            return "test"
        if ans in ("b", "batch", "2"):
            return "batch"
        if ans in ("e", "exit", "q", "quit", "3"):
            return "exit"
        print("  please type t, b, or e")


# ---------- Batch driver --------------------------------------------------
def _prompt_dir(prompt: str, default: Path | None = None) -> Path | None:
    if default is not None:
        prompt = f"{prompt}\n  [Enter = last: {default}]: "
    else:
        prompt = f"{prompt}: "
    raw = input(prompt).strip().strip('"').strip("'")
    if not raw:
        return default
    return Path(raw)


def _resolve_dirs(args, state: dict) -> tuple[Path, Path]:
    """CLI args or interactive prompts -> (input_dir, output_dir).  Exits
    on missing or invalid input dir.  Reads `state['input_dir']` /
    `state['output_dir']` as the Enter-default for each prompt."""
    last_in = Path(state["input_dir"]) if state.get("input_dir") else None
    last_out = Path(state["output_dir"]) if state.get("output_dir") else None
    inp = args.input or _prompt_dir(
        "Input folder (contains .ats files)", default=last_in)
    out = args.output or _prompt_dir(
        "Output folder for .tif + .json + .png", default=last_out)
    if inp is None or not inp.is_dir():
        print(f"ERROR: input folder not valid: {inp}", file=sys.stderr)
        sys.exit(1)
    if out is None:
        print("ERROR: output folder is required", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    return inp, out


def _scan_ats(args, input_dir: Path) -> list[Path]:
    pattern = input_dir.glob if args.no_recurse else input_dir.rglob
    return sorted(pattern("*.ats"))


# ---------- Test mode -----------------------------------------------------
def test_mode(args, input_dir: Path, output_dir: Path,
              all_ats_files: list[Path], state: dict) -> str:
    """Iterative emissivity-sweep loop.  Each round picks ONE .ats file
    and a list of emissivity values, finds the hottest frame in that
    file, and writes a single multi-page TIFF where each page is the
    hottest frame re-decoded with one emissivity (a 'emissivity = X.XXX'
    white-on-dark label strip is burnt into the bottom of each page).

    Returns 'batch' to continue into batch mode, or 'exit' to stop."""
    print()
    print("=== Test mode (emissivity sweep) ===")
    print(f"  input dir:  {input_dir}")
    print(f"  output dir: {output_dir}")
    print(f"  unit:       {args.unit}")
    print("  Per round: one .ats file -> one stack TIFF, one page per")
    print("  emissivity value, hottest frame only.  All other object")
    print("  parameters stay at each file's recorded values.")

    while True:
        ats = prompt_single_file(all_ats_files, input_dir,
                                 default_name=state.get("test_file_name"))
        state["test_file_name"] = _display_name(ats, input_dir)
        save_state(state)

        eps_list, eps_spec = prompt_emissivity_values(
            default=state.get("emissivity_spec"))
        state["emissivity_spec"] = eps_spec
        save_state(state)

        lambda_list, lambda_spec = prompt_lambda_eff_values(
            default=state.get("lambda_eff_spec"))
        state["lambda_eff_spec"] = lambda_spec
        save_state(state)

        # One scan to find the hottest frame -- its data drives the crop
        # preview AND is fed straight into the sweep, so we never scan
        # the same file twice in a single round.
        print(f"\n[test] scanning {ats.name} for the hottest frame "
              f"(crop-preview source) ...", flush=True)
        try:
            hottest_idx, hot_frame = _read_hottest_frame_counts(ats)
        except Exception as exc:
            print(f"  !!! could not scan {ats.name}: {exc!r}", flush=True)
            action = prompt_post_test_action()
            if action == "test":
                continue
            return action

        crop, crop_spec = prompt_crop_range(
            hot_frame, output_dir,
            default_spec=state.get("crop_spec") or None)
        state["crop_spec"] = crop_spec
        save_state(state)

        flip_mode = prompt_flip_mode(default=state.get("flip_mode"))
        state["flip_mode"] = flip_mode
        save_state(state)

        print(f"\n[test] sweep on {ats.name} with "
              f"{len(eps_list)} emissivity value(s) x "
              f"{len(lambda_list)} lambda_eff value(s) = "
              f"{len(eps_list) * len(lambda_list)} page(s), "
              f"crop={crop}, flip={flip_mode}", flush=True)
        try:
            # Test mode always overwrites: users iterate parameter choices
            # and expect the latest values to land on disk, not a stale
            # output from an earlier round to be silently kept.
            r = test_sweep_one_file(ats, output_dir, eps_list,
                                    lambda_eff_values=lambda_list,
                                    unit=args.unit,
                                    overwrite=True,
                                    crop=crop,
                                    flip_mode=flip_mode,
                                    hottest_idx=hottest_idx)
            print(f"  -> {r['status']}  "
                  f"({r.get('n_pages', 0)} pages, "
                  f"{r.get('tif_mb', 0):.2f} MB, "
                  f"hottest frame = {r.get('hottest_frame', '?')}, "
                  f"{r.get('elapsed_s', 0):.1f} s)",
                  flush=True)
        except KeyboardInterrupt:
            print("\n[interrupt] aborting current sweep")
        except Exception as exc:
            print(f"  !!! ERROR: {type(exc).__name__}: {exc}", flush=True)

        action = prompt_post_test_action()
        if action == "test":
            continue
        return action  # 'batch' or 'exit'


# ---------- Batch mode ----------------------------------------------------
def batch_mode(args, input_dir: Path, output_dir: Path,
               all_ats_files: list[Path], state: dict) -> None:
    """The original whole-folder conversion path."""
    # ---- Pick a subset of files ----------------------------------------
    if args.files is not None:
        try:
            idxs = parse_file_selection(args.files, len(all_ats_files))
        except ValueError as exc:
            print(f"ERROR: --files {args.files!r}: {exc}", file=sys.stderr)
            sys.exit(1)
        if not idxs:
            print(f"ERROR: --files {args.files!r} matched no files",
                  file=sys.stderr)
            sys.exit(1)
        ats_files = [all_ats_files[i - 1] for i in idxs]
        state["file_selection"] = args.files
        save_state(state)
    elif args.no_confirm:
        ats_files = all_ats_files
    else:
        ats_files = prompt_file_selection(
            all_ats_files, input_dir,
            default_selection=state.get("file_selection"))
        # The interactive prompt re-parses the user's raw string for us,
        # but we don't get the raw back -- reconstruct a faithful spec by
        # collapsing the picked indices, so press-Enter next round.
        picked_idx = sorted(all_ats_files.index(p) + 1 for p in ats_files)
        state["file_selection"] = (
            "all" if picked_idx == list(range(1, len(all_ats_files) + 1))
            else _compress_indices(picked_idx))
        save_state(state)

    print()
    print(f"[batch] {len(ats_files)} of {len(all_ats_files)} .ats files "
          f"selected for conversion  "
          f"(unit={args.unit}, "
          f"{'overwrite' if args.overwrite else 'skip existing'})")

    # ---- Inspect / override radiometric parameters ---------------------
    overrides: dict[str, float] = {}
    if args.no_confirm:
        print("[batch] --no-confirm: using recorded object_parameters verbatim")
    else:
        print(f"\n[batch] Inspecting recorded radiometric parameters from "
              f"the first file ({ats_files[0].name}) ...")
        try:
            probe = fnv.file.ImagerFile(str(ats_files[0]))
            initial = read_object_params(probe)
            del probe
        except Exception as exc:
            print(f"  [warn] could not read object_parameters from first "
                  f"file: {exc!r}; falling back to per-file recorded values")
            initial = None

        if initial is not None:
            final = prompt_object_param_overrides(
                initial, last_overrides=state.get("object_overrides"))
            overrides = {k: v for k, v in final.items()
                         if abs(v - initial[k]) > 1e-9}
            state["object_overrides"] = overrides
            save_state(state)
            if overrides:
                print(f"\n[batch] {len(overrides)} parameter(s) will be "
                      f"overridden on every file:")
                for k, v in overrides.items():
                    print(f"          {k} = {v}  (was {initial[k]})")
            else:
                print("\n[batch] no overrides -- using each file's recorded "
                      "values verbatim")

    # ---- Crop + flip prompts (shared across the batch) ----------------
    crop: tuple[int, int, int, int] | None = None
    flip_mode = state.get("flip_mode") or DEFAULT_FLIP_MODE
    if args.no_confirm:
        print(f"[batch] --no-confirm: no crop, flip={flip_mode} "
              f"(remembered default)")
    else:
        print(f"\n[batch] scanning {ats_files[0].name} for the hottest "
              f"frame (crop-preview source) ...")
        try:
            _, hot_frame = _read_hottest_frame_counts(ats_files[0])
            crop, crop_spec = prompt_crop_range(
                hot_frame, output_dir,
                default_spec=state.get("crop_spec") or None)
            state["crop_spec"] = crop_spec
            save_state(state)
        except Exception as exc:
            print(f"  [warn] crop-preview scan failed: {exc!r}; "
                  "falling back to no crop")
            crop = None

        flip_mode = prompt_flip_mode(default=state.get("flip_mode"))
        state["flip_mode"] = flip_mode
        save_state(state)

    print()
    results = []
    batch_t0 = time.time()
    for i, ats in enumerate(ats_files, 1):
        rel = (ats.relative_to(input_dir)
               if input_dir in ats.parents else ats.name)
        print(f"=== [{i}/{len(ats_files)}] {rel} ===", flush=True)
        try:
            r = convert_one(ats, output_dir, unit=args.unit,
                            overrides=overrides if overrides else None,
                            overwrite=args.overwrite,
                            crop=crop,
                            flip_mode=flip_mode)
            r["path"] = str(ats)
            print(f"  -> {r['status']}", flush=True)
            results.append(r)
        except KeyboardInterrupt:
            print("\n[interrupt] stopping batch")
            break
        except Exception as exc:
            print(f"  !!! ERROR: {type(exc).__name__}: {exc}", flush=True)
            results.append({"path": str(ats), "status": "error",
                            "error": repr(exc)})

    print()
    elapsed_min = (time.time() - batch_t0) / 60
    ok      = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = sum(1 for r in results if r["status"] == "error")
    print(f"[done] {len(results)} files in {elapsed_min:.1f} min  "
          f"(ok={ok}, skipped={skipped}, error={errors})")


# ---------- main ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",  type=Path,
                    help="Folder containing .ats files (prompts if omitted)")
    ap.add_argument("--output", type=Path,
                    help="Folder for .tif/.json/.png outputs (prompts if omitted)")
    ap.add_argument("--unit", choices=("celsius", "kelvin"), default="celsius",
                    help="Temperature unit written into the TIFF (default celsius)")
    ap.add_argument("--mode", choices=("test", "batch"), default=None,
                    help="Skip the interactive mode prompt and go straight "
                         "to 'test' or 'batch'.")
    ap.add_argument("--no-recurse", action="store_true",
                    help="Only scan the top level of --input; default is recursive")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert files whose outputs already exist")
    ap.add_argument("--no-confirm", action="store_true",
                    help="Skip the interactive object-parameter "
                         "inspection/override prompt AND the file-selection "
                         "prompt; use the values recorded inside each .ats "
                         "verbatim and process every file found")
    ap.add_argument("--files", type=str, default=None,
                    help="Non-interactive file selection.  Same syntax as the "
                         "interactive prompt: 'all', '1-10', '1 3 5', "
                         "'1-5 10 15-20'.  1-based indices into the sorted "
                         "list of .ats files under --input.")
    args = ap.parse_args()

    print(f"[setup] FLIR SDK loaded: {fnv.__file__}")

    # --- Session state (Enter-default for every interactive prompt) -----
    state = load_state()
    if state:
        print(f"[state] remembered defaults from {STATE_FILE}")
        for k in ("mode", "input_dir", "output_dir", "file_selection",
                  "test_file_name", "emissivity_spec", "lambda_eff_spec",
                  "crop_spec", "flip_mode"):
            if k in state:
                print(f"          {k:<16} = {state[k]}")
        if state.get("object_overrides"):
            print(f"          object_overrides = "
                  f"{dict(state['object_overrides'])}")
    else:
        print(f"[state] no previous defaults found ({STATE_FILE})")

    # --- Mode dispatch ---------------------------------------------------
    if args.mode is not None:
        mode = args.mode
    elif args.no_confirm or args.files is not None:
        # Non-interactive shortcuts imply batch mode
        mode = "batch"
    else:
        mode = prompt_mode(default=state.get("mode"))
    state["mode"] = mode
    save_state(state)

    # Folders + .ats listing (shared by both modes)
    input_dir, output_dir = _resolve_dirs(args, state)
    state["input_dir"]  = str(input_dir)
    state["output_dir"] = str(output_dir)
    save_state(state)

    all_ats_files = _scan_ats(args, input_dir)
    if not all_ats_files:
        print(f"No .ats files found under {input_dir}")
        sys.exit(0)

    print(f"[setup] input:  {input_dir}")
    print(f"[setup] output: {output_dir}")
    print(f"[setup] found {len(all_ats_files)} .ats file(s)")

    if mode == "test":
        next_action = test_mode(args, input_dir, output_dir,
                                all_ats_files, state)
        if next_action == "batch":
            batch_mode(args, input_dir, output_dir, all_ats_files, state)
        # 'exit' falls through to end-of-program
        return

    # mode == "batch"
    batch_mode(args, input_dir, output_dir, all_ats_files, state)


if __name__ == "__main__":
    main()
