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
    Test mode is for parameter-sensitivity studies: pick one .ats, then
    type the emissivity values to try (either a range '[start, end,
    step]' or a discrete list '0.3 0.5 0.7') and the script writes one
    full conversion per emissivity value, with the value embedded in the
    output filename (e.g. Rec-000548_eps0.300_temp_C.tif).  After each
    sweep you can run another, hand off to batch mode, or quit.  Batch
    mode is the original whole-folder converter; it never loops back.

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
    from PIL import Image
except ImportError as exc:
    print(
        f"ERROR: missing scientific package ({exc.name}).\n"
        "  Install with:  pip install numpy tifffile pillow",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------- Tunable defaults -----------------------------------------------
PREVIEW_PCT_LO          = 1       # percentile for 16-bit -> 8-bit preview
PREVIEW_PCT_HI          = 99
FALLBACK_PREVIEW_FRAME  = 700
ZLIB_LEVEL              = 5       # tiff compression level (1=fast, 9=best)
KELVIN_OFFSET           = 273.15


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


def prompt_object_param_overrides(initial: dict[str, float]) -> dict[str, float]:
    """Display the recorded parameters and let the user override any of
    them.  Returns the final dict (initial + overrides).  Loops until the
    user confirms."""
    while True:
        print_object_params(initial, "Radiometric inversion parameters "
                            "(recorded inside the first .ats):")
        print("\n  These were applied during the recording.  Press Enter at "
              "each prompt to keep the recorded value, or type a new number.")
        modify = input("\n  Do you want to override any of them? [y/N]: "
                       ).strip().lower()
        if modify not in ("y", "yes"):
            return dict(initial)

        new_vals = dict(initial)
        for name, unit, desc in OBJECT_PARAM_FIELDS:
            cur = new_vals[name]
            raw = input(f"    {name} [{cur:.4f} {unit}]: ").strip()
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
    """Write each override into f.object_parameters before frame reads."""
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
                stem_suffix: str = "") -> dict:
    """Convert one .ats into _temp_{C|K}.tif + _meta.json + _preview.png.

    `stem_suffix` is appended to the output stem (used by test mode to
    label files with the emissivity value they were generated with)."""
    if unit not in ("celsius", "kelvin"):
        raise ValueError(f"unit must be 'celsius' or 'kelvin', got {unit!r}")
    unit_suffix = "C" if unit == "celsius" else "K"
    unit_label  = "celsius" if unit == "celsius" else "kelvin"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem      = ats_path.stem + stem_suffix
    out_tif   = out_dir / f"{stem}_temp_{unit_suffix}.tif"
    out_json  = out_dir / f"{stem}_meta.json"
    out_png   = out_dir / f"{stem}_preview.png"

    if (not overwrite
        and out_tif.exists() and out_json.exists() and out_png.exists()):
        return {"status": "skipped",
                "tif_gb": out_tif.stat().st_size / 1e9}

    t0 = time.time()
    f = fnv.file.ImagerFile(str(ats_path))

    # Snapshot the recorded object_parameters before any override, so the
    # JSON can show both what was in the .ats and what we actually used.
    recorded = read_object_params(f)

    # Apply user-supplied object-parameter overrides BEFORE switching
    # to a temperature unit, so the SDK's internal radiometric model
    # uses the new values.
    if overrides:
        apply_overrides(f, overrides)

    # SDK gives us float32 Kelvin in TEMPERATURE_FACTORY; we shift to
    # Celsius below if asked.  This is the radiometric path that uses
    # the camera's factory calibration plus the (possibly overridden)
    # object parameters (emissivity, distance, reflected temp, etc.).
    f.unit = fnv.Unit.TEMPERATURE_FACTORY

    n = int(f.num_frames)
    H, W = int(f.height), int(f.width)

    print(f"  camera = {f.source_info.camera}  "
          f"serial = {f.source_info.camera_serial}", flush=True)
    print(f"  {n} frames @ {f.source_info.preset_info[0].frame_rate:.1f} fps  "
          f"{H}x{W}  payload (float32) ~ {n*H*W*4/1e9:.2f} GB", flush=True)
    print(f"  writing temperature in {unit_label}", flush=True)

    # ---- Track the hottest frame (max mean temperature) -----------------
    best_mean: float = -float("inf")
    best_idx:  int   = -1
    best_frame: np.ndarray | None = None

    offset = -KELVIN_OFFSET if unit == "celsius" else 0.0
    report_step = max(1, n // 10)

    with tifffile.TiffWriter(str(out_tif), bigtiff=True) as tw:
        for i in range(n):
            f.get_frame(i)
            page = (np.asarray(f.final, dtype=np.float32)
                    .reshape((H, W)) + offset)
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
        best_frame = (np.asarray(f.final, dtype=np.float32)
                      .reshape((H, W)) + offset)
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


def _display_name(p: Path, input_dir: Path) -> str:
    try:
        return str(p.relative_to(input_dir))
    except ValueError:
        return p.name


def prompt_file_selection(ats_files: list[Path],
                          input_dir: Path) -> list[Path]:
    """List the .ats files with 1-based indices and let the user pick a
    subset.  Re-prompts on invalid input or rejected confirmation."""
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
        raw = input("Selection: ").strip()
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


def prompt_single_file(ats_files: list[Path], input_dir: Path) -> Path:
    """List the files with 1-based indices and let the user pick exactly
    one for the test-mode emissivity sweep."""
    n = len(ats_files)
    width = len(str(n))
    while True:
        print(f"\nFound {n} .ats files under {input_dir}:")
        for i, p in enumerate(ats_files, 1):
            sz_gb = p.stat().st_size / 1e9
            print(f"  {i:>{width}}  {_display_name(p, input_dir):<40s}  "
                  f"({sz_gb:5.2f} GB)")
        raw = input("Pick ONE file by its number: ").strip()
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


def prompt_emissivity_values() -> list[float]:
    """Get the list of emissivity values to test from the user.  Repeats
    on parse error until accepted."""
    print("\nEmissivity values to test")
    print("  range form:  [start, end, step]   e.g.  [0.3, 0.9, 0.1]")
    print("  list form:   0.3 0.5 0.7   or   0.3,0.5,0.7")
    while True:
        raw = input("Emissivities: ").strip()
        try:
            vals = parse_emissivity_values(raw)
        except ValueError as exc:
            print(f"  invalid input: {exc}, try again")
            continue
        print(f"  -> {len(vals)} value(s): "
              f"{', '.join(f'{v:.3f}' for v in vals)}")
        ans = input("Proceed? [y/N/edit]: ").strip().lower()
        if ans in ("y", "yes"):
            return vals
        if ans in ("e", "edit"):
            continue
        print("  cancelled, re-entering")


# ---------- Mode prompt and dispatcher helpers ----------------------------
def prompt_mode() -> str:
    """Return 'test' or 'batch'."""
    print("\nWhat would you like to do?")
    print("  [1] Test mode  -- sweep emissivity values on ONE .ats file")
    print("  [2] Batch mode -- convert MANY .ats files with shared parameters")
    while True:
        ans = input("Choice [1/2]: ").strip().lower()
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
def _prompt_dir(prompt: str) -> Path | None:
    raw = input(prompt).strip().strip('"').strip("'")
    return Path(raw) if raw else None


def _resolve_dirs(args) -> tuple[Path, Path]:
    """CLI args or interactive prompts -> (input_dir, output_dir).  Exits
    on missing or invalid input dir."""
    inp = args.input or _prompt_dir("Input folder (contains .ats files): ")
    out = args.output or _prompt_dir("Output folder for .tif + .json + .png: ")
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
              all_ats_files: list[Path]) -> str:
    """Iterative emissivity-sweep loop on a single .ats per round.
    Returns 'batch' to continue into batch mode, or 'exit' to stop."""
    print()
    print("=== Test mode (emissivity sweep) ===")
    print(f"  input dir:  {input_dir}")
    print(f"  output dir: {output_dir}")
    print(f"  unit:       {args.unit}")
    print("  Other radiometric parameters stay at each file's recorded "
          "values; only emissivity is varied per output file.")

    while True:
        ats = prompt_single_file(all_ats_files, input_dir)
        eps_list = prompt_emissivity_values()

        print(f"\n[test] {len(eps_list)} sweep step(s) on {ats.name}",
              flush=True)
        sweep_t0 = time.time()
        for j, eps in enumerate(eps_list, 1):
            suffix = f"_eps{eps:.3f}"
            print(f"--- [{j}/{len(eps_list)}] emissivity = {eps:.3f} "
                  f"(suffix={suffix}) ---", flush=True)
            try:
                r = convert_one(ats, output_dir,
                                unit=args.unit,
                                overrides={"emissivity": eps},
                                overwrite=args.overwrite,
                                stem_suffix=suffix)
                print(f"  -> {r['status']}", flush=True)
            except KeyboardInterrupt:
                print("\n[interrupt] aborting current sweep")
                break
            except Exception as exc:
                print(f"  !!! ERROR: {type(exc).__name__}: {exc}",
                      flush=True)
        print(f"\n[test] sweep finished in "
              f"{(time.time()-sweep_t0)/60:.1f} min", flush=True)

        action = prompt_post_test_action()
        if action == "test":
            continue
        return action  # 'batch' or 'exit'


# ---------- Batch mode ----------------------------------------------------
def batch_mode(args, input_dir: Path, output_dir: Path,
               all_ats_files: list[Path]) -> None:
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
    elif args.no_confirm:
        ats_files = all_ats_files
    else:
        ats_files = prompt_file_selection(all_ats_files, input_dir)

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
            final = prompt_object_param_overrides(initial)
            overrides = {k: v for k, v in final.items()
                         if abs(v - initial[k]) > 1e-9}
            if overrides:
                print(f"\n[batch] {len(overrides)} parameter(s) will be "
                      f"overridden on every file:")
                for k, v in overrides.items():
                    print(f"          {k} = {v}  (was {initial[k]})")
            else:
                print("\n[batch] no overrides -- using each file's recorded "
                      "values verbatim")

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
                            overwrite=args.overwrite)
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

    # --- Mode dispatch ---------------------------------------------------
    if args.mode is not None:
        mode = args.mode
    elif args.no_confirm or args.files is not None:
        # Non-interactive shortcuts imply batch mode
        mode = "batch"
    else:
        mode = prompt_mode()

    # Folders + .ats listing (shared by both modes)
    input_dir, output_dir = _resolve_dirs(args)
    all_ats_files = _scan_ats(args, input_dir)
    if not all_ats_files:
        print(f"No .ats files found under {input_dir}")
        sys.exit(0)

    print(f"[setup] input:  {input_dir}")
    print(f"[setup] output: {output_dir}")
    print(f"[setup] found {len(all_ats_files)} .ats file(s)")

    if mode == "test":
        next_action = test_mode(args, input_dir, output_dir, all_ats_files)
        if next_action == "batch":
            batch_mode(args, input_dir, output_dir, all_ats_files)
        # 'exit' falls through to end-of-program
        return

    # mode == "batch"
    batch_mode(args, input_dir, output_dir, all_ats_files)


if __name__ == "__main__":
    main()
