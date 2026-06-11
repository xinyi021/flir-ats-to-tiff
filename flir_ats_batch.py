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


def collect_metadata(f, *, n_frames, preview_idx, preview_mean, unit_label):
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
        "raw_counts_recovery": (
            "Reopen the source .ats through the FLIR Science File SDK and "
            "set f.unit = fnv.Unit.COUNTS to get the unchanged 14-bit ADC "
            "counts back.  This tool does not modify the source .ats files."
        ),
        "source_info": _jsonable(f.source_info),
        "object_parameters": _jsonable(f.object_parameters),
        "current_preset_index": int(f.preset),
        "preset_info": _jsonable(list(f.source_info.preset_info)),
    }


# ---------- Per-file conversion -------------------------------------------
def convert_one(ats_path: Path, out_dir: Path,
                *, unit: str = "celsius",
                overwrite: bool = False) -> dict:
    """Convert one .ats into _temp_{C|K}.tif + _meta.json + _preview.png."""
    if unit not in ("celsius", "kelvin"):
        raise ValueError(f"unit must be 'celsius' or 'kelvin', got {unit!r}")
    unit_suffix = "C" if unit == "celsius" else "K"
    unit_label  = "celsius" if unit == "celsius" else "kelvin"

    out_dir.mkdir(parents=True, exist_ok=True)
    stem      = ats_path.stem
    out_tif   = out_dir / f"{stem}_temp_{unit_suffix}.tif"
    out_json  = out_dir / f"{stem}_meta.json"
    out_png   = out_dir / f"{stem}_preview.png"

    if (not overwrite
        and out_tif.exists() and out_json.exists() and out_png.exists()):
        return {"status": "skipped",
                "tif_gb": out_tif.stat().st_size / 1e9}

    t0 = time.time()
    f = fnv.file.ImagerFile(str(ats_path))
    # SDK gives us float32 Kelvin in TEMPERATURE_FACTORY; we shift to
    # Celsius below if asked.  This is the radiometric path that uses
    # the camera's factory calibration plus the recorded object
    # parameters (emissivity, distance, reflected temp, etc.).
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


# ---------- Batch driver --------------------------------------------------
def _prompt_dir(prompt: str) -> Path | None:
    raw = input(prompt).strip().strip('"').strip("'")
    return Path(raw) if raw else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",  type=Path,
                    help="Folder containing .ats files (prompts if omitted)")
    ap.add_argument("--output", type=Path,
                    help="Folder for .tif/.json/.png outputs (prompts if omitted)")
    ap.add_argument("--unit", choices=("celsius", "kelvin"), default="celsius",
                    help="Temperature unit written into the TIFF (default celsius)")
    ap.add_argument("--no-recurse", action="store_true",
                    help="Only scan the top level of --input; default is recursive")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert files whose outputs already exist")
    args = ap.parse_args()

    if args.input is None:
        args.input = _prompt_dir("Input folder (contains .ats files): ")
    if args.output is None:
        args.output = _prompt_dir("Output folder for .tif + .json + .png: ")

    if args.input is None or not args.input.is_dir():
        print(f"ERROR: input folder not valid: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.output is None:
        print("ERROR: output folder is required", file=sys.stderr)
        sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)

    pattern = args.input.glob if args.no_recurse else args.input.rglob
    ats_files = sorted(pattern("*.ats"))
    if not ats_files:
        print(f"No .ats files found under {args.input}")
        sys.exit(0)

    print()
    print(f"[setup] FLIR SDK loaded: {fnv.__file__}")
    print(f"[setup] input:  {args.input}")
    print(f"[setup] output: {args.output}")
    print(f"[setup] {len(ats_files)} .ats files to convert  "
          f"(unit={args.unit}, "
          f"{'overwrite' if args.overwrite else 'skip existing'})")
    print()

    results = []
    batch_t0 = time.time()
    for i, ats in enumerate(ats_files, 1):
        rel = (ats.relative_to(args.input)
               if args.input in ats.parents else ats.name)
        print(f"=== [{i}/{len(ats_files)}] {rel} ===", flush=True)
        try:
            r = convert_one(ats, args.output, unit=args.unit,
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


if __name__ == "__main__":
    main()
