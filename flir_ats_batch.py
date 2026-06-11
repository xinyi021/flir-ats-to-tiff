r"""Batch lossless conversion of FLIR ATS-US recordings to 16-bit BigTIFF.

Each input ``Rec-NNNNNN.ats`` produces three files in the output folder:

    Rec-NNNNNN_raw.tif       multi-page BigTIFF, uint16, zlib level 5
    Rec-NNNNNN_meta.json     camera info + Planck-relevant calibration data
    Rec-NNNNNN_preview.png   single 8-bit grayscale preview frame

The TIFF pixels are the camera's raw 14-bit ADC counts stored in a uint16
container.  This is the same numeric data the FLIR SDK reads back when
opening the original .ats; no rescaling, gain, or temperature conversion
has been applied.  Downstream temperature reconstruction is done by
reopening the original .ats through the SDK (or by feeding the raw
counts + camera calibration into the Planck inversion described in the
README).

Preview frame selection:
    During the conversion pass each frame's mean count is computed
    (negligible overhead, ~3 percent vs the disk write).  The frame with
    the highest mean is kept as ``..._preview.png`` so the preview shows
    the hottest moment of the recording.  Falls back to a fixed frame
    index 700 if max-mean tracking is somehow disabled.

Usage:
    python flir_ats_batch.py
        (then enter input and output folders at the prompts)

    python flir_ats_batch.py --input D:\path\to\ATS --output E:\path\to\TIFF
        (non-interactive)

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

# Standard scientific stack
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
PREVIEW_PCT_LO    = 1       # percentile for stretching the 16-bit preview
PREVIEW_PCT_HI    = 99      # to 8-bit
FALLBACK_PREVIEW_FRAME = 700
ZLIB_LEVEL        = 5       # tiff compression level (1=fast, 9=best)


# ---------- JSON helpers ---------------------------------------------------
def _jsonable(x):
    """Best-effort: convert arbitrary SDK objects (enums, FLIR wrappers,
    datetimes, bytes) into something json.dump can serialise."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if hasattr(x, "name") and hasattr(x, "value"):          # enum.Enum
        return f"{x.name}({x.value})"
    if isinstance(x, bytes):
        return x.hex()
    # generic SDK object: dump its non-private, non-callable attrs
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


def collect_metadata(f, *, n_frames, preview_idx, preview_mean):
    return {
        "n_frames": int(n_frames),
        "width": int(f.width),
        "height": int(f.height),
        "data_type": _jsonable(f.data_type),
        "pixel_unit_written": "COUNTS (raw 14-bit ADC in uint16)",
        "preview": {
            "frame_index": int(preview_idx),
            "frame_mean_count": float(preview_mean),
            "selection_rule": (
                "frame with the highest mean ADC count across the whole "
                "recording (= overall hottest frame)"
            ),
        },
        "temperature_recovery": (
            "Reopen this .ats through the FLIR Science File SDK and iterate "
            "frames in Unit.TEMPERATURE_FACTORY (or TEMPERATURE_USER with "
            "custom object_parameters) to get calibrated Kelvin / Celsius. "
            "The SDK applies the camera-specific Planck constants internally."
        ),
        "source_info": _jsonable(f.source_info),
        "object_parameters": _jsonable(f.object_parameters),
        "current_preset_index": int(f.preset),
        "preset_info": _jsonable(list(f.source_info.preset_info)),
    }


# ---------- Per-file conversion -------------------------------------------
def convert_one(ats_path: Path, out_dir: Path,
                overwrite: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem      = ats_path.stem
    out_tif   = out_dir / f"{stem}_raw.tif"
    out_json  = out_dir / f"{stem}_meta.json"
    out_png   = out_dir / f"{stem}_preview.png"

    if not overwrite and out_tif.exists() and out_json.exists() and out_png.exists():
        return {"status": "skipped",
                "tif_gb": out_tif.stat().st_size / 1e9}

    t0 = time.time()
    f = fnv.file.ImagerFile(str(ats_path))
    f.unit = fnv.Unit.COUNTS                  # raw 14-bit ADC, uint16 container

    n = int(f.num_frames)
    H, W = int(f.height), int(f.width)

    print(f"  camera = {f.source_info.camera}  "
          f"serial = {f.source_info.camera_serial}", flush=True)
    print(f"  {n} frames @ {f.source_info.preset_info[0].frame_rate:.1f} fps  "
          f"{H}x{W}  raw payload ~ {n*H*W*2/1e9:.2f} GB", flush=True)

    # ---- Track frame with highest mean count -----------------------------
    best_mean: float = -1.0
    best_idx:  int   = -1
    best_frame: np.ndarray | None = None

    report_step = max(1, n // 10)
    with tifffile.TiffWriter(str(out_tif), bigtiff=True) as tw:
        for i in range(n):
            f.get_frame(i)
            page = np.asarray(f.final, dtype=np.uint16).reshape((H, W))
            tw.write(
                page, photometric="minisblack",
                compression="zlib", compressionargs={"level": ZLIB_LEVEL},
                contiguous=False,
            )
            m = float(page.mean())            # ~0.5 ms per frame on 512x640
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

    # Fall back to a fixed frame if tracking somehow yielded nothing
    if best_frame is None:
        idx = FALLBACK_PREVIEW_FRAME if n > FALLBACK_PREVIEW_FRAME else n // 2
        f.get_frame(idx)
        best_frame = np.asarray(f.final, dtype=np.uint16).reshape((H, W))
        best_idx, best_mean = idx, float(best_frame.mean())

    # ---- Save preview PNG (1-99 % percentile stretch to 8-bit) ----------
    lo, hi = np.percentile(best_frame, [PREVIEW_PCT_LO, PREVIEW_PCT_HI])
    u8 = np.clip((best_frame.astype(np.float32) - lo) /
                 max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_png)

    # ---- Save metadata JSON ---------------------------------------------
    meta = collect_metadata(f, n_frames=n,
                            preview_idx=best_idx,
                            preview_mean=best_mean)
    meta["source_file"] = str(ats_path)
    out_json.write_text(json.dumps(meta, indent=2, default=str))

    return {
        "status": "ok",
        "n_frames": n,
        "preview_frame": best_idx,
        "preview_mean": best_mean,
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
    print(f"[setup] {len(ats_files)} .ats files to convert "
          f"({'overwrite' if args.overwrite else 'skip existing'})")
    print()

    results = []
    batch_t0 = time.time()
    for i, ats in enumerate(ats_files, 1):
        rel = ats.relative_to(args.input) if args.input in ats.parents else ats.name
        print(f"=== [{i}/{len(ats_files)}] {rel} ===", flush=True)
        try:
            r = convert_one(ats, args.output, overwrite=args.overwrite)
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
