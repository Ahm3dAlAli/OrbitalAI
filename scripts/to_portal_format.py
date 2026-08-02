#!/usr/bin/env python3
"""Convert internal per-window predictions to the submission-portal format.

Internal files  : <seq>_bb_windows_40ms.txt  (tab-separated, header:
    window_start_timestamp_us  window_end_timestamp_us  center_x  center_y
    width  height  confidence)

Portal files    : <seq>.txt  — ONE DETECTION PER ROW with the fields the portal
    collects (header included):
        sequence_id, window_start_timestamp_us, window_end_timestamp_us,
        x_centre, y_centre, w, h, class_id, confidence

Mapping: sequence_id = sequence name; the two window timestamps carried through;
(x_centre,y_centre) = box centre; (w,h) = box size; class_id = RSO class (default
0, single class); confidence carried through. The internal files and
Evaluation_Metrics.xlsx are left in place — this only ADDS the <seq>.txt portal
files to the same folder.

    python3 scripts/to_portal_format.py --pred-dir /work/OrbitalAI/DDMMYYYY \
        --class-id 0 --delimiter ,
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

INT_SUFFIX = "_bb_windows_40ms.txt"
# Portal contract (TII OrbitSight, 2026): one detection/row, exactly these 9 fields.
FIELDS = ["sequence_id", "window_start_timestamp_us", "window_end_timestamp_us",
          "x_centre", "y_centre", "w", "h", "class_id", "confidence"]


def convert_file(path, out_dir, class_id, delim):
    seq = os.path.basename(path)[: -len(INT_SUFFIX)]
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append([
                seq,
                int(r["window_start_timestamp_us"]),
                int(r["window_end_timestamp_us"]),
                int(float(r["center_x"])), int(float(r["center_y"])),
                int(float(r["width"])),    int(float(r["height"])),
                class_id,
                float(r.get("confidence", 1.0)),
            ])
    out_path = os.path.join(out_dir, seq + ".txt")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow(FIELDS)
        w.writerows(rows)
    return seq, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="folder with <seq>_bb_windows_40ms.txt (also the output folder)")
    ap.add_argument("--out-dir", default=None, help="defaults to --pred-dir")
    ap.add_argument("--class-id", type=int,
                    default=int(os.environ.get("ORBITSIGHT_CLASS_ID", "0")),
                    help="RSO class id written to every row (single class; default 0)")
    ap.add_argument("--delimiter", default=os.environ.get("ORBITSIGHT_PORTAL_DELIM", ","),
                    help="output field delimiter (default ',' — CSV)")
    args = ap.parse_args()
    out_dir = args.out_dir or args.pred_dir
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*" + INT_SUFFIX)))
    if not files:
        print(f"[portal] no internal prediction files (*{INT_SUFFIX}) in {args.pred_dir}")
        return
    total = 0
    for p in files:
        seq, n = convert_file(p, out_dir, args.class_id, args.delimiter)
        total += n
        print(f"[portal] {seq}.txt  ({n} detections)")
    print(f"[portal] wrote {len(files)} portal files ({total} detections) "
          f"-> {out_dir}  [class_id={args.class_id}, delim='{args.delimiter}']")


if __name__ == "__main__":
    main()
