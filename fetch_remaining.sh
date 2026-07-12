#!/bin/bash
# Fetch the training files still missing after Google Drive throttled us.
# Retries with backoff until all present or attempts exhausted.
DIR="/Users/ahmeda./Desktop/OrbitalAI/OrbitSight_Dataset/Training_sets"
cd "$DIR" || exit 1

# "fileid|output_name"  (NOAA6_19-06-31 npy is NOT present in the source Drive folder)
items=(
  "111dzdo-qc9blPNqTH6VAex7eQjF5E69w|DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50_labeled_events.npy"
  "1kKBQqhNfX6MKI6NdFSf8i80wajr4aMi2|DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50_bb_windows_40ms.txt"
  "1yZ_SwRGmGtgI6R3dav8tg-FwDGmA5eIW|DVX_Filtered_Stars_2025-01-20-19-15-10_labeled_events.npy"
  "1Z1R_DcAJvuIrqLyJPQMOHjnMKVu5uJgB|DVX_Filtered_Stars_2025-01-20-19-15-10_bb_windows_40ms.txt"
  "1QMM32YAbRh1iYka0Mn9ZCX2jbTSE3cIs|DVX_Filtered_Stars2_2025-01-20-19-57-17_labeled_events.npy"
  "1N72O6e5TrEvnlD13biRkFhWOJcEvR_Cf|DVX_Filtered_Stars2_2025-01-20-19-57-17_bb_windows_40ms.txt"
  "1iOH9WSmuTJuJNJ-qiXoENnVrOPGItO0B|DVX_NOAA6_11416_2025-01-20-19-06-31_bb_windows_40ms.txt"
)

for attempt in 1 2 3 4 5 6 7 8; do
  missing=0
  for it in "${items[@]}"; do
    id="${it%%|*}"; out="${it##*|}"
    # consider an npy valid only if >1MB; txt only if non-empty
    if [ -f "$out" ]; then
      case "$out" in
        *.npy) [ "$(stat -f%z "$out")" -gt 1000000 ] && continue ;;
        *)     [ -s "$out" ] && continue ;;
      esac
    fi
    echo "[attempt $attempt] downloading $out"
    gdown "$id" -O "$out" 2>&1 | tail -1
    # re-check
    if [ -f "$out" ]; then
      case "$out" in
        *.npy) [ "$(stat -f%z "$out")" -gt 1000000 ] || { rm -f "$out"; missing=1; } ;;
        *)     [ -s "$out" ] || { rm -f "$out"; missing=1; } ;;
      esac
    else
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    echo "ALL REMAINING FILES DOWNLOADED on attempt $attempt"
    exit 0
  fi
  echo "[attempt $attempt] still throttled/incomplete; sleeping 600s before retry"
  sleep 600
done
echo "EXHAUSTED ATTEMPTS — some files still missing"
exit 1
