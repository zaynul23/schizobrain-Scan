#!/bin/bash
# =============================================================================
# apply_pclamp.sh
#
# Applies percentile clamping (site harmonisation) to existing preprocessed
# scans and masks GM maps. Reads master CSV, finds PP dirs, writes to
# preprocessed2/ alongside the original preprocessed/ folder.
#
# Usage: bash apply_pclamp.sh <master_scan_list.csv> [max_jobs]
# =============================================================================

set -uo pipefail

CSV_FILE="${1:?Usage: bash apply_pclamp.sh <csv> [max_jobs]}"
MAX_JOBS="${2:-6}"

if [ -z "${FSLDIR:-}" ]; then
    echo "[ERROR] FSLDIR not set. Source FSL first."; exit 1
fi

# Windows path → WSL path
win_to_wsl() {
    local p
    p=$(printf '%s' "$1" | tr -d '\r' | sed 's|\\|/|g')
    if [[ "$p" =~ ^([A-Za-z]):(/.*) ]]; then
        p="/mnt/${BASH_REMATCH[1],,}${BASH_REMATCH[2]}"
    fi
    printf '%s' "$p"
}

process_one() {
    local FILENAME="$1"
    local WIN_FILEPATH="$2"

    local BASENAME="${FILENAME%.nii.gz}"
    local INPUT
    INPUT=$(win_to_wsl "$WIN_FILEPATH")

    # Derive paths: .../scans/NIFTI/file.nii.gz → go up to dataset root
    local NIFTI_DIR SCANS_DIR DATASET_DIR PP_DIR PP_SUBDIR OUT_DIR OUT_SUBDIR
    NIFTI_DIR=$(dirname "$INPUT")          # .../scans/NIFTI
    SCANS_DIR=$(dirname "$NIFTI_DIR")      # .../scans
    DATASET_DIR=$(dirname "$SCANS_DIR")    # .../open_neuro_0_71
    PP_DIR="$DATASET_DIR/preprocessed"
    PP_SUBDIR="$PP_DIR/PP_${BASENAME}"
    OUT_DIR="$DATASET_DIR/preprocessed2"
    OUT_SUBDIR="$OUT_DIR/PP_${BASENAME}"

    local SRC_BRAIN="$PP_SUBDIR/${BASENAME}_preprocessed.nii.gz"
    local SRC_GM="$PP_SUBDIR/${BASENAME}_gm.nii.gz"

    if [ ! -f "$SRC_BRAIN" ] || [ ! -f "$SRC_GM" ]; then
        echo "[SKIP] $BASENAME — PP files not found"
        return 0
    fi

    # Skip if already done
    if [ -f "$OUT_SUBDIR/${BASENAME}_preprocessed.nii.gz" ] && \
       [ -f "$OUT_SUBDIR/${BASENAME}_gm.nii.gz" ]; then
        echo "[SKIP] $BASENAME — already in preprocessed2"
        return 0
    fi

    mkdir -p "$OUT_SUBDIR"

    # 1. Create brain mask from existing z-scored volume
    local MASK="$OUT_SUBDIR/mask_tmp.nii.gz"
    fslmaths "$SRC_BRAIN" -abs -thr 0.001 -bin "$MASK" 2>/dev/null

    # 2. Mask GM map
    fslmaths "$SRC_GM" -mas "$MASK" "$OUT_SUBDIR/${BASENAME}_gm.nii.gz" 2>/dev/null

    # 3. Percentile clamp the brain volume: [P1,P99] → [0,1]
    local P1 P99 PRANGE
    P1=$(fslstats "$SRC_BRAIN" -P 1 | tr -d '[:space:]')
    P99=$(fslstats "$SRC_BRAIN" -P 99 | tr -d '[:space:]')
    PRANGE=$(awk "BEGIN{printf \"%.6f\", $P99 - $P1}")

    if awk "BEGIN{exit !($PRANGE < 0.001)}"; then
        echo "[WARN] $BASENAME — degenerate range P1=$P1 P99=$P99, copying as-is"
        cp "$SRC_BRAIN" "$OUT_SUBDIR/${BASENAME}_preprocessed.nii.gz"
    else
        fslmaths "$SRC_BRAIN" \
            -thr "$P1" -uthr "$P99" \
            -sub "$P1" -div "$PRANGE" \
            -mas "$MASK" \
            "$OUT_SUBDIR/${BASENAME}_preprocessed.nii.gz" 2>/dev/null
    fi

    # Cleanup
    rm -f "$MASK"

    echo "[DONE] $BASENAME  (P1=$P1 P99=$P99)"
}

export -f process_one win_to_wsl
export FSLDIR

# Parse CSV — skip header, extract filename and filepath
echo "========================================"
echo " Percentile clamping post-processor"
echo " CSV: $CSV_FILE"
echo " Max jobs: $MAX_JOBS"
echo "========================================"

PIDS=()
DONE=0
TOTAL=0

tail -n +2 "$CSV_FILE" | tr -d '\r' | while IFS=',' read -r FNAME FPATH REST; do
    FNAME=$(echo "$FNAME" | sed 's/^[ "]*//;s/[ "]*$//')
    FPATH=$(echo "$FPATH" | sed 's/^[ "]*//;s/[ "]*$//')
    [ -z "$FNAME" ] && continue
    TOTAL=$((TOTAL + 1))

    # Throttle
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done

    process_one "$FNAME" "$FPATH" &
done

wait
echo ""
echo "[FINISHED] All subjects processed."