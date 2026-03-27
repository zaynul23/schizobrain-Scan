#!/bin/bash
# =============================================================================
# preprocess_from_csv.sh
#
# Full structural MRI preprocessing pipeline for schizophrenia classification.
# Reads a CSV of (filename, filepath) pairs and produces per-subject output:
#   - <subject>_preprocessed.nii.gz  : skull-stripped, bias-corrected,
#                                       nonlinearly registered to MNI152,
#                                       z-score intensity normalised
#   - <subject>_gm.nii.gz            : grey matter PVE map in MNI space
#   - <subject>_qc.txt               : per-subject FSL tool output log
#
# Pipeline order:
#   1. fslreorient2std
#   2. rough BET -f 0.2              → initial brain for FAST
#   3. fast -B on rough brain        → bias-corrected restore
#   4. clean BET -R on bias brain    → final skull-stripped brain
#   5. fast 3-class on clean brain   → GM PVE _pve_1  (native space)
#   6. flirt 12-DOF affine           → MNI152
#   7. fnirt nonlinear (affine init) → MNI152  (falls back gracefully)
#   8. applywarp / flirt -applyxfm   → GM PVE into MNI space
#   9. z-score normalisation         → zero-mean unit-variance brain
#  10. QC: dims, GM fraction, NaN, brain volume
#
# Usage:
#   bash preprocess_from_csv.sh <path_to_csv> [max_parallel_jobs] [options]
#
# CSV format (header row required):
#   filename,filepath
#   sub001.nii.gz,C:\data\sub001\NIFTI\sub001.nii.gz
#
# Options:
#   --no-fnirt      Skip nonlinear registration (faster, less accurate)
#   --mni-res       MNI resolution: 1mm or 2mm (default: 2mm)
#   --bet-f         BET clean threshold (default: 0.3)
#   --log-dir       Directory for aggregate logs (default: next to CSV)
#   --dry-run       Parse CSV and print what would run, don't execute
#   --force         Re-run even if outputs already exist
#
# Requirements: FSL (fslreorient2std, bet, fast, flirt, fnirt, applywarp,
#               fslstats, fslmaths, fslval)
#
# =============================================================================

set -uo pipefail
# Note: -e is intentionally omitted at the top level so the parallel job
# runner can capture individual exit codes. Each background subshell sets
# its own strict mode.

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_JOBS=4
USE_FNIRT=true
MNI_RES="2mm"
BET_F="0.4"       # Raised from 0.3 — runs showed GM frac ~0.70 (under-stripping)
                  # If GM still > 0.65 after this, raise to 0.45
LOG_DIR=""
DRY_RUN=false
FORCE=false
CSV_FILE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
print_usage() {
    sed -n '/^# Usage:/,/^# Requirements:/p' "$0" | head -n -2 | sed 's/^# \?//'
    exit 1
}

if [ $# -eq 0 ]; then print_usage; fi

CSV_FILE="$1"; shift

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_JOBS="$1"; shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-fnirt) USE_FNIRT=false ;;
        --mni-res)  MNI_RES="$2";  shift ;;
        --bet-f)    BET_F="$2";    shift ;;
        --log-dir)  LOG_DIR="$2";  shift ;;
        --dry-run)  DRY_RUN=true ;;
        --force)    FORCE=true ;;
        -h|--help)  print_usage ;;
        *) echo "[ERROR] Unknown option: $1"; print_usage ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ -z "$CSV_FILE" ]; then
    echo "[ERROR] No CSV file specified."; print_usage
fi
if [ ! -f "$CSV_FILE" ]; then
    echo "[ERROR] CSV not found: $CSV_FILE"; exit 1
fi
if [ -z "${FSLDIR:-}" ]; then
    echo "[ERROR] FSLDIR is not set. Source your FSL setup first."
    echo "        e.g.: source \$FSLDIR/etc/fslconf/fsl.sh"; exit 1
fi

# MNI reference and expected output dimensions
if [ "$MNI_RES" = "2mm" ]; then
    MNI_REF="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz"
    EXPECTED_DIM="91x109x91"
else
    MNI_REF="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
    EXPECTED_DIM="182x218x182"   # FIX: 1mm MNI is 182x218x182, not 91x109x91
fi
if [ ! -f "$MNI_REF" ]; then
    echo "[ERROR] MNI reference not found: $MNI_REF"
    echo "        Try --mni-res 2mm or check your FSL data directory."; exit 1
fi

# FIX: FNIRT config — search multiple FSL install layouts instead of one hardcoded path
FNIRT_CFG=""
if $USE_FNIRT; then
    for candidate in \
        "$FSLDIR/etc/flirtsch/T1_2_MNI152_2mm.cnf" \
        "$FSLDIR/share/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf" \
        "/usr/share/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf" \
        "/etc/fsl/flirtsch/T1_2_MNI152_2mm.cnf"
    do
        if [ -f "$candidate" ]; then FNIRT_CFG="$candidate"; break; fi
    done
    if [ -z "$FNIRT_CFG" ]; then
        echo "[WARNING] FNIRT config not found in any standard location."
        echo "          Falling back to affine-only registration."
        echo "          Use --no-fnirt to suppress this warning."
        USE_FNIRT=false
    else
        echo "[INFO] FNIRT config: $FNIRT_CFG"
    fi
fi

# Log directory
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="$(dirname "$CSV_FILE")/preprocessing_logs"
fi
mkdir -p "$LOG_DIR"

AGGREGATE_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
QC_SUMMARY="$LOG_DIR/qc_summary_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "subject\tstatus\tdim_x\tdim_y\tdim_z\tgm_fraction\thas_nan\tbrain_volume_mm3\telapsed_s\tnotes" \
    > "$QC_SUMMARY"

# ---------------------------------------------------------------------------
# Helper: Windows path → WSL path
# ---------------------------------------------------------------------------
win_to_wsl() {
    local raw="$1"
    local p
    p=$(printf '%s' "$raw" | sed 's|\\|/|g')
    if [[ "$p" =~ ^([A-Za-z]):(/.*) ]]; then
        local drive="${BASH_REMATCH[1],,}"
        local rest="${BASH_REMATCH[2]}"
        p="/mnt/${drive}${rest}"
    fi
    printf '%s' "$p"
}

# ---------------------------------------------------------------------------
# Helper: timestamped log — FIX: flock for safe parallel writes
# ---------------------------------------------------------------------------
log() {
    local level="$1"; shift
    local msg="$*"
    local line="[$(date '+%H:%M:%S')][$level] $msg"
    echo "$line"
    ( flock -x 9; echo "$line" >> "$AGGREGATE_LOG" ) 9>> "$AGGREGATE_LOG"
}

# FIX: QC summary writes also use flock — parallel jobs write simultaneously
qc_append() {
    ( flock -x 9; echo -e "$1" >> "$QC_SUMMARY" ) 9>> "$QC_SUMMARY"
}

# ---------------------------------------------------------------------------
# Core: process a single scan
# ---------------------------------------------------------------------------
process_scan() {
    # FIX: strict mode set inside subshell — top-level omits -e intentionally
    set -uo pipefail

    local FILENAME="$1"
    local WIN_FILEPATH="$2"

    local INPUT
    INPUT=$(win_to_wsl "$WIN_FILEPATH")
    local BASENAME="${FILENAME%.nii.gz}"

    # Output layout mirrors original: .../scans/NIFTI/sub.nii.gz
    #                              → .../scans/preprocessed/PP_sub/
    local NIFTI_DIR SCANS_DIR OUTPUT_DIR SUBJECT_DIR
    NIFTI_DIR=$(dirname "$INPUT")
    SCANS_DIR=$(dirname "$NIFTI_DIR")
    OUTPUT_DIR="$SCANS_DIR/preprocessed"
    SUBJECT_DIR="$OUTPUT_DIR/PP_${BASENAME}"

    local QC_FILE="$SUBJECT_DIR/${BASENAME}_qc.txt"
    local OUT_BRAIN="$SUBJECT_DIR/${BASENAME}_preprocessed.nii.gz"
    local OUT_GM="$SUBJECT_DIR/${BASENAME}_gm.nii.gz"

    # Skip check
    if [ "$FORCE" = false ] && [ -f "$OUT_BRAIN" ] && [ -f "$OUT_GM" ]; then
        log "SKIP " "$BASENAME — outputs exist (use --force to reprocess)"
        qc_append "${BASENAME}\tSKIP\t-\t-\t-\t-\t-\t-\t-\talready done"
        return 0
    fi

    # Input check
    if [ ! -f "$INPUT" ]; then
        log "ERROR" "$BASENAME — input not found: $INPUT"
        qc_append "${BASENAME}\tFAIL_INPUT\t-\t-\t-\t-\t-\t-\t-\tinput not found"
        return 1
    fi

    mkdir -p "$SUBJECT_DIR"
    : > "$QC_FILE"

    # FIX: TEMP_DIR defined then trap set immediately in same scope
    # Previous version defined them separately → TEMP_DIR unbound on EXIT
    local TEMP_DIR
    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TEMP_DIR"' EXIT

    local START_TIME=$SECONDS
    log "START" "$BASENAME"

    # Named intermediates
    local STD="$TEMP_DIR/std.nii.gz"
    local ROUGH_BRAIN="$TEMP_DIR/rough_brain.nii.gz"
    local BIAS_BASE="$TEMP_DIR/rough_brain"
    local BIAS_RESTORED="$TEMP_DIR/bias_restored.nii.gz"
    local CLEAN_BRAIN="$TEMP_DIR/clean_brain.nii.gz"
    local FAST_BASE="$TEMP_DIR/seg"
    local GM_NATIVE="$TEMP_DIR/seg_pve_1.nii.gz"
    local AFFINE_BRAIN="$TEMP_DIR/affine_brain.nii.gz"
    local AFFINE_MAT="$TEMP_DIR/affine.mat"
    local MNI_BRAIN="$TEMP_DIR/mni_brain.nii.gz"
    local MNI_GM="$TEMP_DIR/mni_gm.nii.gz"
    local WARP_FIELD="$TEMP_DIR/warp.nii.gz"
    local FINAL="$TEMP_DIR/final.nii.gz"

    # ------------------------------------------------------------------
    # Step 1: Reorient to standard
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 1/9: reorient to standard"
    if ! fslreorient2std "$INPUT" "$STD" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — fslreorient2std failed"
        qc_append "${BASENAME}\tFAIL_REORIENT\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 2: Rough BET -f 0.2 (loose — exposes enough brain for FAST)
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 2/9: rough skull strip (BET -f 0.2)"
    if ! bet "$STD" "${ROUGH_BRAIN%.nii.gz}" -f 0.2 >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — rough BET failed"
        qc_append "${BASENAME}\tFAIL_BET_ROUGH\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi
    if [ ! -f "$ROUGH_BRAIN" ]; then
        log "ERROR" "$BASENAME — rough BET produced no output"
        qc_append "${BASENAME}\tFAIL_BET_ROUGH_OUT\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 3: Bias field correction on brain-only volume (FAST -B)
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 3/9: bias field correction (FAST -B on brain)"
    if ! fast -B -o "$BIAS_BASE" "$ROUGH_BRAIN" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — FAST -B failed"
        qc_append "${BASENAME}\tFAIL_BIAS\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi
    # FIX: FAST -B restore file naming varies across FSL versions — check both
    if [ -f "${BIAS_BASE}_restore.nii.gz" ]; then
        BIAS_RESTORED="${BIAS_BASE}_restore.nii.gz"
    elif [ -f "${ROUGH_BRAIN%.nii.gz}_restore.nii.gz" ]; then
        BIAS_RESTORED="${ROUGH_BRAIN%.nii.gz}_restore.nii.gz"
    else
        log "ERROR" "$BASENAME — FAST -B restore file not found (checked both naming conventions)"
        echo "Temp dir contents:" >> "$QC_FILE"
        ls "$TEMP_DIR/" >> "$QC_FILE" 2>&1
        qc_append "${BASENAME}\tFAIL_BIAS_OUT\t-\t-\t-\t-\t-\t-\t-\trestore missing"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 4: Clean BET -R on bias-corrected brain
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 4/9: clean skull strip (BET -R -f ${BET_F})"
    if ! bet "$BIAS_RESTORED" "${CLEAN_BRAIN%.nii.gz}" -R -g -0.15 -f "$BET_F" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — clean BET failed"
        qc_append "${BASENAME}\tFAIL_BET_CLEAN\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi
    if [ ! -f "$CLEAN_BRAIN" ]; then
        log "ERROR" "$BASENAME — clean BET output missing"
        qc_append "${BASENAME}\tFAIL_BET_OUT\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi

    local BRAIN_VOL
    BRAIN_VOL=$(fslstats "$CLEAN_BRAIN" -V | awk '{printf "%.0f", $2}')

    # ------------------------------------------------------------------
    # Step 5: Tissue segmentation in native space
    #         CSF=pve_0  GM=pve_1  WM=pve_2
    #         Must run in native space — FAST needs undistorted intensities
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 5/9: tissue segmentation (FAST 3-class)"
    if ! fast -n 3 -t 1 -o "$FAST_BASE" "$CLEAN_BRAIN" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — FAST segmentation failed"
        qc_append "${BASENAME}\tFAIL_SEG\t-\t-\t-\t-\t-\t-\t-\t-"
        return 1
    fi
    if [ ! -f "$GM_NATIVE" ]; then
        log "ERROR" "$BASENAME — GM PVE (seg_pve_1) not found after FAST"
        qc_append "${BASENAME}\tFAIL_GM_MAP\t-\t-\t-\t-\t-\t-\t-\tseg_pve_1 missing"
        return 1
    fi

    # GM fraction QC (healthy adult ~0.35–0.55; schiz may be lower)
    local GM_FRAC
    # FIX: pipe through awk directly — avoids the intermediate variable
    # that could carry whitespace into later comparisons
    GM_FRAC=$(fslstats "$GM_NATIVE" -M | awk '{printf "%.4f", $1}')
    if awk "BEGIN{exit !($GM_FRAC < 0.05 || $GM_FRAC > 0.70)}"; then
        log "WARN " "$BASENAME — unusual GM fraction: $GM_FRAC (check BET quality)"
    fi

    # ------------------------------------------------------------------
    # Step 6: Affine registration to MNI152 (FLIRT 12-DOF)
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 6/9: affine registration to MNI (FLIRT 12-DOF)"
    if ! flirt \
            -in       "$CLEAN_BRAIN" \
            -ref      "$MNI_REF" \
            -out      "$AFFINE_BRAIN" \
            -omat     "$AFFINE_MAT" \
            -dof      12 \
            -cost     corratio \
            -searchrx -90 90 \
            -searchry -90 90 \
            -searchrz -90 90 >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — FLIRT failed"
        qc_append "${BASENAME}\tFAIL_FLIRT\t-\t-\t-\t${GM_FRAC}\t-\t-\t-\t-"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 7: Nonlinear registration (FNIRT) — graceful fallback to affine
    # ------------------------------------------------------------------
    local FNIRT_OK=false
    if $USE_FNIRT; then
        log "INFO " "$BASENAME — step 7/9: nonlinear registration (FNIRT)"
        local FNIRT_LOG="$TEMP_DIR/fnirt.log"
        # FIX: Do NOT use the T1_2_MNI152_2mm.cnf config for 1mm registration.
        # That config hardcodes 2mm-space refmask paths and 2mm warp parameters.
        # Even --applyrefmask=0 doesn't fix the spatial parameter mismatch.
        # Instead, pass all parameters explicitly tuned for 1mm T1→MNI152 1mm.
        # These match the parameters used by fsl_anat for this registration.
        if fnirt \
                --in="$CLEAN_BRAIN" \
                --ref="$MNI_REF" \
                --aff="$AFFINE_MAT" \
                --iout="$MNI_BRAIN" \
                --fout="$WARP_FIELD" \
                --warpres=10,10,10 \
                --subsamp=4,2,1,1 \
                --infwhm=8,4,2,2 \
                --reffwhm=4,2,0,0 \
                --lambda=300,75,30,30 \
                --applyrefmask=0 \
                --applyinmask=0 \
                --logout="$FNIRT_LOG" >> "$QC_FILE" 2>&1; then
            FNIRT_OK=true
            log "INFO " "$BASENAME — FNIRT succeeded"
        else
            log "WARN " "$BASENAME — FNIRT failed, falling back to affine result"
            echo "=== FNIRT LOG ===" >> "$QC_FILE"
            cat "$FNIRT_LOG" >> "$QC_FILE" 2>/dev/null || true
            cp "$AFFINE_BRAIN" "$MNI_BRAIN"
        fi
    else
        log "INFO " "$BASENAME — step 7/9: nonlinear registration SKIPPED (--no-fnirt)"
        cp "$AFFINE_BRAIN" "$MNI_BRAIN"
    fi

    # ------------------------------------------------------------------
    # Step 8: Warp GM PVE map into MNI space
    #         Use FNIRT warp field if available, else affine matrix
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 8/9: warp GM PVE map to MNI space"
    local GM_WARP_OK=false
    if $FNIRT_OK && [ -f "$WARP_FIELD" ]; then
        if applywarp \
                --in="$GM_NATIVE" \
                --ref="$MNI_REF" \
                --warp="$WARP_FIELD" \
                --out="$MNI_GM" \
                --interp=trilinear >> "$QC_FILE" 2>&1; then
            GM_WARP_OK=true
        else
            log "WARN " "$BASENAME — applywarp failed for GM, using FLIRT fallback"
        fi
    fi
    if ! $GM_WARP_OK; then
        if ! flirt \
                -in      "$GM_NATIVE" \
                -ref     "$MNI_REF" \
                -applyxfm -init "$AFFINE_MAT" \
                -out     "$MNI_GM" >> "$QC_FILE" 2>&1; then
            log "ERROR" "$BASENAME — GM warp (affine fallback) failed"
            qc_append "${BASENAME}\tFAIL_GM_WARP\t-\t-\t-\t${GM_FRAC}\t-\t${BRAIN_VOL}\t-\t-"
            return 1
        fi
    fi
    # Clamp PVE to [0,1] — interpolation can produce tiny out-of-range values
    # FIX: avoid in-place overwrite (same input and output) which can be
    # unreliable on some FSL versions. Write to temp then move.
    local MNI_GM_CLAMPED="$TEMP_DIR/mni_gm_clamped.nii.gz"
    fslmaths "$MNI_GM" -thr 0 -uthr 1 "$MNI_GM_CLAMPED" >> "$QC_FILE" 2>&1
    mv "$MNI_GM_CLAMPED" "$MNI_GM"

    # ------------------------------------------------------------------
    # Step 9: Brain masking in MNI space
    #
    # FIX (NaN): After affine/FNIRT registration, interpolation leaves tiny
    # non-zero values throughout the background. fslstats -M then includes
    # those in the mean, biasing it low (observed: post-norm mean = -1.81).
    # Masking first ensures the background is exactly 0 so z-score stats
    # are computed over brain voxels only.
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 9/10: brain masking in MNI space"
    local MNI_BRAIN_MASK="$TEMP_DIR/mni_brain_mask.nii.gz"
    local MNI_BRAIN_MASKED="$TEMP_DIR/mni_brain_masked.nii.gz"

    # FIX: Use -abs before thresholding. After affine registration, background
    # voxels can be slightly negative (interpolation artefacts). -thr 0 alone
    # zeroes negatives correctly, but -abs -thr 0.001 gives a tighter mask
    # that excludes near-zero boundary voxels from the z-score computation.
    if ! fslmaths "$MNI_BRAIN" -abs -thr 0.001 -bin "$MNI_BRAIN_MASK" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — brain mask generation failed"
        qc_append "${BASENAME}\tFAIL_MASK\t-\t-\t-\t${GM_FRAC}\t-\t${BRAIN_VOL}\t-\t-"
        return 1
    fi

    if ! fslmaths "$MNI_BRAIN" -mas "$MNI_BRAIN_MASK" "$MNI_BRAIN_MASKED" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — brain masking failed"
        qc_append "${BASENAME}\tFAIL_MASK_APPLY\t-\t-\t-\t${GM_FRAC}\t-\t${BRAIN_VOL}\t-\t-"
        return 1
    fi

    # ------------------------------------------------------------------
    # Step 10: Z-score intensity normalisation (NaN-safe)
    #
    # Background is now exactly 0, so -M and -S correctly compute stats
    # over brain voxels only. Final -mas pins background back to 0 after
    # the shift: (0 - MEAN)/STD would otherwise be non-zero.
    # ------------------------------------------------------------------
    log "INFO " "$BASENAME — step 10/10: z-score intensity normalisation"
    local MEAN STD
    MEAN=$(fslstats "$MNI_BRAIN_MASKED" -M | tr -d '[:space:]')
    STD=$(fslstats  "$MNI_BRAIN_MASKED" -S | tr -d '[:space:]')
    echo "pre-norm: mean=$MEAN  std=$STD" >> "$QC_FILE"

    if awk "BEGIN{exit !(${STD} < 0.001)}"; then
        log "ERROR" "$BASENAME — std dev near zero ($STD) — registration likely failed"
        qc_append "${BASENAME}\tFAIL_NORM\t-\t-\t-\t${GM_FRAC}\t-\t${BRAIN_VOL}\t-\tstd=${STD}"
        return 1
    fi

    # z-score then re-mask: stops background 0s becoming -MEAN/STD
    if ! fslmaths "$MNI_BRAIN_MASKED" \
            -sub "$MEAN" -div "$STD" \
            -mas "$MNI_BRAIN_MASK" \
            "$FINAL" >> "$QC_FILE" 2>&1; then
        log "ERROR" "$BASENAME — fslmaths z-score failed"
        qc_append "${BASENAME}\tFAIL_NORM_MATH\t-\t-\t-\t${GM_FRAC}\t-\t${BRAIN_VOL}\t-\t-"
        return 1
    fi

    # ------------------------------------------------------------------
    # QC checks: dimensions, NaN/Inf, post-norm mean
    # FIX: fslval also returns trailing whitespace — tr -d applied here too
    # ------------------------------------------------------------------
    local DIM_X DIM_Y DIM_Z DIM_STR
    DIM_X=$(fslval "$FINAL" dim1 | tr -d '[:space:]')
    DIM_Y=$(fslval "$FINAL" dim2 | tr -d '[:space:]')
    DIM_Z=$(fslval "$FINAL" dim3 | tr -d '[:space:]')
    DIM_STR="${DIM_X}x${DIM_Y}x${DIM_Z}"

    local QC_STATUS="PASS"
    local QC_NOTES=""

    if [ "$DIM_STR" != "$EXPECTED_DIM" ]; then
        QC_STATUS="WARN_DIM"
        QC_NOTES="expected ${EXPECTED_DIM}, got ${DIM_STR}"
        log "WARN " "$BASENAME — unexpected dimensions: $DIM_STR (expected $EXPECTED_DIM)"
    fi

    local RANGE HAS_NAN="no"
    RANGE=$(fslstats "$FINAL" -R)
    # FIX: previous awk logic was inverted — flagged every valid scan as NaN.
    # awk exits 0 (success) when numbers ARE valid, so "if !" always fired.
    # Replaced with grep -iE which correctly matches the literal strings
    # "nan" and "inf" that FSL writes when volumes contain invalid values.
    if echo "$RANGE" | grep -iqE "nan|inf"; then
        HAS_NAN="yes"
        QC_STATUS="WARN_NAN"
        QC_NOTES="${QC_NOTES} NaN/Inf in final volume"
        log "WARN " "$BASENAME — NaN or Inf values in final brain volume"
    fi

    # Post-norm mean should be ~0; log and warn if it drifts
    local FINAL_MEAN
    FINAL_MEAN=$(fslstats "$FINAL" -M | tr -d '[:space:]')
    echo "post-norm mean: $FINAL_MEAN  (expected ~0)" >> "$QC_FILE"
    if awk "BEGIN{v=$FINAL_MEAN; if(v<0)v=-v; exit !(v>0.1)}"; then
        log "WARN " "$BASENAME — post-norm mean is $FINAL_MEAN (expected ~0, check masking)"
    fi

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    cp "$FINAL"  "$OUT_BRAIN"
    cp "$MNI_GM" "$OUT_GM"

    local ELAPSED=$(( SECONDS - START_TIME ))
    qc_append "${BASENAME}\t${QC_STATUS}\t${DIM_X}\t${DIM_Y}\t${DIM_Z}\t${GM_FRAC}\t${HAS_NAN}\t${BRAIN_VOL}\t${ELAPSED}\t${QC_NOTES}"
    log "DONE " "$BASENAME  [dims=${DIM_STR}  gm=${GM_FRAC}  vol=${BRAIN_VOL}mm3  ${ELAPSED}s  ${QC_STATUS}]"
    return 0
}

export -f process_scan
export -f win_to_wsl
export -f log
export -f qc_append
export FSLDIR MNI_REF EXPECTED_DIM BET_F USE_FNIRT FNIRT_CFG QC_SUMMARY AGGREGATE_LOG FORCE

# ---------------------------------------------------------------------------
# Parse CSV
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Structural MRI preprocessing pipeline"
echo "============================================================"
echo " CSV:           $CSV_FILE"
echo " Max jobs:      $MAX_JOBS"
echo " FNIRT:         $USE_FNIRT"
echo " MNI ref:       $MNI_REF"
echo " BET threshold: $BET_F"
echo " Log dir:       $LOG_DIR"
echo " Force rerun:   $FORCE"
echo " Dry run:       $DRY_RUN"
echo "============================================================"
echo ""

TMPLIST=$(mktemp)
tail -n +2 "$CSV_FILE" | awk -F',' '
    NF >= 2 {
        gsub(/^[ \t"]+|[ \t"]+$/, "", $1)
        gsub(/^[ \t"]+|[ \t"]+$/, "", $2)
        if ($1 != "" && $2 != "")
            print $1 "|" $2
    }
' > "$TMPLIST"

TOTAL=$(wc -l < "$TMPLIST" | tr -d '[:space:]')
echo " Subjects found in CSV: $TOTAL"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "[ERROR] No valid rows found in CSV. Check format: filename,filepath"
    rm -f "$TMPLIST"; exit 1
fi

if $DRY_RUN; then
    echo "[DRY RUN] Would process:"
    while IFS= read -r LINE; do
        FNAME="${LINE%%|*}"
        FPATH="${LINE#*|}"
        echo "  $FNAME  ->  $(win_to_wsl "$FPATH")"
    done < "$TMPLIST"
    rm -f "$TMPLIST"
    echo ""
    echo "[DRY RUN] Done. No files were processed."
    exit 0
fi

# ---------------------------------------------------------------------------
# Parallel execution — poll-based reaper, no wait -n dependency
# ---------------------------------------------------------------------------
SUCCEEDED=0
FAILED=0
PIDS=()
NAMES=()

track_job() {
    local pid="$1" name="$2"
    PIDS+=("$pid")
    NAMES+=("$name")
}

reap_finished() {
    local NEW_PIDS=() NEW_NAMES=()
    local i
    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local name="${NAMES[$i]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                SUCCEEDED=$((SUCCEEDED + 1))
            else
                FAILED=$((FAILED + 1))
                log "FAIL " "$name exited with code $EXIT_CODE"
            fi
        else
            NEW_PIDS+=("$pid")
            NEW_NAMES+=("$name")
        fi
    done
    PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
    NAMES=("${NEW_NAMES[@]+"${NEW_NAMES[@]}"}")
}

wait_for_slot() {
    while [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; do
        reap_finished
        if [ "${#PIDS[@]}" -ge "$MAX_JOBS" ]; then sleep 1; fi
    done
}

log "INFO " "Starting preprocessing of $TOTAL subjects (max $MAX_JOBS parallel)"

while IFS= read -r LINE; do
    FILENAME="${LINE%%|*}"
    FILEPATH="${LINE#*|}"
    wait_for_slot
    process_scan "$FILENAME" "$FILEPATH" &
    PID=$!
    track_job "$PID" "$FILENAME"
done < "$TMPLIST"

log "INFO " "All jobs submitted. Waiting for remaining jobs to finish..."
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    wait "$pid" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        FAILED=$((FAILED + 1))
        log "FAIL " "$name exited with code $EXIT_CODE"
    fi
done

rm -f "$TMPLIST"

echo ""
echo "============================================================"
echo " Preprocessing complete"
echo "============================================================"
echo " Total subjects:   $TOTAL"
echo " Succeeded:        $SUCCEEDED"
echo " Failed:           $FAILED"
echo " QC summary:       $QC_SUMMARY"
echo " Full log:         $AGGREGATE_LOG"
echo "============================================================"

[ "$FAILED" -eq 0 ] && exit 0 || exit 1