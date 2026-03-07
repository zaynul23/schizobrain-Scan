# #!/bin/bash

# # Step 1: Reorientation
# fslreorient2std \
# /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/sub-01_T1w.nii \
# /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/sub-01_T1w_reoriented.nii

# For all the datapoints
# #!/bin/bash

# # dataset folder
# DATA_DIR="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro"

# # loop through all .nii files
# find $DATA_DIR -name "*.nii" | while read file
# do
#     echo "Processing: $file"

#     # create output filename
#     output="${file%.nii}_reoriented.nii"

#     # run FSL reorientation
#     fslreorient2std "$file" "$output"

# done

# fast -B /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/sub-01_T1w_reoriented.nii

# bet /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/sub-01_T1w_reoriented_restore.nii.gz /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain.nii.gz

# flirt \
# -in brain.nii.gz \
# -ref /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain.nii.gz \
# -out /mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain_mni.nii.gz \
# -omat brain_to_mni.mat

# fsleyes $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz brain_mni.nii.gz


#!/bin/bash

# -------------------------------------------------------
# Function: Intensity Normalization
# Tool: fslstats + fslmaths
# Purpose: Normalize MRI voxel intensities so that
# different subjects have comparable intensity ranges.
# -------------------------------------------------------

# # Input brain MRI (after skull stripping)
# INPUT="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain_mni.nii.gz"

# # Output normalized MRI
# OUTPUT="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain_intensity_normalized_mni.nii.gz"


# # -------------------------------------------------------
# # Step 1: Compute mean intensity
# # -------------------------------------------------------

# MEAN=$(fslstats "$INPUT" -M)

# echo "Mean intensity = $MEAN"


# # -------------------------------------------------------
# # Step 2: Divide image by mean intensity
# # -------------------------------------------------------

# fslmaths "$INPUT" -div $MEAN "$OUTPUT"

# echo "Intensity normalization completed."

#!/bin/bash

# -------------------------------------------------------
# Generate Gray Matter Map
# Tool: FAST (FSL)
# -------------------------------------------------------

# # Input preprocessed MRI
# INPUT="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99/sub-01_T1w/brain_intensity_normalized_mni.nii.gz"

# # Run FAST segmentation
# fast "$INPUT"

# echo "Segmentation complete."
# echo "Gray Matter Map:"
# echo "brain_intensity_normalized_mni_pve_1.nii.gz"







# #!/bin/bash

# echo "Starting preprocessing..."

# python3 preprocessing.py

# echo "Done"




# #!/bin/bash
# set -e

# echo "Starting MRI preprocessing..."

# INPUT="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99_2/sub-01_T1w.nii"
# OUTPUT_DIR="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99_2"

# mkdir -p $OUTPUT_DIR

# BRAIN="$OUTPUT_DIR/brain.nii.gz"

# # 1️⃣ Reorientation

# echo "Reorienting image..."
# fslreorient2std $INPUT $BRAIN

# # 2️⃣ Bias Field Correction

# echo "Bias field correction..."
# fast -B $BRAIN
# mv ${OUTPUT_DIR}/brain_restore.nii.gz $BRAIN

# # 3️⃣ Skull Stripping

# echo "Skull stripping..."
# bet $BRAIN $BRAIN

# # 4️⃣ Registration to MNI

# echo "Registering to MNI template..."
# flirt -in $BRAIN 
# -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz 
# -out $BRAIN

# # 5️⃣ Intensity Normalization

# echo "Intensity normalization..."
# MEAN=$(fslstats $BRAIN -M)
# fslmaths $BRAIN -div $MEAN $BRAIN

# # 6️⃣ Gray Matter Map Generation

# echo "Generating gray matter map..."
# fast $BRAIN

# echo "Preprocessing complete."

# echo "Final outputs:"
# echo "$OUTPUT_DIR/brain.nii.gz"
# echo "$OUTPUT_DIR/brain_pve_1.nii.gz"







# #!/bin/bash
# set -e

# echo "Starting MRI preprocessing..."

# INPUT="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99_2/sub-01_T1w.nii"
# OUTPUT_DIR="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99_2"

# mkdir -p $OUTPUT_DIR

# REORIENT="$OUTPUT_DIR/reoriented.nii.gz"
# BIAS="$OUTPUT_DIR/bias_corrected.nii.gz"
# BRAIN="$OUTPUT_DIR/brain.nii.gz"
# MNI="$OUTPUT_DIR/brain_mni.nii.gz"
# NORM="$OUTPUT_DIR/brain_normalized.nii.gz"

# # 1 Reorientation
# echo "Reorienting..."
# fslreorient2std $INPUT $REORIENT

# # 2 Bias field correction
# echo "Bias correction..."
# fast -B $REORIENT
# mv $OUTPUT_DIR/reoriented_restore.nii.gz $BIAS

# # 3 Skull stripping
# echo "Skull stripping..."
# bet $BIAS $BRAIN

# # 4 Registration to MNI
# echo "Registering to MNI..."
# flirt -in $BRAIN \
# -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
# -out $MNI

# # 5 Intensity normalization
# echo "Intensity normalization..."
# MEAN=$(fslstats $MNI -M)
# fslmaths $MNI -div $MEAN $NORM

# # 6 GM segmentation (NOW on the fully preprocessed image)
# echo "Generating GM map..."
# fast $NORM

# echo "Done."

# echo "Final outputs:"
# echo "$NORM"
# echo "$OUTPUT_DIR/brain_normalized_pve_1.nii.gz"







#!/bin/bash
set -e

BASE_DIR="/mnt/c/Users/sucha/OneDrive/Documents/schizobrain-Scan/MRI_data/open_neuro/open_neuro_4_99_2"

echo "Starting MRI preprocessing..."

for INPUT in "$BASE_DIR"/sub-*_T1w.nii; do

echo "Processing: $INPUT"

OUTPUT_DIR="$BASE_DIR"

REORIENT="$OUTPUT_DIR/reoriented.nii.gz"
BIAS="$OUTPUT_DIR/bias_corrected.nii.gz"
BRAIN="$OUTPUT_DIR/brain.nii.gz"
MNI="$OUTPUT_DIR/brain_mni.nii.gz"
NORM="$OUTPUT_DIR/brain_normalized.nii.gz"

# 1 Reorientation

echo "Reorienting..."
fslreorient2std "$INPUT" "$REORIENT"

# 2 Bias field correction

echo "Bias correction..."
fast -B "$REORIENT"
mv "$OUTPUT_DIR/reoriented_restore.nii.gz" "$BIAS"

# 3 Skull stripping

echo "Skull stripping..."
bet "$BIAS" "$BRAIN"

# 4 Registration to MNI

echo "Registering to MNI..."
flirt -in "$BRAIN" 
-ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz 
-out "$MNI"

# 5 Intensity normalization

echo "Intensity normalization..."
MEAN=$(fslstats "$MNI" -M)
fslmaths "$MNI" -div $MEAN "$NORM"

# 6 GM segmentation (on preprocessed image)

echo "Generating GM map..."
fast "$NORM"

# Remove unnecessary FAST outputs

rm -f "$OUTPUT_DIR"/brain_normalized_pve_0.nii.gz
rm -f "$OUTPUT_DIR"/brain_normalized_pve_2.nii.gz
rm -f "$OUTPUT_DIR"/brain_normalized_seg.nii.gz
rm -f "$OUTPUT_DIR"/brain_normalized_mixeltype.nii.gz
rm -f "$OUTPUT_DIR"/brain_normalized_pveseg.nii.gz

# Remove intermediate preprocessing files

rm -f "$REORIENT"
rm -f "$BIAS"
rm -f "$BRAIN"
rm -f "$MNI"

echo "Finished processing: $INPUT"

done

echo "All preprocessing completed."
echo "Final outputs:"
echo "brain_normalized.nii.gz"
echo "brain_normalized_pve_1.nii.gz"