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







#!/bin/bash
python preprocess.py