"""
apply_combat.py

Applies ComBat harmonization to all preprocessed2 brain volumes.
Reads master CSV to find subjects, loads flattened brain voxels,
runs neuroCombat with site as batch and label as biological covariate,
saves harmonized volumes to preprocessed3/.

Usage:
    pip install neuroCombat nibabel pandas --break-system-packages
    python apply_combat.py --csv master_scan_list_v2.csv

Output structure mirrors preprocessed2/:
    .../preprocessed3/PP_<basename>/<basename>_preprocessed.nii.gz
    .../preprocessed3/PP_<basename>/<basename>_gm.nii.gz  (copied, not combatted)
"""

import os
import sys
import argparse
import platform
import numpy as np
import nibabel as nib
import pandas as pd
from neuroCombat import neuroCombat


def normalize_path(path: str) -> str:
    path = path.strip().strip("\r")
    if platform.system() != "Windows":
        path = path.replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--channel", default="brain", choices=["brain", "gm", "both"],
                   help="Which channel(s) to harmonize. Default: brain only. "
                        "GM is already [0,1] PVE and less affected by scanner.")
    args = p.parse_args()

    # --- Parse CSV, collect subject info ---
    df = pd.read_csv(args.csv)
    df = df[df["grp"].isin([1, 3])].copy()
    df["filename"] = df["filename"].str.strip().str.strip("\r")
    df["filepath"] = df["filepath"].str.strip().str.strip("\r")

    subjects = []
    for _, row in df.iterrows():
        basename = row["filename"].replace(".nii.gz", "")
        native_path = normalize_path(row["filepath"])
        nifti_dir = os.path.dirname(native_path)
        scans_dir = os.path.dirname(nifti_dir)
        dataset_dir = os.path.dirname(scans_dir)

        pp2_dir = os.path.join(dataset_dir, "preprocessed2", f"PP_{basename}")
        pp3_dir = os.path.join(dataset_dir, "preprocessed3", f"PP_{basename}")
        brain_path = os.path.join(pp2_dir, f"{basename}_preprocessed.nii.gz")
        gm_path = os.path.join(pp2_dir, f"{basename}_gm.nii.gz")

        if not os.path.exists(brain_path):
            continue

        subjects.append({
            "basename": basename,
            "brain_path": brain_path,
            "gm_path": gm_path,
            "pp3_dir": pp3_dir,
            "site": int(row["dscode"]),
            "label": int(row["label"]),
        })

    print(f"Found {len(subjects)} subjects with preprocessed2 files")
    if len(subjects) == 0:
        print("ERROR: No subjects found. Check paths.")
        sys.exit(1)

    # --- Load reference image for affine/header ---
    ref_img = nib.load(subjects[0]["brain_path"])
    affine = ref_img.affine
    header = ref_img.header
    shape = ref_img.shape
    n_voxels = np.prod(shape)
    print(f"Volume shape: {shape}, voxels per scan: {n_voxels}")

    # --- Load all brain volumes into matrix (n_voxels x n_subjects) ---
    print("Loading brain volumes...")
    brain_matrix = np.zeros((n_voxels, len(subjects)), dtype=np.float32)
    for i, s in enumerate(subjects):
        vol = nib.load(s["brain_path"]).get_fdata(dtype=np.float32).ravel()
        brain_matrix[:, i] = vol
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(subjects)}")
    print(f"  Loaded all {len(subjects)} brains")

    # --- Build covariates DataFrame ---
    covars = pd.DataFrame({
        "site": [s["site"] for s in subjects],
        "label": [s["label"] for s in subjects],
    })
    print(f"Sites: {dict(covars['site'].value_counts().sort_index())}")
    print(f"Labels: {dict(covars['label'].value_counts().sort_index())}")

    # --- Create brain mask (voxels that are non-zero in >50% of subjects) ---
    # ComBat on zero-background voxels adds noise. Mask to brain-only voxels.
    nonzero_frac = (brain_matrix != 0).mean(axis=1)
    brain_mask = nonzero_frac > 0.5
    n_brain_voxels = brain_mask.sum()
    print(f"Brain mask: {n_brain_voxels} voxels ({100*n_brain_voxels/n_voxels:.1f}% of volume)")

    # --- Run ComBat on brain voxels only ---
    print("Running ComBat on brain channel...")
    brain_masked = brain_matrix[brain_mask, :]

    combat_result = neuroCombat(
        dat=brain_masked,
        covars=covars,
        batch_col="site",
        categorical_cols=["label"],
    )
    harmonized_brain = combat_result["data"]
    print("ComBat complete.")

    # --- Reconstruct full volumes and save ---
    print("Saving harmonized volumes to preprocessed3/...")
    for i, s in enumerate(subjects):
        os.makedirs(s["pp3_dir"], exist_ok=True)

        # Reconstruct brain volume
        full_vol = np.zeros(n_voxels, dtype=np.float32)
        full_vol[brain_mask] = harmonized_brain[:, i]
        vol_3d = full_vol.reshape(shape)

        out_brain = os.path.join(s["pp3_dir"], f"{s['basename']}_preprocessed.nii.gz")
        nib.save(nib.Nifti1Image(vol_3d, affine, header), out_brain)

        # Copy GM as-is (or harmonize if requested)
        out_gm = os.path.join(s["pp3_dir"], f"{s['basename']}_gm.nii.gz")
        if os.path.exists(s["gm_path"]):
            import shutil
            shutil.copy2(s["gm_path"], out_gm)

        if (i + 1) % 100 == 0:
            print(f"  Saved {i+1}/{len(subjects)}")

    print(f"\nDone. Saved {len(subjects)} harmonized volumes to preprocessed3/ directories.")
    print("Update model script: change 'preprocessed2' -> 'preprocessed3' in build_entries_from_csv")


if __name__ == "__main__":
    main()