import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

# Load MRI scan (.nii or .nii.gz)
img = nib.load(r"C:\Users\sucha\OneDrive\Documents\schizobrain-Scan\MRI_data\open_neuro\open_neuro_4_99\sub-01_T1w\sub-01_T1w.nii")   # change path if needed
data = img.get_fdata()

print("MRI Shape:", data.shape)

# Function to display slice
def view_slice(slice_index):
    plt.figure(figsize=(6,6))
    plt.imshow(data[:, :, slice_index], cmap="gray")
    plt.title(f"Slice {slice_index}")
    plt.axis("off")
    plt.show()

# Create interactive slider
interact(
    view_slice,
    slice_index=IntSlider(
        min=0,
        max=data.shape[2]-1,
        step=1,
        value=data.shape[2]//2
    )
)


# Load MRI scan (.nii or .nii.gz)
# img = nib.load(r"C:\Users\sucha\OneDrive\Documents\schizobrain-Scan\MRI_data\open_neuro\open_neuro_4_99\sub-01_T1w\sub-01_T1w_reoriented.nii.gz")   # change path if needed
# data = img.get_fdata()

# print("MRI Shape:", data.shape)

# # Function to display slice
# def view_slice(slice_index):
#     plt.figure(figsize=(6,6))
#     plt.imshow(data[:, :, slice_index], cmap="gray")
#     plt.title(f"Slice {slice_index}")
#     plt.axis("off")
#     plt.show()

# # Create interactive slider
# interact(
#     view_slice,
#     slice_index=IntSlider(
#         min=0,
#         max=data.shape[2]-1,
#         step=1,
#         value=data.shape[2]//2
#     )
# )

