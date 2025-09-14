# Glioma Segmentation Pipeline for BraTS 2025

This repository contains a pipeline for the segmentation of Glioma in pre-treatment and post-operative MRIs, developed for a contribution to the BraTS 2025 Lighthouse challenge.

## Overview

The pipeline processes multi-modal MRI scans (T1-weighted pre-contrast, T1-weighted post-contrast, T2-weighted, and T2-FLAIR) to produce segmentation masks of different tumor sub-regions.

The workflow is as follows:
1.  **Data Separation**: Renaming the data into Decathlon filenaming convention.
2.  **Inference**:  segmentation model is run on the Decathlon formated data by using a trained model based on nnUnet.
3.  **LabelRemoval**: The model was trained to segment an additional context label which will be removed in this step.
4.  **Output**: The final segmentation masks are saved to a specified output directory.

## Dependencies

This pipeline relies on the following pre-installed models:
*   [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
Please download the model checkpoint and put it in the right directory which is available in your environment's PATH

e.g. `export nnUNet_results="env PATH"` 

## Data Preparation

The input data must follow the BraTS standard naming convention. Each subject's data should be in its own folder, containing four NIfTI files (`.nii.gz`) corresponding to the four MRI sequences:

*   `...-t1c.nii.gz`: T1-weighted post-contrast
*   `...-t1n.nii.gz`: T1-weighted pre-contrast (native)
*   `...-t2w.nii.gz`: T2-weighted
*   `...-t2f.nii.gz`: T2-FLAIR

An example of the data format that this model works with is:
```
├── BraTS-GLI-00001-000
│   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   └── BraTS-GLI-00001-000-t2w.nii.gz
├── BraTS-GLI-00001-001
│   ├── BraTS-GLI-00001-001-t1c.nii.gz
│   ├── BraTS-GLI-00001-001-t1n.nii.gz
│   ├── BraTS-GLI-00001-001-t2f.nii.gz
│   └── BraTS-GLI-00001-001-t2w.nii.gz
├── BraTS-GLI-00013-000
│   ├── BraTS-GLI-00013-000-t1c.nii.gz
│   ├── BraTS-GLI-00013-000-t1n.nii.gz
│   ├── BraTS-GLI-00013-000-t2f.nii.gz
│   └── BraTS-GLI-00013-000-t2w.nii.gz
├── BraTS-GLI-00013-001
│   ├── BraTS-GLI-00013-001-t1c.nii.gz
│   ├── BraTS-GLI-00013-001-t1n.nii.gz
│   ├── BraTS-GLI-00013-001-t2f.nii.gz
│   └── BraTS-GLI-00013-001-t2w.nii.gz
└── BraTS-GLI-00015-000
    ├── BraTS-GLI-00015-000-t1c.nii.gz
    ├── BraTS-GLI-00015-000-t1n.nii.gz
    ├── BraTS-GLI-00015-000-t2f.nii.gz
    └── BraTS-GLI-00015-000-t2w.nii.gz
```



## Usage

To run the segmentation pipeline, execute the `main.py` script from the root of the project directory. You must provide the absolute paths to the input data directory and the desired output directory.

```bash
python main.py -i /path/to/your/input_data -o /path/to/your/output_directory
```

### Arguments
*   `-i`: Absolute path to the input data directory. This directory should contain one subfolder per subject.
*   `-o`: Absolute path to the output directory where segmentation masks will be saved.

The final segmentation masks will be saved in the output directory with the same naming convention as the input subjects.

### Checkpoints
model checkpoints can be found from the following [Gdrive](https://drive.google.com/drive/folders/1-vapUxHYaedN-vvjr-LrG0rQLH_zWkME?usp=sharing)

please download, and unzip it, then put `Dataset771_BraTSGLIPreBrainCropRegion` into the `nnUNet_results PATH`.