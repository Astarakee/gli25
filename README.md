# Glioma Segmentation Pipeline for BraTS 2025

This repository contains a pipeline for the segmentation of Glioma in pre-treatment and post-operative MRIs, developed for a contribution to the BraTS 2025 Lighthouse challenge.

## Overview

The pipeline processes multi-modal MRI scans (T1-weighted pre-contrast, T1-weighted post-contrast, T2-weighted, and T2-FLAIR) to produce segmentation masks of different tumor sub-regions. It leverages an ensemble of deep learning models to achieve robust and accurate results.

The workflow is as follows:
1.  **Data Separation**: Pre- and post-operative scans are separated based on their filenames.
2.  **Inference**: Separate segmentation models are run on the pre- and post-operative data.
3.  **Ensembling**: The predictions from the different models are combined to generate a final segmentation.
4.  **Output**: The final segmentation masks are saved to a specified output directory.

## Dependencies

This pipeline relies on the following pre-installed models:
*   [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
*   [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt)

Please ensure that these frameworks are properly installed and their prediction scripts (`nnUNetv2_predict` and `mednextv1_predict`) are available in your environment's PATH.

## Data Preparation

The input data must follow the BraTS standard naming convention. Each subject's data should be in its own folder, containing four NIfTI files (`.nii.gz`) corresponding to the four MRI sequences:

*   `...-t1c.nii.gz`: T1-weighted post-contrast
*   `...-t1n.nii.gz`: T1-weighted pre-contrast (native)
*   `...-t2w.nii.gz`: T2-weighted
*   `...-t2f.nii.gz`: T2-FLAIR

Example directory structure:
```
input_directory/
└── BraTS-GLI-00001-000/
    ├── BraTS-GLI-00001-000-t1c.nii.gz
    ├── BraTS-GLI-00001-000-t1n.nii.gz
    ├── BraTS-GLI-00001-000-t2f.nii.gz
    └── BraTS-GLI-00001-000-t2w.nii.gz
└── BraTS-GLI-00002-001/
    ├── BraTS-GLI-00002-001-t1c.nii.gz
    ├── BraTS-GLI-00002-001-t1n.nii.gz
    ├── BraTS-GLI-00002-001-t2f.nii.gz
    └── BraTS-GLI-00002-001-t2w.nii.gz
...
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
All model checkpoints can be found from the following [Gdrive](https://drive.google.com/drive/folders/1-vapUxHYaedN-vvjr-LrG0rQLH_zWkME?usp=sharing)
Please follow `nnU-Net` and `MedNeXt` instructions to properly integrate the model checkpoints.
Generally speaking, the model names begin with `Dataset` belong to `nnUnetV2` pipeline while those begings with `Task` 
belong to `MedNeXt` pipeline.

