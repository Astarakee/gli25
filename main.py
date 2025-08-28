import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List
from tools.data_reformat import data_prepare, prepost_separate, ens_proces_pre, ens_proces_post, move_preds


def run_command(command: List[str]):
    """Runs an external command, checks for errors, and prints output on failure."""
    print(f"Running command: {' '.join(command)}")
    try:
        # Using capture_output to get stdout/stderr for better error reporting
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error output:\n{e.stderr}")
        raise  # Re-raise the exception to halt the script


def main(input_dir: str, output_dir: str):
    """
    Main pipeline for the BraTS2025 GLIs Task.

    This script orchestrates a multi-step process:
    1. Creates a temporary directory for all intermediate files.
    2. Separates pre- and post-operative data from the input directory.
    3. Runs inference using an ensemble of models for both pre- and post-op data.
    4. Ensembles the predictions from the different models.
    5. Moves the final segmentation masks to the specified output directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_temp_path = Path(tmpdir)
        print(f"Using temporary directory: {base_temp_path}")

        # Define all intermediate paths relative to the temporary directory
        paths = {
            "orig_pre": base_temp_path / "orig_pre",
            "orig_post": base_temp_path / "orig_post",
            "nnunet_in_pre": base_temp_path / "nnunetinput_pre",
            "nnunet_in_post": base_temp_path / "nnunetinput_post",
            "nnunet_out1_pre": base_temp_path / "nnunetpreds_pre1",
            "nnunet_out2_pre": base_temp_path / "nnunetpreds_pre2",
            "nnunet_out1_post": base_temp_path / "nnunetpreds_post1",
            "nnunet_out2_post": base_temp_path / "nnunetpreds_post2",
            "nnunet_out3_post": base_temp_path / "nnunetpreds_post3",
            "pred_ens": base_temp_path / "pred_ens",
        }

        # Create all necessary subdirectories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        print("Step 1: Separating pre- and post-operative data...")
        prepost_separate(input_dir, str(paths["orig_pre"]), str(paths["orig_post"]))

        # --- Pre-operative model pipeline ---
        print("\nStep 2: Running Pre-operative Pipeline...")
        data_prepare(str(paths["orig_pre"]), str(paths["nnunet_in_pre"]))

        run_command([
            'nnUNetv2_predict', '-i', str(paths["nnunet_in_pre"]), '-o', str(paths["nnunet_out1_pre"]),
            '-d', 'Dataset771_BraTSGLIPreBrainCropRegion', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetPlans'
        ])
        run_command([
            'mednextv1_predict', '-i', str(paths["nnunet_in_pre"]), '-o', str(paths["nnunet_out2_pre"]),
            '-t', 'Task773_BraTSGLIPreCropLabel', '-tr', 'nnUNetTrainerV2_MedNeXt_L_kernel3',
            '-m', '3d_fullres', '-p', 'nnUNetPlansv2.1_trgSp_1x1x1', '-f', 'all'
        ])

        print("Step 3: Ensembling pre-operative predictions...")
        ens_proces_pre(str(paths["nnunet_out1_pre"]), str(paths["nnunet_out2_pre"]), str(paths["pred_ens"]))

        # --- Post-operative model pipeline ---
        print("\nStep 4: Running Post-operative Pipeline...")
        data_prepare(str(paths["orig_post"]), str(paths["nnunet_in_post"]))

        run_command([
            'nnUNetv2_predict', '-i', str(paths["nnunet_in_post"]), '-o', str(paths["nnunet_out1_post"]),
            '-d', 'Dataset760_BraTsGLIPosCropLabel', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetPlans'
        ])
        run_command([
            'nnUNetv2_predict', '-i', str(paths["nnunet_in_post"]), '-o', str(paths["nnunet_out2_post"]),
            '-d', 'Dataset763_BraTsGLIPostBrainCropRegion', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetPlans'
        ])
        run_command([
            'mednextv1_predict', '-i', str(paths["nnunet_in_post"]), '-o', str(paths["nnunet_out3_post"]),
            '-t', 'Task765_BraTSGLIPostCropLabelMedNeXt', '-tr', 'nnUNetTrainerV2_MedNeXt_L_kernel3',
            '-m', '3d_fullres', '-p', 'nnUNetPlansv2.1_trgSp_1x1x1', '-f', 'all'
        ])

        print("Step 5: Ensembling post-operative predictions...")
        ens_proces_post(
            str(paths["nnunet_out1_post"]),
            str(paths["nnunet_out2_post"]),
            str(paths["nnunet_out3_post"]),
            str(paths["pred_ens"])
        )

        print("\nStep 6: Moving final predictions to output directory...")
        move_preds(str(paths["pred_ens"]), output_dir)

        print("\nPipeline finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BraTS2025 GLIs Task Prediction Pipeline',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', type=str, help='Absolute path to the input data directory.\nThis directory should contain one subfolder per subject.', required=True)
    parser.add_argument('-o', type=str, help='Absolute path to the output directory where segmentation masks will be saved.', required=True)
    args = parser.parse_args()

    main(input_dir=args.i, output_dir=args.o)
