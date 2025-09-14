import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List
from tools.data_reformat import data_prepare, remove_additional_label, move_preds


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
    pipeline for the GLI-pre segmentatoin adopted from BraTS25 Task.
    1. Creates a temporary directory for all intermediate files.
    2. Reformat the input data into Decathlon format
    3. Runs inference using a trained model with nnUnet pipeline.
    4. Remove the contextual label and save the final segmentation results to the specified output directory.
    """
    base_temp_path = os.path.join(tempfile.gettempdir(), "gli_temp")
    # Define all intermediate paths relative to the temporary directory
    paths = {
        "nnunet_in_pre": base_temp_path + "/nnunetinput_pre",
        "nnunet_out1_pre": base_temp_path + "/nnunetpreds_pre1",
    }

    # Create all necessary subdirectories
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # --- Pre-operative model pipeline ---
    print("\nStep 2: Running Pre-operative Pipeline...")
    data_prepare(input_dir, str(paths["nnunet_in_pre"]))

    run_command([
        'nnUNetv2_predict', '-i', str(paths["nnunet_in_pre"]), '-o', str(paths["nnunet_out1_pre"]),
        '-d', 'Dataset771_BraTSGLIPreBrainCropRegion', '-c', '3d_fullres', '-p', 'nnUNetResEncUNetPlans'
    ])

    print("Step 3: Removing context label and save predicted masks...")
    remove_additional_label(str(paths["nnunet_out1_pre"]), output_dir, 4)

    print("\nPipeline finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline adopted from BraTS 25 for GLI-pre segmentation',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', type=str, help='Absolute path to the input data directory.\nThis directory should contain one subfolder per subject.', required=True)
    parser.add_argument('-o', type=str, help='Absolute path to the output directory where segmentation masks will be saved.', required=True)
    args = parser.parse_args()

    main(input_dir=args.i, output_dir=args.o)