import os
import argparse
from tools.data_reformat import data_prepare, prepost_separate, ens_proces_pre, ens_proces_post, move_preds
from tools.paths_dirs_stuff import create_path


parser = argparse.ArgumentParser(description='BraTS2025 GLIs Task')
parser.add_argument('-i', type=str, help='abs path to data directory, subfolders stand for subjects', required=True)
parser.add_argument('-o', type=str, help='abs path to save masks', required=True)
args = parser.parse_args()
path_in = args.i
path_out = args.o

def main():
    data_path_pre = "/tmp/orig_pre"
    data_path_post = "/tmp/orig_post"
    nnunet_in_pre = "/tmp/nnunetinput_pre"
    nnunet_in_post = "/tmp/nnunetinput_post"

    nnunet_out1_pre = "/tmp/nnunetpreds_pre1"
    nnunet_out2_pre = "/tmp/nnunetpreds_pre2"

    nnunet_out1_post = "/tmp/nnunetpreds_post1"
    nnunet_out2_post = "/tmp/nnunetpreds_post2"
    nnunet_out3_post = "/tmp/nnunetpreds_post3"

    model_ens_path_PrePost = "/tmp/pred_pre_ens"

    create_path(data_path_pre)
    create_path(data_path_post)
    create_path(nnunet_in_pre)
    create_path(nnunet_in_post)
    create_path(nnunet_out1_pre)
    create_path(nnunet_out1_post)
    create_path(nnunet_out2_pre)
    create_path(nnunet_out2_post)
    create_path(nnunet_out3_post)
    create_path(model_ens_path_PrePost)

    prepost_separate(path_in, data_path_pre, data_path_post)
    # pre-operative model
    data_prepare(data_path_pre, nnunet_in_pre)
    os.system('nnUNetv2_predict -i %s -o %s -d Dataset771_BraTSGLIPreBrainCropRegion -c 3d_fullres -p nnUNetResEncUNetPlans' % (nnunet_in_pre, nnunet_out1_pre))
    os.system('mednextv1_predict -i %s -o %s -t Task773_BraTSGLIPreCropLabel -tr nnUNetTrainerV2_MedNeXt_L_kernel3 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_1x1x1 -f all' % (nnunet_in_pre, nnunet_out2_pre))
    ens_proces_pre(nnunet_out1_pre, nnunet_out2_pre, model_ens_path_PrePost)
    # post treatment model
    data_prepare(data_path_post, nnunet_in_post)
    os.system('nnUNetv2_predict -i %s -o %s -d Dataset760_BraTsGLIPosCropLabel -c 3d_fullres -p nnUNetResEncUNetPlans' % (nnunet_in_post, nnunet_out1_post))
    os.system('nnUNetv2_predict -i %s -o %s -d Dataset763_BraTsGLIPostBrainCropRegion -c 3d_fullres -p nnUNetResEncUNetPlans' % (nnunet_in_post, nnunet_out2_post))
    os.system('mednextv1_predict -i %s -o %s -t Task765_BraTSGLIPostCropLabelMedNeXt -tr nnUNetTrainerV2_MedNeXt_L_kernel3 -m 3d_fullres -p nnUNetPlansv2.1_trgSp_1x1x1 -f all' % (nnunet_in_post, nnunet_out3_post))
    ens_proces_post(nnunet_out1_post, nnunet_out2_post, nnunet_out3_post, model_ens_path_PrePost)

    move_preds(model_ens_path_PrePost, path_out)

    return None

if __name__ == '__main__':
    main()


#docker build . -t astarakee/brats25_gli:latest
#docker run --rm --gpus all --network none --memory="30g" -v /mnt/workspace/data/BraTS25/GLI/sample_test:/input -v /mnt/workspace/data/BraTS25/GLI/sample_pred:/output --shm-size 4g astarakee/brats25_gli:latest
