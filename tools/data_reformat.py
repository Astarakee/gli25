import os
import shutil
import numpy as np
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_itk, write_nifti_from_vol
from tools.paths_dirs_stuff import path_contents_pattern, path_contents, create_path


def data_prepare(in_path, out_path):
    """
    reformulate the standard brats data structure into Decathlon file naming convention
    :param in_path: Abs path to standard BraTS data: each subject presented by a separate folder
    :param out_path: Abs path to saving image data in Decathlon format
    :return:
    """
    print('-'*8)
    subjects = path_contents(in_path)
    out_path_img = os.path.join(out_path, "imagesTr")
    out_path_mask = os.path.join(out_path, "labelsTr")
    create_path(out_path_img)
    create_path(out_path_mask)
    n_subjects = len(subjects)
    for ix, case in enumerate(subjects):
        print("data reformat in process for case {} out of {} ...".format(ix + 1, n_subjects))

        case_path = os.path.join(in_path, case)
        case_t1n = path_contents_pattern(case_path, 't1n.nii.gz')[0]
        case_t1c = path_contents_pattern(case_path, 't1c.nii.gz')[0]
        case_t2w = path_contents_pattern(case_path, 't2w.nii.gz')[0]
        case_t2f = path_contents_pattern(case_path, 't2f.nii.gz')[0]
        case_sg = path_contents_pattern(case_path, 'seg.nii.gz')[0]
        decathlon_t1n_name = case+'_0000.nii.gz'
        decathlon_t1c_name = case+'_0001.nii.gz'
        decathlon_t2w_name = case+'_0002.nii.gz'
        decathlon_t2f_name = case+'_0003.nii.gz'
        decathlon_sg_name = case + '.nii.gz'
        t1n_src = os.path.join(case_path, case_t1n)
        t1c_src = os.path.join(case_path, case_t1c)
        t2w_src = os.path.join(case_path, case_t2w)
        t2f_src = os.path.join(case_path, case_t2f)
        sg_src = os.path.join(case_path, case_sg)
        t1n_dst = os.path.join(out_path_img, decathlon_t1n_name)
        t1c_dst = os.path.join(out_path_img, decathlon_t1c_name)
        t2w_dst = os.path.join(out_path_img, decathlon_t2w_name)
        t2f_dst = os.path.join(out_path_img, decathlon_t2f_name)
        sg_dst = os.path.join(out_path_mask, decathlon_sg_name)

        if not os.path.exists(t1n_dst):
            shutil.copy(t1n_src, t1n_dst)
        if not os.path.exists(t1c_dst):
            shutil.copy(t1c_src, t1c_dst)
        if not os.path.exists(t2w_dst):
            shutil.copy(t2w_src, t2w_dst)
        if not os.path.exists(t2f_dst):
            shutil.copy(t2f_src, t2f_dst)
        if not os.path.exists(sg_dst):
            shutil.copy(sg_src, sg_dst)

    print('-' * 8)
    print('All files were reformated, ready for segmentation!')
    return None

def move_preds(nnunet_out, path_out):
    """
    moving the prediciton to the output folders
    :param nnunet_out: temp path where the preds are stored
    :param path_out: Abs path to save the results
    :return:
    """
    nifti_pred_files = path_contents_pattern(nnunet_out, '.nii.gz')
    for item in nifti_pred_files:
        src = os.path.join(nnunet_out, item)
        dst = os.path.join(path_out, item)
        shutil.move(src, dst)
    return None

def remove_additional_label(save_path_preds, save_path_preds_labelRemoved, remove_label):
    """
    Removing the additional predicted labels (context labels)
    :param save_path_preds: Abs path where the raw prediciton are stored
    :param save_path_preds_labelRemoved: Abs path where the cleaned prediciton will be stored
    :param remove_label: Integer showing the label to be removed
    :return:
    """
    if not os.path.exists(save_path_preds_labelRemoved):
        os.makedirs(save_path_preds_labelRemoved)
    pred_files = path_contents_pattern(save_path_preds, ".nii.gz")
    for pred_mask in pred_files:
        src_pred = os.path.join(save_path_preds, pred_mask)
        dst_pred = os.path.join(save_path_preds_labelRemoved, pred_mask)
        img_array, img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(src_pred)
        img_itk[img_itk==remove_label] = 0
        write_nifti_from_itk(img_itk, img_origin, img_spacing, img_direction, dst_pred)

    return None

