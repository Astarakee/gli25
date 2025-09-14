import os
import shutil
import numpy as np
from tools.sitk_stuff import read_nifti
from tools.writer import write_nifti_from_itk, write_nifti_from_vol
from tools.paths_dirs_stuff import path_contents_pattern, path_contents, create_path


def prepost_separate(in_path, data_path_pre, data_path_post):
    """
    separating the pre from post treatmemnt data based in BaTS naming convention
    :param in_path: path to input directory containing all data; each subject are presented in a separate folder
    :param data_path_pre: abs path to copy the pre operative data
    :param data_path_post: abs path to copy the post operative data
    :return:
    """
    subjects = path_contents(in_path)
    for item in subjects:
        src = os.path.join(in_path, item)
        item_order_scan = item[-3]
        if item_order_scan == "0":
            dst = os.path.join(data_path_pre, item)
        elif item_order_scan == "1":
            dst = os.path.join(data_path_post, item)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return None

def data_prepare(in_path, out_path):
    """
    reformulate the standard brats data structure into Decathlon file naming convention
    :param in_path: Abs path to standard BraTS data: each subject presented by a separate folder
    :param out_path: Abs path to saving image data in Decathlon format
    :return:
    """
    print('-'*8)
    subjects = path_contents(in_path)
    n_subjects = len(subjects)
    for ix, case in enumerate(subjects):
        print("data reformat in process for case {} out of {} ...".format(ix + 1, n_subjects))

        case_path = os.path.join(in_path, case)
        case_t1n = path_contents_pattern(case_path, 't1n.nii.gz')[0]
        case_t1c = path_contents_pattern(case_path, 't1c.nii.gz')[0]
        case_t2w = path_contents_pattern(case_path, 't2w.nii.gz')[0]
        case_t2f = path_contents_pattern(case_path, 't2f.nii.gz')[0]
        decathlon_t1n_name = case+'_0000.nii.gz'
        decathlon_t1c_name = case+'_0001.nii.gz'
        decathlon_t2w_name = case+'_0002.nii.gz'
        decathlon_t2f_name = case+'_0003.nii.gz'
        t1n_src = os.path.join(case_path, case_t1n)
        t1c_src = os.path.join(case_path, case_t1c)
        t2w_src = os.path.join(case_path, case_t2w)
        t2f_src = os.path.join(case_path, case_t2f)
        t1n_dst = os.path.join(out_path, decathlon_t1n_name)
        t1c_dst = os.path.join(out_path, decathlon_t1c_name)
        t2w_dst = os.path.join(out_path, decathlon_t2w_name)
        t2f_dst = os.path.join(out_path, decathlon_t2f_name)

        if not os.path.exists(t1n_dst):
            shutil.copy(t1n_src, t1n_dst)
        if not os.path.exists(t1c_dst):
            shutil.copy(t1c_src, t1c_dst)
        if not os.path.exists(t2w_dst):
            shutil.copy(t2w_src, t2w_dst)
        if not os.path.exists(t2f_dst):
            shutil.copy(t2f_src, t2f_dst)

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


def ens_proces_pre(nnunet_out1_pre, nnunet_out2_pre, model_ens_path_PrePost):
    """
    Experimental label merging among the predcted masks by different models for pre-treatment Glioma
    :param nnunet_out1_pre: Abs path where the raw prediciton from the 1st model are stored
    :param nnunet_out2_pre: Abs path where the raw prediciton from the 2nd model are stored
    :param model_ens_path_PrePost: Abs path where the merged labels will be saved
    :return:
    """
    pred_list1 = path_contents_pattern(nnunet_out1_pre, ".nii.gz")
    for item in pred_list1:
        src_item1 = os.path.join(nnunet_out1_pre, item)
        src_item2 = os.path.join(nnunet_out2_pre, item)
        dst_item = os.path.join(model_ens_path_PrePost, item)
        img_array, img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(src_item1)
        img_array2, _, _, _, _, _ = read_nifti(src_item2)
        new_array = np.zeros_like(img_array)
        new_array[img_array2 == 3] = 3
        new_array[img_array2 == 1] = 1
        new_array[img_array == 2] = 2
        write_nifti_from_vol(new_array, img_origin, img_spacing, img_direction, dst_item)
    return None


def ens_proces_post(nnunet_out1_post, nnunet_out2_post, nnunet_out3_post, model_ens_path_PrePost):
    """
    Experimental label merging among the predcted masks by different models for post-treatment Glioma
    :param nnunet_out1_post:Abs path where the raw prediciton from the 1st model are stored
    :param nnunet_out2_post:Abs path where the raw prediciton from the 2nd model are stored
    :param nnunet_out3_post:Abs path where the raw prediciton from the 3rd model are stored
    :param model_ens_path_PrePost: Abs path where the merged labels will be saved
    :return:
    """
    pred_list1 = path_contents_pattern(nnunet_out1_post, ".nii.gz")
    for item in pred_list1:
        src_item1 = os.path.join(nnunet_out1_post, item)
        src_item2 = os.path.join(nnunet_out2_post, item)
        src_item3 = os.path.join(nnunet_out3_post, item)
        dst_item = os.path.join(model_ens_path_PrePost, item)
        img_array, img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(src_item1)
        img_array2, _, _, _, _, _ = read_nifti(src_item2)
        img_array3, _, _, _, _, _ = read_nifti(src_item3)

        new_array = np.zeros_like(img_array)
        new_array[img_array == 2] = 2  # SNFH
        new_array[img_array2 == 3] = 3  # ET
        new_array[img_array3 == 1] = 1  # NETC
        new_array[img_array3 == 4] = 4  # RC

        write_nifti_from_vol(new_array, img_origin, img_spacing, img_direction, dst_item)

    return None