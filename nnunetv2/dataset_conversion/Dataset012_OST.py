import multiprocessing
import shutil
from multiprocessing import Pool
import json

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
# from acvl_utils.morphology.morphology_helper import generic_filter_components
# from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    seg = io.imread(input_seg)
    image = io.imread(input_image)

    io.imsave(output_seg, seg, check_contrast=False)
    io.imsave(output_image, image, check_contrast=False)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/jhcnas5/chenzhixuan/data/OstPathData/train_test'

    dataset_name = 'Dataset012_OST'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        train_ids = json.load(open(join('/jhcnas5/chenzhixuan/data/OstPathData/train.json')))['image_ids']
        num_train = len(train_ids)
        r = []
        for v in train_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'filtered_images', v + '.png'),
                         join(source, 'filtered_masks', v + '.png'),
                         join(imagestr, v + '_0000.png'),
                         join(labelstr, v + '.png'),
                     ),)
                )
            )

        # test set
        test_ids = json.load(open(join('/jhcnas5/chenzhixuan/data/OstPathData/test.json')))['image_ids']
        for v in test_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'filtered_images', v + '.png'),
                         join(source, 'filtered_masks', v + '.png'),
                         join(imagests, v + '_0000.png'),
                         join(labelsts, v + '.png'),
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, 
                          {'background': 0, 'Non-Bone Active Tumor': 1, 'Bone Active Tumor': 2, 'Non-Bone Necrosis': 3, 'Bone Necrosis': 4, 'Normal Tissue': 5, 'WSI_Background': 6, 'Sparse Cellular Area': 7},
                          num_train, '.png', dataset_name=dataset_name) 