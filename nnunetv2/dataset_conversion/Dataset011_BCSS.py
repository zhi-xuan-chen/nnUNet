import multiprocessing
import shutil
from multiprocessing import Pool
import json

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    seg = io.imread(input_seg)
    seg[seg == 1] = 0
    # 除了 0 and 1 的像素值，其他的像素值都要减去 1
    seg[seg > 1] -= 1
    
    image = io.imread(input_image)

    io.imsave(output_seg, seg, check_contrast=False)
    io.imsave(output_image, image, check_contrast=False)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/data/chenzhixuan/data/BCSS/merged_dataset'

    dataset_name = 'Dataset011_BCSS'

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
        train_ids = json.load(open(join(source, 'train_val.json')))['image_ids']
        num_train = len(train_ids)
        r = []
        for v in train_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'img', v + '.jpg'),
                         join(source, 'mask', v + '.png'),
                         join(imagestr, v + '_0000.png'),
                         join(labelstr, v + '.png'),
                     ),)
                )
            )

        # test set
        test_ids = json.load(open(join(source, 'test.json')))['image_ids']
        for v in test_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(source, 'img', v + '.jpg'),
                         join(source, 'mask', v + '.png'),
                         join(imagests, v + '_0000.png'),
                         join(labelsts, v + '.png'),
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, 
                          {'background': 0, 'tumor': 1, 'stroma': 2, 'lymphocytic_infiltrate': 3, 'necrosis_or_debris': 4},
                          num_train, '.png', dataset_name=dataset_name)
