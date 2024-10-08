# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, json, tqdm
import numpy as np
from argparse import ArgumentParser
from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='COCO data processing.',
        description='Create json files for unlabeled dataset.',
    )

    parser.add_argument('--img_folder', type=str, required=True)
    

    args = parser.parse_args()

    root = os.path.join(args.img_folder)

    new = {'annotations': [],
      'categories': [],
      'images': []}

    assert os.path.isdir(root) and len(os.listdir(root)) > 0, f'did not find folder {root} with images in it !'
    cnt = 0
    for x in tqdm.tqdm(os.listdir(root)):
        if x.endswith('.jpg') or x.endswith('.JPG'):
            # if cnt > 7000:
            #     break
            img_ = Image.open(os.path.join(root, x))
            width, height = img_.size
            if x.endswith('.jpg'):
                id_ = x.split('.jpg')[0] #'_'.join(x.split('_')[:3])
            if x.endswith('.JPG'):
                id_ = x.split('.JPG')[0]
            img_info = {
                'file_name': x,
                'id': id_,
                'height': height,
                'width': width,
            }
            annot_info = {
                'file_name': '',
                'image_id': id_,
                'segments_info': [],
            }
            # print(img_info)
            new['images'].append(img_info)
            new['annotations'].append(annot_info)
            cnt += 1

    with open(os.path.join(args.img_folder, 'coco_unlabel_train.json'), 'w') as f: #'./datasets/coco_unlabel/coco_unlabel_train.json', 'w') as f:
        json.dump(new, f)