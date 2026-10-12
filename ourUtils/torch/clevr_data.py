import json
import os
import pickle
from PIL import Image
import shutil
from collections import Counter
from torch.utils.data import Dataset
import numpy as np
import ourUtils.torch.clevr_utils as utils
import torch
from collections import OrderedDict
from ourUtils.funct_band_aids import setup_directory


"""
Code adapted from https://github.com/mesnico/RelationNetworks-CLEVR/edit/master/clevr_dataset_connector.py
to parse the json files easily.
"""


used_classes = OrderedDict({
    'count': ['1'],
    'material': ['rubber', 'metal'],
    'color': ['blue', 'red', 'green', 'gray', 'yellow', 'purple'],
    'shape': ['sphere', 'cube', 'cylinder'],
    'size': ['large'],
})


def load_json(scene_path):
    with open(scene_path) as json_file:
        data = json.load(json_file)
    return data


def save_json(data, scene_path):
    with open(scene_path, 'w') as outfile:
        json.dump(data, outfile)


def generate_scenes(scene_dir, nr_of_images, mode="train"):
    wrapper_d = {"scenes": [],
                 'info': {'split': 'train',
                          'license': 'Creative Commons Attribution (CC-BY 4.0)',
                          'version': '1.0',
                          'date': '07/24/2019'}}
    for i in range(nr_of_images):
        padded_index = str(i).rjust(6, '0')
        scene_path = os.path.join(
            scene_dir, 'CLEVR_{}_{}.json'.format(mode, padded_index))
        data = load_json(scene_path)
        wrapper_d["scenes"].append(data.copy())
    return wrapper_d


def clean_labels(labels):
    l = np.array(labels).squeeze(1)
    print("Labels: ")
    for col_idx in range(l.shape[1]):
        label_list = l[:, col_idx].tolist()
        unique_labels = list(set(label_list))
        unique_labels.sort()
        print(unique_labels)
        mapping = {}
        for label_idx, label_i in enumerate(unique_labels):
            mapping[label_i] = label_idx

        l[:, col_idx] = [mapping[i] for i in label_list]
    l = l.astype(int)
    return l


def get_raw_labels(scene_json_filename):
    with open(scene_json_filename, 'rb') as f:
        raw_labels = json.load(f)
    return raw_labels


def get_constraint_indices(scene_dir, constraints):
    scene_labels = clevr_data.get_raw_labels(scene_dir)
    n_labels = len(scene_labels)
    label_indices = []
    for i in range(n_labels):
        n_objects = len(scene_labels[i]["objects"])
        if n_objects in constraints["count"]:
            success = 0
            for obj_i in scene_labels[i]["objects"]:
                for key in constraints.keys():
                    if key != "count":
                        if obj_i[key] in constraints[key]:
                            success += 1
            if success == n_objects * 4:
                label_indices.append(scene_labels[i]["image_index"])
    return label_indices


class ClevrDatasetImagesAndDescriptions(Dataset):
    """
    Loads only images and scene descriptions from the CLEVR dataset
    """

    def __init__(self, clevr_dir, train, transform=None, use_cached=True, classes=None):
        """
        :param clevr_dir: Root directory of CLEVR dataset
        :param mode: Specifies if we want to read in val, train or test folder
        :param transform: Optional transform to be applied on a sample.
        :param use_cached: Optional flag to force to compute labels again
        """
        self.mode = 'train' if train else 'val'
        self.img_dir = os.path.join(clevr_dir, 'images', self.mode)
        self.transform = transform
        self.classes = classes if classes is not None else small_classes
        if train:
            self.scene_json_filename = os.path.join(
                clevr_dir, 'scenes', 'CLEVR_train_scenes.json')
        else:
            self.scene_json_filename = os.path.join(
                clevr_dir, 'scenes', 'CLEVR_val_scenes.json')

        cached_scenes = self.scene_json_filename.replace('.json', '.pkl')
        if use_cached and os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            all_scene_objs = OrderedDict({})
            with open(self.scene_json_filename, 'r') as json_file:
                scenes = json.load(json_file)['scenes']
                print('caching all labels in all scenes...')
                for s in scenes:
                    labels = s['objects']
                    labels_attr = []
                    for obj in labels:
                        attr_values = []
                        for attr in sorted(obj):
                            # convert object attributes in indexes
                            if attr in self.classes:
                                attr_values.append(obj[attr])
                            else:
                                if attr == '3d_coords':
                                    attr_values.append(str(tuple(obj[attr])))
                            #     '''if attr=='rotation':
                            #         attr_values.append(float(obj[attr]) / 360)'''
                        labels_attr.append(attr_values)
                    # all_scene_objs.append(torch.FloatTensor(labels_attr))
                    all_scene_objs[s["image_index"]] = labels_attr

                self.labels = all_scene_objs
            with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)

    def get_raw_labels(self):
        return get_raw_labels(self.scene_json_filename)

    def __len__(self):
        return len(self.labels)

    def get_pt_data(self, N):
        """Returns two pytorch.tensor of length N for data and labels"""
        pt_data = []
        pt_labels = []
        for i in range(N):
            img, label = self.__getitem__(i)
            pt_data.append(img)
            pt_labels.append(label)
        pt_data = torch.cat(pt_data, dim=0)
        pt_labels = clean_labels(pt_labels)
        return pt_data, pt_labels

    def __getitem__(self, idx):
        padded_index = str(idx).rjust(6, '0')
        img_filename = os.path.join(
            self.img_dir, 'CLEVR_{}_{}.png'.format(self.mode, padded_index))
        image = Image.open(img_filename).convert('RGB')
        labels = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, labels
