from fastai.conv_learner import *
from matplotlib import patches, patheffects

coco_path = Path('/scratch/arka/Ark_git_files/coco/')
ann_path = coco_path / 'annotations'
train_path = coco_path / 'train2017'
val_path = coco_path / 'val2017'

captions_train2017 = json.load((ann_path / 'captions_train2017.json').open('r'))
captions_val2017 = json.load((ann_path / 'captions_val2017.json').open('r'))
instances_train2017 = json.load((ann_path / 'instances_train2017.json').open('r'))
instances_val2017 = json.load((ann_path / 'instances_val2017.json').open('r'))
person_keypoints_train2017 = json.load((ann_path / 'person_keypoints_train2017.json').open('r'))
person_keypoints_val2017 = json.load((ann_path / 'person_keypoints_val2017.json').open('r'))
