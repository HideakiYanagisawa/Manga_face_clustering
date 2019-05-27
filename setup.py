import os
import shutil
import random

TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'

if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)

cls_file = './clsname.txt'
with open(cls_file) as f:
    classes = [x.strip() for x in f.readlines()]

for name in classes:
    os.mkdir(os.path.join(TEST_DIR, name))

for d in os.listdir(TRAIN_DIR):
    files = os.listdir(os.path.join(TRAIN_DIR, d))
    random.shuffle(files)
    for f in files[:5]:
        sourse = os.path.join(TRAIN_DIR, d, f)
        dest = os.path.join(TEST_DIR, d)
        shutil.move(sourse, dest)
