## File used for mAP evaluation

import os
import ntpath
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='CreateGroundTruthFiles')
parser.add_argument('filename')
args = parser.parse_args()
if not os.path.exists('groundtruths'):
    os.mkdir('groundtruths')

with open(args.filename) as f:
    lines = f.readlines()
lines = [line for line in lines if len(line.strip()) > 0]
for line in lines:
    bounding_boxes = line.split()[1:]
    bounding_boxes = np.array([np.array(box.split(',')).astype(int) for box in bounding_boxes])
    path, file = ntpath.split(line.split()[0])
    ground_truths_file_path = os.path.join('groundtruths', ntpath.split(path)[1] + '_' + file + '.txt')
    with open(ground_truths_file_path, 'w') as f:
        for box in bounding_boxes:
            x, y, r, b = box[0], box[1], box[2], box[3]
            f.write('{} {} {} {} {}\n'.format('Spine', x, y, r, b))
