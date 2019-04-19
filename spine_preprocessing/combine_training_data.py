import os

out_path = os.path.join('data', 'training_data_combined')
if not os.path.isdir(out_path):
    os.makedirs(out_path)

folders = [os.path.join('data', 'hipp_vis_train'),
           os.path.join('data', 'purkinje_train_1'),
           os.path.join('data', 'purkinje_train_2')]

with open(os.path.join(out_path, 'train.txt'), 'w') as out_file:
    for folder in folders:
        with open(os.path.join(folder, 'train.txt')) as in_file:
            for line in in_file:
                out_file.write(line)

with open(os.path.join(out_path, 'validation.txt'), 'w') as out_file:
    for folder in folders:
        with open(os.path.join(folder, 'validation.txt')) as in_file:
            for line in in_file:
                out_file.write(line)
