import os, shutil

def create_dir(base_dir, name):
    new_dir = os.path.join(base_dir, name)
    if(os.path.isdir(new_dir) == False):
        os.mkdir(new_dir)
    return new_dir

dataset_dir = 'd:\code\ml\data'
base_dir = 'd:\code\ml\data\dogs_and_cats'
if(os.path.isdir(base_dir) == False):
    os.mkdir(base_dir)

train_dir = create_dir(base_dir, 'train')
val_dir = create_dir(base_dir, 'validation')
test_dir = create_dir(base_dir, 'test')
train_cats_dir = create_dir(train_dir, 'cats')
train_dogs_dir = create_dir(train_dir, 'dogs')
val_cats_dir = create_dir(val_dir, 'cats')
val_dogs_dir = create_dir(val_dir, 'dogs')
test_cats_dir = create_dir(base_dir, 'cats')
test_dogs_dir = create_dir(base_dir, 'dogs')
test_dir = create_dir(base_dir, 'train')
test_dir = create_dir(base_dir, 'train')
test_dir = create_dir(base_dir, 'train')


def copy_files(file_name, index_range, src_dir, dest_dir):
    fnames = ['{}.{}.jpg'.format(file_name, i) for i in index_range]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dest = os.path.join(dest_dir, fname)
        shutil.copyfile(src, dest)


copy_files('cat', range(1000), train_dir, val_cats_dir)


print('done')
