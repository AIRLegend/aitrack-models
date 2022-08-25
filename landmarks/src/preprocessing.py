import pandas as pd
import os
import sys
import glob
import numpy as np
import argparse



def preproc_ls3d_video():
    print("---- PREPROCESSING LS3D-W (Video version) ----")

    base_path = os.path.abspath('./datasets/LS3D-W/')
    path =  os.path.abspath('./datasets/LS3D-W/300VW-3D/')

    train_imgs = glob.glob(path + '/*/*/*.jpg')
    
    if len(train_imgs) == 0:
        raise RuntimeError("Dataset direcotry not found. Are you sure it's there?")

    train_landmarks = [p.replace(".jpg", '.t7') for p in train_imgs]

    data = pd.DataFrame({'imgs': train_imgs, 'lms': train_landmarks})
    data['is_train'] = data.apply(lambda r: 0 if 'CatA' in r['imgs'] else 1, axis=1)

    print("   Ignoring nonexisting files...")
    # correct dataset (remove nonexisting files)
    valid_samples = data.apply(lambda r: os.path.exists(r['imgs']) and os.path.exists(r['lms']),axis=1)
    data = data[valid_samples]

    data.to_csv(os.path.join(base_path, "LS3D-W-Video.csv"), index=False)
    print(f"Split file written to {path}")


def preproc_ls3d():
    print("---- PREPROCESSING LS3D-W ----")

    base_path = os.path.abspath('./datasets/LS3D-W/')

    path =  os.path.abspath('./datasets/LS3D-W/LS3D-W_balanced/')
    imgs = glob.glob(path + '/*.jpg')
    t7s = [i.replace('.jpg', '.t7') for i in imgs]
    data_train = pd.DataFrame({'imgs': imgs, 'lms': t7s, 'is_train': 1})

    if len(data_train) == 0:
        raise RuntimeError("Dataset direcotry not found. Are you sure it's there?")

    path =  os.path.abspath('./datasets/LS3D-W/300W-Testset-3D/')
    imgs = glob.glob(path + '/*.png')
    t7s = [i.replace('.png', '.t7') for i in imgs]
    data_eval = pd.DataFrame({'imgs': imgs, 'lms': t7s, 'is_train': 0})

    if len(data_eval) == 0:
        raise RuntimeError("Test Dataset direcotry not found. Are you sure it's there?")
    
    data = pd.concat([data_train, data_eval])

    # Filter invalid entries
    valid_samples = data.apply(lambda r: os.path.exists(r['lms']), axis=1)
    data = data[valid_samples]

    data.to_csv(os.path.join(base_path, "LS3D-W.csv"), index=False)
    print(f"Split file written to {path}")



def preproc_menpo():
    print("---- PREPROCESSING MENPO ----")

    base_path = os.path.abspath('./datasets/LS3D-W/')

    path =  os.path.abspath('./datasets/LS3D-W/Menpo-3D/')
    imgs = glob.glob(path + '/*.jpg')
    t7s = [i.replace('.jpg', '.t7') for i in imgs]
    data_train = pd.DataFrame({'imgs': imgs, 'lms': t7s, 'is_train': 1})

    if len(data_train) == 0:
        raise RuntimeError("Dataset directory not found. Are you sure it's there?")

    path =  os.path.abspath('./datasets/LS3D-W/AFLW2000-3D-Reannotated/')
    imgs = glob.glob(path + '/*.jpg')
    t7s = [i.replace('.jpg', '.t7') for i in imgs]
    data_aflw = pd.DataFrame({'imgs': imgs, 'lms': t7s, 'is_train': 1})

    if len(data_aflw) == 0:
        raise RuntimeError("Reannotated AFLW Dataset direcotry not found. Are you sure it's there?")
    
    data = pd.concat([data_train, data_aflw])

    # Filter invalid entries
    valid_samples = data.apply(lambda r: os.path.exists(r['lms']), axis=1)
    data = data[valid_samples]

    # Train / test split
    data['is_train'] = (np.random.uniform(size=len(data)) > .3).astype(int)  # 70/30 split

    data.to_csv(os.path.join(base_path, "menpo.csv"), index=False)
    print(f"Split file written to {path}")



def main(args):
    dataset = args.dataset
    
    if dataset == 'ls3d':
        preproc_ls3d()

    if dataset == 'ls3d-video':
        preproc_ls3d_video()

    if dataset == 'menpo':
        preproc_menpo()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                    choices=['ls3d', 'ls3d-video', 'menpo'],
                    help='dataset for processing')

    args  = parser.parse_args(sys.argv[1:])

    main(args)