import pandas as pd
import os
import sys
import glob
import itertools
import argparse



def preproc_ls3d():
    print("---- PREPROCESSING LS3D-W ----")

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

    data.to_csv(os.path.join(base_path, "splits.csv"), index=False)
    print(f"Split file written to {path}")


def main(args):
    dataset = args.dataset
    
    if dataset == 'ls3d':
        preproc_ls3d()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                    choices=['ls3d'],
                    help='dataset for processing')

    args  = parser.parse_args(sys.argv[1:])

    main(args)