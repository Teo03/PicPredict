import urllib.request
from config import *
import numpy as np
import glob
import os

def download_data():
    # downloads .npy files and saves them to NPY_DIR
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    for c in CLASSES:
        c_name = c.replace('_', '%20')
        final_url = f'{url}{c_name}.npy'
        
        print(f'saving: {c}')
        urllib.request.urlretrieve(final_url, f'./{NPY_DIR}/{c}.npy')

def create_dataset():
    npy_files = glob.glob(os.path.join(NPY_DIR, '*.npy'))

    # initialize X with one empty drawing
    X = np.empty([0, 784])
    y = np.empty([0])

    class_names = []

    for idx, file in enumerate(npy_files):
        file_data = np.load(file)
        # every drawing (total drawings: 144721) in file_data has 784 elements

        # create indexes for each of the drawings (total drawings: file_data.shape[0])
        indices = np.arange(0, file_data.shape[0])
        
        # randomly choose ITEMS_PER_CLASS from each class
        indices = np.random.choice(indices, ITEMS_PER_CLASS, replace=False)
        file_data = file_data[indices]

        # create 144721 (file_data.shape[0]) labels set to idx (0,1,2..)
        labels = np.full(file_data.shape[0], idx)
        
        # concatenate new drawing to X
        X = np.concatenate((X, file_data))
        # add the labels on the same axis (so they are inline)
        y = np.append(y, labels)

        class_name, _ = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

        print(f'loaded: {class_name}')

    # after loading, save and randomize the datasets

    # randomize
    permutation = np.random.permutation(y.shape[0])
    X = X[permutation, :]
    y = y[permutation]

    # create index at which there will be a split
    split_index = int(X.shape[0] * TEST_PCT)

    X_train = X[split_index:X.shape[0], :]
    y_train = y[split_index:y.shape[0]]

    X_test = X[0:split_index, :]
    y_test = y[0:split_index]

    np.savez_compressed(f'{DATA_DIR}/train', data=X_train, target=y_train)
    np.savez_compressed(f'{DATA_DIR}/test', data=X_test, target=y_test)

    print(f'train/test data saved to ./{DATA_DIR}')