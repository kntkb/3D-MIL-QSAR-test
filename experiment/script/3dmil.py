"""
3D multi-instance machine learning.
"""
import sys, os
import pickle
import numpy as np
import pandas as pd
import logging


def calc_descriptor(filename, nconf, ncpu):
    """
    Calculate descriptors. 
    
    This will create the several files under `descriptor` folder:
        - conf-CHEMBL1075104_1.pkl - pickle file with RDKit the lowest-energy conformations
        - conf-CHEMBL1075104_5.pkl - pickle file with RDKit the generated conformations
        - conf-5_CHEMBL1075104_log.pkl - pickle file with the conformation energies
        - PhFprPmapper_conf-CHEMBL1075104_1.txt - SVM file with pmapper 3D descriptors for the lowest-energy conformatons
        - PhFprPmapper_conf-CHEMBL1075104_1.colnames - names of pmapper 3D descriptors for the lowest-energy conformatons
        - PhFprPmapper_conf-catalyst_data_1.rownames - ids of the lowest-energy conformatons
        - PhFprPmapper_conf-CHEMBL1075104_5.txt - SVM file with pmapper 3D descriptors for generated conformations
        - PhFprPmapper_conf-CHEMBL1075104_5.colnames - names of pmapper 3D descriptors for generated conformations
        - PhFprPmapper_conf-CHEMBL1075104_5.rownames - ids of generated conformations

    Parameters
    ----------
    filename : str
        Path of the filename of the dataset.
    nconfs : int, default=5
        Number of conformations to generate.
        Calculation is time consuming, so here we set the default to 5.
        Increase number for production run.
    ncpu : int
        Number of CPU cores.

    Returns
    -------
    """
    # Settings
    nconfs_list = [1, nconfs]
    ncpu = ncpu
    dataset_file = filename
    descriptor_folder = os.path.join('descriptor')
    if not os.path.exists('descriptor'):
        os.mkdirs('descriptor')

    from miqsar.utils import calc_3d_pmapper
    out_fname = calc_3d_pmapper(input_fname=dataest_file, nconfs_list=nconfs_list, energy=100,  descr_num=[4],
                                path=descriptors_folder, ncpu=ncpu)



# ----------------------------
# SUBMODULE TO PREPARE DATASET
# ----------------------------
def _str_to_vec(dsc_str, dsc_num):
    tmp = {}
    for i in dsc_str.split(' '):
        tmp[int(i.split(':')[0])] = int(i.split(':')[1])
    tmp_sorted = {}
    for i in range(dsc_num):
        tmp_sorted[i] = tmp.get(i, 0)
    vec = list(tmp_sorted.values())
    return vec


def prepare_dataset(descriptor_filename):
    """
    Prepare traing and test set.

    Parameters
    ----------
    descriptor_filename : str
        Path to the descriptor filename.

    Returns
    -------
    x_train : numpy array
    x_test : numpy array
    y_train : numpy array
    y_test : numpy array
    idx_train : numpy array
    idx_test : numpy array
    """

    # Check extension
    assert os.path.splitext(descriptor_filenam) == "txt"

    # Load files
    with open(descriptor_filename) as f:
        dsc_tmp = [i.strip() for i in f.readlines()]
    with open(descriptor_filename.replace('txt', 'rownames')) as f:
        mol_names = [i.strip() for i in f.readlines()]
    
    # 
    labels_tmp = [float(i.split(':')[1]) for i in mol_names]
    idx_tmp = [i.split(':')[0] for i in mol_names]
    dsc_num = max([max([int(j.split(':')[0]) for j in i.strip().split(' ')]) for i in dsc_tmp])
    
    # 
    bags, labels, idx = [], [], []
    for mol_idx in list(np.unique(idx_tmp)):
        bag, labels_, idx_ = [], [], []
        for dsc_str, label, i in zip(dsc_tmp, labels_tmp, idx_tmp):
            if i == mol_idx:
                bag.append(_str_to_vec(dsc_str, dsc_num))
                labels_.append(label)
                idx_.append(i)
        bags.append(np.array(bag).astype('uint8'))
        labels.append(labels_[0])
        idx.append(idx_[0])

    # TODO: change to logging
    print(f'There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')

    # Split dataset
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(bags, labels, idx)
    # TODO: change to logging
    print(f'There are {len(x_train)} training molecules and {len(x_test)} test molecules')

    # Save numpy
    np.savez_compressed(
        file='input.npz', 
        x_train=xtrain,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        idx_train=idx_train,
        idx_test=idx_test
        )

    return x_train, x_test, y_train, y_test, idx_train, idx_test


# ----------------------------
# SUBMODULE FOR TRAINING MODEL
# ----------------------------
def _scale_data(x_train, x_test):
    """
    Scale dataset.

    Parameters
    ----------
    x_train : numpy array
    x_test : numpy array

    Returns
    -------
    x_train_scaled : numpy array
    x_test_scaled : numpy array
    """
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(x_train))
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    
    for i, bag in enumerate(x_train):
        x_train_scaled[i] = scaler.transform(bag)
    for i, bag in enumerate(x_test):
        x_test_scaled[i] = scaler.transform(bag)
    
    x_train_scaled, x_test_scaled = np.array(x_train_scaled), np.array(x_test_scaled)
    
    return x_train_scaled, x_test_scaled


def train_model(config, x_train, x_test):
    """
    Train model.

    Parameters
    ----------
    config : dictionary
        Configuration of the network model.
        config[method] : str, default="instance"
            Wrapping method to use. Choices are "instance" or "bag".
        config[layers] : list of int, default=[256, 128, 64]
            List of number of hidden layers in the main network.
        config[pool] : str, default="mean"
            Pooling method. Choices are "mean", "max", and "min".
        config[epochs] : int, default=1000
            Maximum number of learning epochs.
        config[lr] : float, default=0.001
            Learning rate.
        config[weight_decay] : float, default=0.001
            Weight decay for L2 regularization.
        config[batch_size] : int, default=99999999
            Batch size.
        config[dropout] : float, default=0.0
            Dropout for training.
        config[use_gpu] : boolean, default=True,
            Train using gpu if True. Otherwise use cpu.
        config[random_seed] : str, default=43
            Random seed 
    x_train : numpy array
    x_test : numpy array

    Returns
    -------
    """
    from miqsar.estimators.wrappers import InstanceWrapperMLPRegressor, BagWrapperMLPRegressor
    from miqsar.estimators.utils import set_seed

    # Scaled dataset
    x_train_scaled, x_test_scaled = _scale_data(x_train, x_test)

    # Define model
    set_seed(43)
    ndim = (x_train_scaled[0].shape[1], 256, 128, 64)
    pool = 'mean'
    n_epoch = 1000
    lr = 0.001
    weight_decay = 0.001
    batch_size = 99999999
    dropout = 0
    init_cuda = True

    # Train
    if wrapper_method == "instance":
        net = InstanceWrapperMLPRegressor(ndim=ndim, pool=pool, init_cuda=init_cuda)
    else:
        net = BagWrapperMLPRegressor(ndim=ndim, pool=pool, init_cuda=init_cuda)
    net.fit(x_train_scaled, y_train, 
            n_epoch=n_epoch, 
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            dropout=dropout)

    # Metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    y_pred = net.predict(x_test_scaled)
    print('3D/MI/Instance-Wrapper: r2_score test = {:.2f}'.format(r2_score(y_test, y_pred)))



#
# Run
#
def _config(kwargs):

    return config


def run(kwargs):
    """
    """
    config = {}
    for k in kwargs:
        if k == filename:
            filename = kwargs[filename]
        if k != "filename":
            config[str(k)] = k

    # Calculate 3D descriptors
    calc_3d_descriptors()

    # Split data into a training and test set
    descriptor_filename = os.path.join('descriptors', 'PhFprPmapper_conf-CHEMBL1075104_5.txt') # descriptors file
    x_train, x_test, y_train, y_test, idx_train, idx_test = prepare_dataset(descriptor_filename)
    
    # Train model
    train_model()

    # Run baseline qsar?
    if baseline == True:
        run_baseline()


@click.command()
@click.option("--input_prefix", required=True, help="path to input file")
@click.option("--ref_prefix", required=True, help="path to reference files")
@click.option("--keep_solvent", default=True, help="keep solvent during analysis")
def cli(**kwargs):
    run(kwargs)



if __name__ == '__main__':
    cli()