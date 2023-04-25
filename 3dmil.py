"""
3D multi-instance machine learning.
"""
import sys, os
import pickle
import numpy as np
import pandas as pd
import glob
import click
import logging


# Settings
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# ----------------------------------
# SUBMODULE TO CALCULATE DESCRIPTORS
# ----------------------------------
def calc_descriptors(config):
    """ Calculate descriptors.

    """
    method = config["descriptor"]["method"]
    filename = os.path.join(config["option"]["path_to_dataset"], config["option"]["dataset"] + ".smi")

    if not os.path.exists('descriptor'):
        os.mkdir('descriptor')

    if method == "pmapper":
        calc_pmapper(filename, config)
    elif method == "rdkit_2d":
        calc_rdkit_2d(filename)
    elif method == "rdkit_morgan":
        calc_rdkit_morgan(filename)


def calc_pmapper(filename, config, energy_threshold=10, nconfs=5, rms_threshold=0.5, ncpu=1):
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
    energy_threshold : int, default=10
        Energy threshold for conformer generation. Units in kcal/mol.
    ncpu : int, default=1
        Number of CPU cores.

    Returns
    -------
    """
    logging.debug(f" Calculate 3D-pmapper descriptors")

    for key in config["descriptor"]:
        if key == "energy_threshold":
            energy_threshold = config["descriptor"][key]
        if key == "n_conformation":
            nconfs = config["descriptor"][key]
        if key == rms_threshold:
            rms_threshold = config["descriptor"][key]
    nconfs_list = [1, nconfs]
    from miqsar.utils import calc_3d_pmapper
    out_fname = calc_3d_pmapper(input_fname=filename, nconfs_list=nconfs_list, energy=energy_threshold,  descr_num=[4],
                                path='descriptor', ncpu=ncpu)


def calc_rdkit_2d(filename):
    """ Calculate 2D descriptors using RDKit.
    """
    from miqsar.descriptor_calculation.rdkit_2d import calc_2d_descriptors
    out_path = calc_2d_descriptors(fname=filename, path='descriptor')


def calc_rdkit_morgan(filename):
    """ Calculate morgan fingerprint using RDKit.
    """
    from miqsar.descriptor_calculation.rdkit_morgan import calc_morgan_descriptors
    out_path = calc_morgan_descriptors(fname=filename, path='descriptor')


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


def _load_descriptor_file(descriptor_path):
    descriptor_file = glob.glob(descriptor_path + '/*.txt')
    descriptor_file.sort()
    descriptor_file = descriptor_file[-1]

    # Check extension
    assert os.path.splitext(descriptor_file)[-1] == '.txt'
    logging.debug(f' Descriptor file is "{descriptor_file}"')
    
    # Load files
    with open(descriptor_file) as f:
        dsc_tmp = [i.strip() for i in f.readlines()]
    with open(descriptor_file.replace('txt', 'rownames')) as f:
        mol_names = [i.strip() for i in f.readlines()]

    return dsc_tmp, mol_names


def _load_descriptor_file_rdkit(descriptor_path):
    descriptor_file = glob.glob(descriptor_path + '/*.csv')[0]
    df = pd.read_csv(descriptor_file, sep=",")
    colnames = df.columns.to_list()

    molid_header = colnames[-2]
    label_header = colnames[-3]
    dsc_header = colnames[0:-3]
    mol_names = df[molid_header].to_list()
    labels = df[label_header].to_numpy()
    dsc = df[dsc_header].to_numpy()

    assert len(mol_names) == labels.shape[0] == dsc.shape[0]
    logging.debug(f' Found {len(dsc_header)} descriptors')

    return dsc, labels, mol_names


def prepare_dataset(config):
    """
    Prepare traing and test set.

    Parameters
    ----------
    descriptor_file : str
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
    logging.debug(f" Prepare datasets")
    descriptor_path = config['descriptor']['path']

    if config["descriptor"]["method"] == "pmapper":
        #
        dsc_tmp, mol_names = _load_descriptor_file(descriptor_path)

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
        
        bags, labels, idx = np.array(bags, dtype='object'), np.array(labels), np.array(idx)
        logging.debug(f' There are {len(bags)} molecules encoded with {bags[0].shape[1]} descriptors')
    elif config["descriptor"]["method"].startswith("rdkit"):
        bags, labels, idx = _load_descriptor_file_rdkit(descriptor_path)

    # Split dataset
    from sklearn.model_selection import train_test_split
    # Note: train_test_split will return [n_confs, n_desriptors]. n_confs can differ among molecules meaning that the array is hetero array.
    random_seed = config["option"]["random_seed"]
    train_val_test_ratio = config["option"]["train_val_test_ratio"]
    test_size = train_val_test_ratio[-1]
    train_val_size = 1 - test_size
    x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(bags, labels, idx, test_size=test_size, train_size=train_val_size, random_state=random_seed)
    logging.debug(f' There are {len(x_train_val)} training/validate molecules and {len(x_test)} test molecules')

    #min_max_conf = [ _x.shape[0] for _x in bags ]
    #min_max_conf = np.array(min_max_conf)
    #logging.debug(f' Number of conformations range between {min_max_conf.min()}-{min_max_conf.max()}')

    # Save numpy
    np.savez_compressed(
        file='input.npz', 
        x_train_val=x_train_val,
        x_test=x_test,
        y_train_val=y_train_val,
        y_test=y_test,
        idx_train_val=idx_train_val,
        idx_test=idx_test
        )

    return x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test


# ----------------------------
# SUBMODULE FOR TRAINING MODEL
# ----------------------------
def _scale_data(method, x_train_val, x_test):
    """ Scale dataset.

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
    scaler.fit(np.vstack(x_train_val))
    #scaler.fit(np.vstack((x_train_val, x_test)))
    x_train_val_scaled = x_train_val.copy()
    x_test_scaled = x_test.copy()
    
    if method == "pmapper":
        for i, bag in enumerate(x_train_val):
            x_train_val_scaled[i] = scaler.transform(bag)
        for i, bag in enumerate(x_test):
            x_test_scaled[i] = scaler.transform(bag)
        x_train_val_scaled, x_test_scaled = np.array(x_train_val_scaled), np.array(x_test_scaled)
    else:
        x_train_val_scaled = scaler.transform(x_train_val)
        x_test_scaled = scaler.transform(x_test)
    
    return x_train_val_scaled, x_test_scaled


def _get_params(config):
    config_ml = config["ml_model"]
    hidden_layer_units = config_ml["hidden_layer_units"]
    pool = config_ml["pool"]
    n_epoch = config_ml["n_epoch"]
    lr = config_ml["lr"]
    weight_decay = config_ml["weight_decay"]
    batch_size = config_ml["batch_size"]
    dropout = config_ml["dropout"]

    return hidden_layer_units, pool, n_epoch, lr, weight_decay, batch_size, dropout


def _get_params_attention(config):
    config_ml = config["ml_model"]
    hidden_layer_units = config_ml["hidden_layer_units"]
    det_ndim = tuple(config_ml["attention_hidden_layer_units"])
    n_epoch = config_ml["n_epoch"]
    lr = config_ml["lr"]
    weight_decay = config_ml["weight_decay"]
    batch_size = config_ml["batch_size"]
    dropout = config_ml["dropout"]

    return hidden_layer_units, det_ndim, n_epoch, lr, weight_decay, batch_size, dropout


def train_and_predict(config, x_train_val, x_test, y_train_val, idx_train_val):
    """ Train model and predict.

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
    logging.debug(f" Train model")

    # Scaled dataset
    x_train_val_scaled, x_test_scaled = _scale_data(config["descriptor"]["method"], x_train_val, x_test)
    
    # Train
    method = config["ml_model"]["method"]
    random_seed = config["option"]["random_seed"]

    if method in ["InstanceWrapperMLPRegressor", "BagWrapperMLPRegressor", "InstanceWrapperMLPClassifier", "BagWrapperMLPClassifier"]:
        from miqsar.estimators.utils import set_seed
        set_seed(random_seed)
        hidden_layer_units, pool, n_epoch, lr, weight_decay, batch_size, dropout = _get_params(config)
        input_unit = x_train_val_scaled[0].shape[1]
        hidden_layer_units.insert(0, input_unit)
        ndim = tuple(hidden_layer_units)

        from miqsar.estimators.wrappers import InstanceWrapperMLPRegressor, BagWrapperMLPRegressor
        if method == "InstanceWrapperMLPRegressor":
            net = InstanceWrapperMLPRegressor(ndim=ndim, pool=pool, init_cuda=True)
        elif method == "BagWrapperMLPRegressor":
            net = BagWrapperMLPRegressor(ndim=ndim, pool=pool, init_cuda=True)
        elif method == "InstanceWrapperMLPClassifier":
            #net = InstanceWrapperMLPClassifier(ndim=ndim, pool=pool, init_cuda=True)
            raise NotImplementedError("InstanceWrapperMLPClassifier not supported")
        elif method == "BagWrapperMLPClassifier":
            #net = BagWrapperMLPClassifier(ndim=ndim, pool=pool, init_cuda=True)
            raise NotImplementedError("BagWrapperMLPClassifierr not supported")

        net.fit(x_train_val_scaled, y_train_val, 
            n_epoch=n_epoch, 
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            dropout=dropout)

    elif method in ["AttentionNetRegressor", "AttentionNetClassifier"]:
        from miqsar.estimators.utils import set_seed
        set_seed(random_seed)
        hidden_layer_units, det_ndim, n_epoch, lr, weight_decay, batch_size, dropout =  _get_params_attention(config)
        input_unit = x_train_val_scaled[0].shape[1]
        hidden_layer_units.insert(0, input_unit)
        ndim = tuple(hidden_layer_units)

        from miqsar.estimators.attention_nets import AttentionNetRegressor, AttentionNetClassifier
        if method == "AttentionNetRegressor":
            net = AttentionNetRegressor(ndim=ndim, det_ndim=det_ndim, init_cuda=True)
        elif method == "AttentionNetClassifier":
            #net = AttentionNetClassifier(ndim=ndim, det_ndim=det_ndim, init_cuda=True)
            raise NotImplementedError("AttentionNetClassifier not supported")

        net.fit(x_train_val_scaled, y_train_val, 
            n_epoch=n_epoch, 
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            instance_dropout=dropout)

    elif method == "RandomForestRegressor":
        from miqsar.estimators.wrappers_baseline import RandomForestRegressorWrapper
        search_method = config['ml_model']['search_method']
        params = config['ml_model']['params']
        n_cv = config['ml_model']['n_cv']
        n_iter = config['ml_model']['n_iter']
        net = RandomForestRegressorWrapper(search_method=search_method, params=params, n_cv=n_cv, n_iter=n_iter, random_seed=random_seed)
        # Perform hyperparameter tuning before fitting
        net.fit(x_train_val_scaled, y_train_val)

        #raise NotImplementedError("RandomForestRegressor not supported")
    elif method == "RandomForestClassifier":
        raise NotImplementedError("RandomForestClassifier not supported")

    # Predict
    y_test_pred = net.predict(x_test_scaled)
    y_test_pred = np.array(y_test_pred).flatten()
    y_train_val_pred = net.predict(x_train_val_scaled)
    y_train_val_pred = np.array(y_train_val_pred).flatten()

    return y_test_pred, y_train_val_pred


# -------------------------------
# SUBMODULE FOR REPORTING RESULTS
# -------------------------------
def _calc_statistics(y_exp, y_pred, suffix):
    """
    """
    from miqsar.compute_statistics import calc_errors, mean_signed_error, root_mean_squared_error, mean_unsigned_error, kendall_tau, pearson_r
    statistics = [mean_signed_error, root_mean_squared_error, mean_unsigned_error, kendall_tau, pearson_r]
    computed_statistics = dict()
    for statistic in statistics:
        name = statistic.__doc__
        computed_statistics[name] = dict()
        computed_statistics[name]['mle'] = statistic(y_pred, y_exp)
    with open(f"metric_{suffix}.txt", "w") as wf:
        for name, value in computed_statistics.items():
            wf.write(f"{name:25} {value['mle']:8.4f}\n")
    return computed_statistics


def _export_results(idx, y_exp, y_pred, suffix):
    """
    """
    with open(f"results_{suffix}.txt", "w") as wf:
        wf.write("{:8s}\t{:8s}\t{:15s}\t{:15s}\n".format("INDEX", "NAME", "EXPERIMENT", "PREDICTION"))
        for i, name in enumerate(idx):
            wf.write("{:8d}\t{:8s}\t{:15.2f}\t{:15.2f}\n".format(i, name, y_exp[i], y_pred[i]))


def report_result(ml_method, label, idx_test, y_test, y_test_pred, idx_train_val, y_train_val, y_train_val_pred):
    """ Save result.
    """
    # Export
    _export_results(idx_train_val, y_train_val, y_train_val_pred, suffix="train_val")
    _export_results(idx_test, y_test, y_test_pred, suffix="test")

    # Statistics
    if "regressor" in ml_method.lower():
        computed_statistics_train_val = _calc_statistics(y_train_val, y_train_val_pred, suffix="train_val")
        computed_statistics_test = _calc_statistics(y_test, y_test_pred, suffix="test")

    # Plot (Test dataset only)
    import matplotlib as mpl
    mpl.use('Agg')
    import seaborn
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[6,6])
    plt.scatter(y_test, y_test_pred, c='k', marker='o', s=10)
    xmin = min(y_test.min(), y_test_pred.min()) - 0.5
    xmax = min(y_test.max(), y_test_pred.max()) + 0.5
    plt.plot([xmin, xmax], [xmin, xmax], 'k-', linewidth=1)
    plt.axis([xmin, xmax, xmin, xmax])

    title = f"{label}"
    plt.title(title)

    statistics_text = f"N = {len(y_test)} compounds\n"
    for name, value in computed_statistics_test.items():
        statistics_text += f"{name}: {value['mle']:.2f}\n"
    plt.legend([statistics_text], fontsize=7)
    
    plt.xlabel('Experimental potency/affinity')
    plt.ylabel('Calculated potency/affinity')
    plt.tight_layout()
    figure_filename = ml_method.lower() + "_" + label + '.pdf'
    plt.savefig(figure_filename)
    figure_filename = ml_method.lower() + "_" + label + '.png'
    plt.savefig(figure_filename)
    print(f'Figure written to {figure_filename}')


#
# Run
#
def run(config):
    """
    """
    import yaml
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    logging.debug(f' Configuration {config}')

    # Calculate descriptors if descriptor path is not specified
    if "path" in config['descriptor'].keys():
        if not os.path.exists(config['descriptor']['path']):
            raise FileNotFoundError("Descriptor path specified but could not find directory")
    else:
        config['descriptor']['path'] = './descriptor'
        calc_descriptors(config)
    # Split data into a training and test set
    x_train_val, x_test, y_train_val, y_test, idx_train_val, idx_test = prepare_dataset(config)
    # Train and predict
    y_test_pred, y_train_val_pred = train_and_predict(config, x_train_val, x_test, y_train_val, idx_train_val)
    # Report
    ml_method = config['ml_model']['method']
    label = config['option']['dataset']
    report_result(ml_method, label, idx_test, y_test, y_test_pred, idx_train_val, y_train_val, y_train_val_pred)

@click.command()
@click.option("--yaml", required=True, help="yaml file")
def cli(**kwargs):
    run(kwargs)


if __name__ == '__main__':
    cli()