import os
import numpy as np
import utils.util as util
# from modules import data_collector
import importlib


def main(args):
    np.random.seed(0)
    dataset_path, testset_path = util.get_dataset_path(args)

    # Dictionary for outputs
    output_path = os.path.join('data', args.dataset_name)

    # Load modules according to dataset_name
    try:
        module_data = importlib.import_module('modules.' + args.dataset_name + '.data_collector')
        DataCollector = module_data.DataPrepare
    except:
        raise RuntimeError('Please select a supported dataset.')

    print("Preparing: ", args.dataset_name)
    prepare = DataCollector(dataset_path=dataset_path, output_path=output_path)

    # Collect Dataset
    prepare.collect()


def gen_meter_args(args, dataset_path, output_path):
    dict_meter_args = {'dataset_path': dataset_path,
                       'output_path': output_path,
                       'n_targets': args.n_targets}
    if args.dataset_name == 'timit':
        dict_meter_args['phn'] = args.phn
    return dict_meter_args
