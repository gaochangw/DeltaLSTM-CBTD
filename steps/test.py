import random as rnd
import time
import os
import numpy as np
import torch.nn as nn
import torch.utils.data.dataloader
from steps.train import net_eval, gen_log_stat
import model as model
from utils import pandaslogger, util
import importlib
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from utils.util import gen_paths

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


def main(args):
    torch.Tensor.__repr__ = tensor_info
    ###########################################################################################################
    # Overhead
    ###########################################################################################################
    # Load modules according to dataset_name
    try:
        module_log = importlib.import_module('modules.' + args.dataset_name + '.log')
        module_dataloader = importlib.import_module('modules.' + args.dataset_name + '.dataloader')
        module_metric = importlib.import_module('modules.' + args.dataset_name + '.metric')
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    gen_model_id = module_log.gen_model_id
    print_log = module_log.print_log
    CustomDataLoader = module_dataloader.CustomDataLoader
    Meter = module_metric.Meter
    gen_meter_args = module_metric.gen_meter_args

    # Select Loss function
    dict_loss = {'crossentropy': nn.CrossEntropyLoss(), 'ctc': CTCLoss(), 'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
    loss_func_name = args.loss
    try:
        criterion = dict_loss[loss_func_name]
    except:
        raise RuntimeError('Please use a valid loss function. See modules/argument.py.')

    print("::: Phase:   ", args.phase)
    print("::: Loading: ", args.trainfile)
    print("::: Loading: ", args.valfile)
    print("::: Loading: ", args.testfile)

    # Find Available GPUs
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_device)
        idx_gpu = torch.cuda.current_device()
        name_gpu = torch.cuda.get_device_name(idx_gpu)
        print("::: Available GPUs: %s" % (torch.cuda.device_count()))
        print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
        print("--------------------------------------------------------------------")
    else:
        print("::: Available GPUs: None")
        print("--------------------------------------------------------------------")

    # Set seeds
    rnd.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # Create Dataset Iterators
    try:
        qf = args.qf
        log_feat = args.log_feat
    except AttributeError:
        qf = None
        log_feat = None
        pass
    iterator = CustomDataLoader(trainfile=args.trainfile,
                                valfile=args.valfile,
                                testfile=args.testfile,
                                args=args,
                                qf=qf,
                                log_feat=log_feat,
                                normalization=args.normalization
                                )

    n_features = iterator.n_features
    n_classes = iterator.n_classes
    if loss_func_name == 'ctc':
        n_classes += 1  # CTC Label Shift

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model
    net = model.Model(args=args, input_size=n_features, n_classes=n_classes)

    ###########################################################################################################
    # Save & Log Naming Convention
    ###########################################################################################################
    # Model ID
    model_id, pretrain_model_id = gen_model_id(args,
                                               n_features=n_features)

    # Create Folders
    dir_paths, file_paths, pretrain_file = gen_paths(args,
                                                     model_id=model_id,
                                                     pretrain_model_id=pretrain_model_id)

    save_dir, _, _, log_dir_test = dir_paths
    save_file, _, _, logfile_test = file_paths

    # Logger
    logger = pandaslogger.PandasLogger(logfile_test)

    # Create Folders
    util.create_folder([log_dir_test])

    ###########################################################################################################
    # Settings
    ###########################################################################################################
    # Load Test Network
    test_model = os.path.join(save_dir, model_id + '.pt')
    if args.pretrain_model is None:
        net = util.load_model(net, test_model, args.use_cuda, args.gpu_device)
        print('::: Loading test model: ', test_model)
    else:
        net = util.load_model(net, args.pretrain_model, args.use_cuda, args.gpu_device)
        print('::: Loading test model: ', args.pretrain_model)

    # Use CUDA
    if args.use_cuda:
        net = net.cuda()

    # Create Meters
    dict_meter_args = gen_meter_args(args, n_classes)
    meter = Meter(dict_meter_args)

    ###########################################################################################################
    # Testing
    ###########################################################################################################

    # Timer
    start_time = time.time()

    # -----------
    # Validation
    # -----------
    val_stat = None
    if args.dataset_name != 'sensorsgas':
        _, meter, val_stat = net_eval(args,
                                      net=net,
                                      set_name='val',
                                      batch_size=args.batch_size_eval,
                                      meter=meter,
                                      criterion=criterion,
                                      iterator=iterator,
                                      epoch=0,
                                      shuffle_type='high_throughput',
                                      enable_gauss=0)
        val_stat = meter.get_metrics(val_stat)

    # -----------
    # Test
    # -----------
    _, meter, test_stat = net_eval(args,
                                   net=net,
                                   set_name='test',
                                   batch_size=args.batch_size_eval,
                                   meter=meter,
                                   criterion=criterion,
                                   iterator=iterator,
                                   epoch=0,
                                   shuffle_type='high_throughput',
                                   enable_gauss=0)
    test_stat = meter.get_metrics(test_stat)

    ###########################################################################################################
    # Logging & Saving
    ###########################################################################################################
    # Generate Log Dict
    log_stat = gen_log_stat(args, epoch=0, start_time=start_time, net=net,
                            loss_func_name=loss_func_name, alpha=0, optimizer=None,
                            train_stat=None, val_stat=val_stat, test_stat=test_stat)

    # Write Log
    logger.write_log(log_stat)

    # Print
    print_log(args, log_stat, None, val_stat, test_stat)

    print("Testing Completed...                                               ")
    print(" ")
