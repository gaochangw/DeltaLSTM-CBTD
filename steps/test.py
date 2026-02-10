import random as rnd
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader
from steps.train import net_eval, gen_log_stat
import model as model
from utils import pandaslogger, util
import importlib
from torch.nn import CTCLoss
from utils.util import gen_paths

# Import Excel logger and model stats
try:
    from utils.excel_logger import ExcelLogger
    EXCEL_LOGGER_AVAILABLE = True
except ImportError:
    EXCEL_LOGGER_AVAILABLE = False

try:
    from utils.json_logger import JsonLogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

try:
    from utils.model_stats import get_model_summary_dict
    MODEL_STATS_AVAILABLE = True
except ImportError:
    MODEL_STATS_AVAILABLE = False

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
    dict_loss = {'crossentropy': nn.CrossEntropyLoss(), 'ce': nn.CrossEntropyLoss(),
                 'ctc': CTCLoss(), 'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
    # Add weighted MSE / RMSE for 3ET+ regression
    if args.loss in ('weighted_mse', 'weighted_rmse'):
        spatial_factor = getattr(args, 'spatial_factor', 0.125)
        sensor_w = getattr(args, 'sensor_width', 640) * spatial_factor
        sensor_h = getattr(args, 'sensor_height', 480) * spatial_factor
        loss_weights = torch.tensor([sensor_w / sensor_h, 1.0])
        from utils.weighted_mse import weighted_MSELoss
        dict_loss['weighted_mse'] = weighted_MSELoss(weights=loss_weights)
        if args.loss == 'weighted_rmse':
            from utils.weighted_mse import weighted_RMSELoss
            dict_loss['weighted_rmse'] = weighted_RMSELoss(weights=loss_weights)
    loss_func_name = args.loss
    try:
        criterion = dict_loss[loss_func_name]
    except (KeyError, RuntimeError):
        raise RuntimeError('Please use a valid loss function. See modules/argument.py.')

    print("::: Phase:   ", args.phase)
    print("::: Loading: ", args.trainfile)
    print("::: Loading: ", args.valfile)
    print("::: Loading: ", args.testfile)

    # Find Available Accelerators
    if torch.cuda.is_available() and args.use_cuda:
        torch.cuda.set_device(args.gpu_device)
        idx_gpu = torch.cuda.current_device()
        name_gpu = torch.cuda.get_device_name(idx_gpu)
        print("::: Available GPUs: %s" % (torch.cuda.device_count()))
        print("::: Using GPU %s:   %s" % (idx_gpu, name_gpu))
        args.device = torch.device('cuda')
        print("--------------------------------------------------------------------")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("::: Using Apple MPS (Metal Performance Shaders)")
        args.device = torch.device('mps')
        print("--------------------------------------------------------------------")
    else:
        print("::: Available GPUs: None, using CPU")
        args.device = torch.device('cpu')
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

    # Print model configuration and analysis
    try:
        print_training_header = module_log.print_training_header
        print_model_analysis = module_log.print_model_analysis
        n_param = sum(p.numel() for p in net.parameters())
        print_training_header(args, n_features, n_classes, n_param)
        seq_len = 30 if args.dataset_name == 'threeet' else (98 if args.dataset_name == 'gscdv2' else 777)
        print_model_analysis(net, args, n_features, seq_len=seq_len, device='cpu')
    except (AttributeError, TypeError):
        pass

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

    # Move model to device (CUDA, MPS, or CPU)
    net = net.to(args.device)

    # Move criterion to device (needed for weighted_MSELoss which has buffers)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(args.device)

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

    # Create and save Excel log
    if EXCEL_LOGGER_AVAILABLE:
        try:
            excel_logger = ExcelLogger(
                log_dir=log_dir_test,
                model_id=model_id,
                dataset_name=args.dataset_name,
                phase=f"{args.phase}_test"
            )
            excel_logger.set_model_config(args, n_features, n_classes)

            # Add model stats
            if MODEL_STATS_AVAILABLE:
                seq_len = 30 if args.dataset_name == 'threeet' else (98 if args.dataset_name == 'gscdv2' else 777)
                model_stats = get_model_summary_dict(net, seq_len, n_features)
                excel_logger.set_model_stats(model_stats)

            # Add test results
            excel_logger.add_epoch(log_stat)

            # Save
            excel_path = excel_logger.save()
            print(f"::: Excel log saved to: {excel_path}")
        except Exception as e:
            print(f"::: Failed to save Excel log: {e}")

    # Create and save JSON log
    if JSON_LOGGER_AVAILABLE:
        try:
            json_logger = JsonLogger(
                log_dir=log_dir_test,
                model_id=model_id,
                dataset_name=args.dataset_name,
                phase=f"{args.phase}_test"
            )
            json_logger.set_config(args, n_features, n_classes)

            # Add model stats
            if MODEL_STATS_AVAILABLE:
                seq_len = 30 if args.dataset_name == 'threeet' else (98 if args.dataset_name == 'gscdv2' else 777)
                model_stats = get_model_summary_dict(net, seq_len, n_features)
                json_logger.set_complexity(model_stats)

            # Add test results
            json_logger.add_epoch(log_stat)

            # Save
            json_path = json_logger.save()
            print(f"::: JSON log saved to: {json_path}")
        except Exception as e:
            print(f"::: Failed to save JSON log: {e}")

    # Print
    print_log(args, log_stat, None, val_stat, test_stat)

    # Print sparsity analysis and delta stats
    try:
        print_sparsity_analysis = module_log.print_sparsity_analysis
        print_sparsity_analysis(net, args)
    except (AttributeError, TypeError):
        pass

    if 'Delta' in args.hid_type:
        try:
            print_delta_stats = module_log.print_delta_stats
            print_delta_stats(net, test_stat)
        except (AttributeError, TypeError):
            pass

    # Bit-level sparsity & speedup analysis
    if 'Delta' in args.hid_type:
        try:
            from utils.sparsity_analysis import analyze_sparsity_scenarios, print_sparsity_speedup_report
            representation = 'sign_magnitude' if getattr(args, 'bitwave', 0) else 'twos_complement'
            analysis = analyze_sparsity_scenarios(net, args, temporal_stats=test_stat, representation=representation)
            print_sparsity_speedup_report(analysis)
        except Exception as e:
            print(f"::: Sparsity analysis failed: {e}")

    # Generate 3ET+ challenge submission CSV
    if args.dataset_name == 'threeet':
        try:
            _generate_threeet_submission(net, iterator, args, model_id, log_dir_test)
        except Exception as e:
            print(f"::: 3ET+ submission generation failed: {e}")

    print("Testing Completed...                                               ")
    print(" ")


def _generate_threeet_submission(net, iterator, args, model_id, output_dir):
    """Generate 3ET+ challenge submission CSV with test predictions."""
    import csv

    spatial_factor = getattr(args, 'spatial_factor', 0.125)
    sensor_width = getattr(args, 'sensor_width', 640)
    sensor_height = getattr(args, 'sensor_height', 480)
    ds_width = sensor_width * spatial_factor
    ds_height = sensor_height * spatial_factor

    module_train_func = importlib.import_module('modules.threeet.train_func')
    get_batch_data = module_train_func.get_batch_data

    net.eval()
    net.set_qa_fc_final(args.qa_fc_final)

    all_preds = []
    with torch.no_grad():
        for dict_batch_array in iterator.iterate(epoch=0, set_name='test',
                                                  batch_size=args.batch_size_eval,
                                                  mode='batch',
                                                  shuffle_type='high_throughput'):
            dict_batch_tensor = get_batch_data(args, dict_batch_array)
            outputs, _, _ = net(dict_batch_tensor['features'], None,
                                dict_batch_tensor['feature_lengths'])
            out_final = outputs[0]  # (B, T, 2)
            all_preds.append(out_final.cpu())

    all_preds = torch.cat(all_preds, dim=0)  # (N, T, 2)
    # Flatten to (N*T, 2)
    preds_flat = all_preds.reshape(-1, 2).numpy()

    # Rescale from [0,1] to pixel space
    preds_flat[:, 0] *= ds_width
    preds_flat[:, 1] *= ds_height

    # Write submission CSV
    submission_path = os.path.join(output_dir, f'submission_{model_id}.csv')
    os.makedirs(output_dir, exist_ok=True)
    with open(submission_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'x', 'y'])
        for i, (x, y) in enumerate(preds_flat):
            writer.writerow([i, f'{x:.4f}', f'{y:.4f}'])

    print(f"::: 3ET+ submission saved to: {submission_path}")
    print(f"::: Total predictions: {len(preds_flat)}")
