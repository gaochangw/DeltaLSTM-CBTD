import os
import random as rnd
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import model as model
from utils import pandaslogger, util
from tqdm import tqdm
import importlib
from torch.nn import CTCLoss

from utils.util import gen_paths, count_net_params

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


def load_train_modules(dataset_name):
    # Load modules according to dataset_name
    try:
        module_log = importlib.import_module('modules.' + dataset_name + '.log')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_dataloader = importlib.import_module('modules.' + dataset_name + '.dataloader')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_train_func = importlib.import_module('modules.' + dataset_name + '.train_func')
    except:
        raise RuntimeError('Please select a supported dataset.')
    try:
        module_metric = importlib.import_module('modules.' + dataset_name + '.metric')
    except:
        raise RuntimeError('Please select a supported dataset.')
    return module_log, module_dataloader, module_train_func, module_metric


def main(args):
    torch.Tensor.__repr__ = tensor_info
    ###########################################################################################################
    # Overhead
    ###########################################################################################################
    # Load Modules
    module_log, module_dataloader, module_train_func, module_metric = load_train_modules(args.dataset_name)

    # Assign methods to be used
    gen_model_id = module_log.gen_model_id
    save_best_model = module_log.save_best_model
    print_log = module_log.print_log
    CustomDataLoader = module_dataloader.CustomDataLoader
    process_network = module_train_func.process_network
    initialize_network = module_train_func.initialize_network
    Meter = module_metric.Meter
    gen_meter_args = module_metric.gen_meter_args

    # Select Loss function
    dict_loss = {'crossentropy': nn.CrossEntropyLoss(), 'ctc': CTCLoss(blank=0, reduction='sum', zero_infinity=True),
                 'mse': nn.MSELoss(), 'l1': nn.L1Loss()}
    loss_func_name = args.loss
    try:
        criterion = dict_loss[loss_func_name]
    except AttributeError:
        raise AttributeError('Please use a valid loss function. See modules/argument.py.')

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

    # Reproducibility
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
    dataloader = CustomDataLoader(trainfile=args.trainfile,
                                  valfile=args.valfile,
                                  testfile=args.testfile,
                                  args=args,
                                  qf=qf,
                                  log_feat=log_feat,
                                  normalization=args.normalization
                                  )
    n_features = dataloader.n_features
    n_classes = dataloader.n_classes
    if loss_func_name == 'ctc':
        n_classes += 1  # CTC Label Shift

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate Model
    net = model.Model(args=args, input_size=n_features, n_classes=n_classes)

    # Get parameter count
    n_param = count_net_params(net)
    print("::: Number of Parameters: ", n_param)

    ###########################################################################################################
    # Save & Log Naming Convention
    ###########################################################################################################

    # Model ID
    model_id, pretrain_model_id = gen_model_id(args, n_features=n_features)

    # Create Folders
    dir_paths, file_paths, pretrain_file = gen_paths(args,
                                                     model_id=model_id,
                                                     pretrain_model_id=pretrain_model_id)
    save_dir, log_dir_hist, log_dir_best, _ = dir_paths
    save_file, logfile_hist, logfile_best, _ = file_paths
    util.create_folder([save_dir, log_dir_hist, log_dir_best])
    print("::: Save Path: ", save_file)
    print("::: Log Path: ", logfile_hist)

    # Logger
    logger = pandaslogger.PandasLogger(logfile_hist)

    ###########################################################################################################
    # Settings
    ###########################################################################################################

    # Load Pretrained Model if Running Retrain
    if args.phase == 'retrain':
        if args.pretrain_model is None:
            print('::: Loading pretrained model: ', pretrain_file)
            net = util.load_model(net, pretrain_file, args.use_cuda, args.gpu_device)

        else:
            print('::: Loading pretrained model: ', args.pretrain_model)
            net = util.load_model(net, args.pretrain_model, args.use_cuda, args.gpu_device)

    # Use CUDA
    if args.use_cuda:
        net = net.cuda()

    # Create Optimizer
    optimizer = create_optimizer(args, net)

    # Setup Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                        mode='min',
                                                        factor=args.decay_factor,
                                                        patience=args.patience,
                                                        verbose=True,
                                                        threshold=1e-4,
                                                        min_lr=args.lr_end)

    # Initialize Network Parameters
    if args.phase == 'pretrain':
        initialize_network(net, args)

    # Create Meters
    dict_meter_args = gen_meter_args(args, n_classes)
    meter = Meter(dict_meter_args)

    ###########################################################################################################
    # Training
    ###########################################################################################################
    # Value for Saving Best Model
    best_model = None
    # Timer
    start_time = time.time()

    # Epoch loop
    print("Starting training...")
    for epoch in range(args.n_epochs):
        # Update Alpha
        alpha = 1 if args.phase == 'retrain' else min(epoch / (args.alpha_anneal_epoch - 1), 1.0)

        # -----------
        # Train
        # -----------
        net, _, train_stat = net_train(args,
                                       net=net,
                                       batch_size=args.batch_size,
                                       meter=None,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       dataloader=dataloader,
                                       epoch=epoch,
                                       shuffle_type='random',
                                       enable_gauss=0)

        # Process Network after training per epoch
        net, net_for_eval = process_network(args, net=net, alpha=alpha)

        # -----------
        # Validation
        # -----------
        val_stat = None
        if args.eval_val:
            _, meter, val_stat = net_eval(args,
                                          net=net_for_eval,
                                          set_name='val',
                                          batch_size=args.batch_size_eval,
                                          meter=meter,
                                          criterion=criterion,
                                          iterator=dataloader,
                                          epoch=epoch,
                                          shuffle_type='high_throughput',
                                          enable_gauss=0)
            if args.score_val:
                val_stat = meter.get_metrics(val_stat)
            meter.clear_data()

        # -----------
        # Test
        # -----------
        test_stat = None
        if args.eval_test:
            _, meter, test_stat = net_eval(args,
                                           net=net_for_eval,
                                           set_name='test',
                                           batch_size=args.batch_size_eval,
                                           meter=meter,
                                           criterion=criterion,
                                           iterator=dataloader,
                                           epoch=epoch,
                                           shuffle_type='high_throughput',
                                           enable_gauss=0)
            if args.score_test:
                test_stat = meter.get_metrics(test_stat)
            meter.clear_data()
            # print("Max: %3.4f | Min: %3.4f" % (test_stat['net_out_max'], test_stat['net_out_min']))

        ###########################################################################################################
        # Logging & Saving
        ###########################################################################################################
        # Generate Log Dict
        log_stat = gen_log_stat(args, epoch, start_time, net, loss_func_name, alpha,
                                optimizer, train_stat, val_stat, test_stat)

        # Write Log
        logger.write_log(log_stat)

        # Print
        print_log(args, log_stat, train_stat, val_stat, test_stat)

        # Save best model
        best_model = save_best_model(args=args,
                                     best_val=best_model,
                                     net=net_for_eval,
                                     save_file=save_file,
                                     logger=logger,
                                     logfile_best=logfile_best,
                                     epoch=epoch,
                                     val_stat=val_stat,
                                     score_val=args.score_val)

        ###########################################################################################################
        # Learning Rate Schedule
        ###########################################################################################################
        # Schedule at the beginning of retrain
        if args.lr_schedule:
            if args.phase == 'retrain':
                lr_scheduler.step(val_stat['lr_criterion'])
            # Schedule after the alpha annealing is over
            elif args.cbtd:
                if epoch >= args.alpha_anneal_epoch:
                    lr_scheduler.step(val_stat['lr_criterion'])
            else:
                lr_scheduler.step(val_stat['lr_criterion'])

    print("Training Completed...                                               ")
    print(" ")


def gen_log_stat(args, epoch, start_time, net, loss_func_name, alpha,
                 optimizer=None, train_stat=None, val_stat=None, test_stat=None):
    # End Timer
    time_curr = util.timeSince(start_time)

    # Get Epoch & Batch Size
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Get current learning rate
    lr_curr = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            lr_curr = param_group['lr']

    # Get parameter count
    n_param = 0
    for name, param in net.named_parameters():
        sizes = 1
        for el in param.size():
            sizes = sizes * el
        n_param += sizes

    # Create log dictionary
    retrain = 1 if args.phase == 'retrain' else 0
    log_stat = {'EPOCH': epoch,
                'N_EPOCH': n_epochs,
                'TIME_CURR': time_curr,
                'BATCH_SIZE': batch_size,
                'N_PARAM': n_param,
                'LOSS_FUNC': loss_func_name,
                'RETRAIN': retrain,
                'OPT': args.opt,
                'LR_CURR': lr_curr,
                'CLIP_GRAD_NORM_MAX': args.clip_grad_norm_max,
                'ALPHA': alpha,
                'LAMBDA': args.beta,
                'WEIGHT_DECAY': args.weight_decay,
                'HID_DROPOUT': args.hid_dropout,
                'NUM_ARRAY': args.num_array,
                'NUM_ARRAY_PE': args.num_array_pe
                }

    # Merge stat dicts into the log dict
    if train_stat is not None:
        train_stat_log = {f'TRAIN_{k.upper()}': v for k, v in train_stat.items()}
        log_stat = {**log_stat, **train_stat_log}
    if val_stat is not None:
        val_stat_log = {f'VAL_{k.upper()}': v for k, v in val_stat.items()}
        del val_stat_log['VAL_LR_CRITERION']
        log_stat = {**log_stat, **val_stat_log}
    if test_stat is not None:
        test_stat_log = {f'TEST_{k.upper()}': v for k, v in test_stat.items()}
        del test_stat_log['TEST_LR_CRITERION']
        log_stat = {**log_stat, **test_stat_log}

    # Evaluate RNN Weight Sparsity
    n_nonzero_weight_elem = 0
    n_weight_elem = 0
    for name, param in net.named_parameters():
        if 'rnn' in name:
            if 'weight' in name:
                n_nonzero_weight_elem += len(torch.nonzero(param.data))
                n_weight_elem += param.data.nelement()
    log_stat['SP_W_RNN'] = 1 - (n_nonzero_weight_elem / n_weight_elem)

    # Evaluate FC Extra Layer Weight Sparsity
    log_stat['SP_W_FC'] = 0
    if args.fc_extra_size:
        n_nonzero_weight_elem = 0
        n_weight_elem = 0
        for name, param in net.named_parameters():
            if 'fc_extra' in name:
                if 'weight' in name:
                    n_nonzero_weight_elem += len(torch.nonzero(param.data))
                    n_weight_elem += param.data.nelement()
        log_stat['SP_W_FC'] = 1 - (n_nonzero_weight_elem / n_weight_elem)
    return log_stat


def create_optimizer(args, net):
    if args.opt == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=False, weight_decay=args.weight_decay)
    elif args.opt == 'ADAMW':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=False, weight_decay=args.weight_decay)
    elif args.opt == 'qADAM':
        optimizer = qADAM(net.parameters(), amsgrad=True, lr=args.lr,
                          qw=args.qw, qg=args.qg, wqi=args.wqi, wqf=args.wqf, gqi=args.gqi, gqf=args.gqf)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.opt == 'RMSPROP':
        optimizer = optim.RMSprop(net.parameters(), lr=0.0016, alpha=0.95, eps=1e-08, weight_decay=0, momentum=0,
                                  centered=False)
    else:
        raise RuntimeError('Please use a valid optimizer.')
    return optimizer


def net_eval(args,
             net,
             set_name,
             batch_size,
             meter,
             criterion,
             iterator,
             epoch,
             shuffle_type,
             enable_gauss):
    # Load modules according to dataset_name
    try:
        module_train_func = importlib.import_module('modules.' + args.dataset_name + '.train_func')
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    add_meter_data = module_train_func.add_meter_data
    forward_propagation = module_train_func.forward_propagation
    try:
        get_net_out_stat = module_train_func.get_net_out_stat
    except (NameError, AttributeError):
        get_net_out_stat = None
        pass

    with torch.no_grad():
        # Set Network Properties
        net = net.eval()
        net.set_qa_fc_final(args.qa_fc_final)

        # Statistics
        epoch_loss = 0.
        epoch_regularizer = 0.
        n_batches = 0

        # Meter data buffer
        dict_meter_data = {'net_out': [], 'net_qout': []}

        # Batch Iteration
        for dict_batch_array in tqdm(iterator.iterate(epoch=epoch,
                                                      set_name=set_name,
                                                      batch_size=batch_size,
                                                      mode='batch',
                                                      shuffle_type=shuffle_type,
                                                      enable_gauss=enable_gauss),
                                     desc='Eval',
                                     unit='batches',
                                     total=iterator.get_num_batch(set_name, batch_size)):

            # Get Batch Data
            dict_batch_tensor = get_batch_data(args, dict_batch_array)

            # Forward Propagation
            net_out, _, reg = forward_propagation(net, dict_batch_tensor)

            # Calculate Loss
            loss, loss_reg = calculate_loss(loss_fn=criterion,
                                            net_out=net_out,
                                            dict_targets=dict_batch_tensor,
                                            reg=reg,
                                            beta=args.beta)

            # Increment monitoring variables
            batch_loss = loss.item()
            epoch_loss += batch_loss  # Accumulate loss
            if reg:
                epoch_regularizer += loss_reg.detach().item()
            n_batches += 1  # Accumulate count so we can calculate mean later

            # Collect Meter Data
            dict_meter_data['net_out'].append(net_out.detach().cpu())
            for k, v in dict_batch_tensor.items():
                if k == 'features':
                    continue
                try:
                    dict_meter_data[k].append(v.detach().cpu())
                except KeyError:
                    dict_meter_data[k] = []
                    dict_meter_data[k].append(v.detach().cpu())

            # Garbage collection to free VRAM
            del dict_batch_tensor, dict_batch_array, loss, net_out

        # Average loss and regularizer values across all batches
        epoch_loss = epoch_loss / float(n_batches)
        epoch_regularizer = epoch_regularizer / float(n_batches)

        # Add meter data
        if meter is not None:
            meter = add_meter_data(args, meter, dict_meter_data)

        #######################
        # Save Statistics
        #######################
        # Add basic stats
        stat = {'loss': epoch_loss, 'reg': epoch_regularizer, 'lr_criterion': epoch_loss}
        # Get DeltaRNN Stats
        if "Delta" in args.hid_type:
            # Evaluate temporal sparsity
            dict_stats = net.rnn.get_temporal_sparsity()
            stat['sp_dx'] = dict_stats['sparsity_delta_x']
            stat['sp_dh'] = dict_stats['sparsity_delta_h']

            # Evaluate workload
            dict_stats = net.rnn.get_workload()
            stat['balance'] = dict_stats['balance']

        # Evaluate network output
        if get_net_out_stat is not None:
            stat = get_net_out_stat(args, stat, dict_meter_data)
        return net, meter, stat


def net_train(args,
              net,
              batch_size,
              meter,
              optimizer,
              criterion,
              dataloader,
              epoch,
              shuffle_type,
              enable_gauss):
    # Load modules according to dataset_name
    try:
        module_train_func = importlib.import_module('modules.' + args.dataset_name + '.train_func')
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Assign methods to be used
    get_batch_data = module_train_func.get_batch_data
    calculate_loss = module_train_func.calculate_loss
    add_meter_data = module_train_func.add_meter_data
    forward_propagation = module_train_func.forward_propagation
    try:
        get_net_out_stat = module_train_func.get_net_out_stat
    except (NameError, AttributeError):
        get_net_out_stat = None
        pass

    # Set Network Properties
    net = net.train()
    net.set_qa_fc_final(0)

    # Stat
    epoch_loss = 0
    epoch_regularizer = 0
    n_batches = 0

    # Meter data buffer
    dict_meter_data = {'net_out': [], 'net_qout': []}

    # Iterate through batches
    batch_iterator = dataloader.iterate(epoch=epoch,
                                        set_name='train',
                                        batch_size=batch_size,
                                        mode='batch',
                                        shuffle_type=shuffle_type,
                                        enable_gauss=enable_gauss)
    for dict_batch_array in tqdm(batch_iterator,
                                 desc='Train',
                                 unit='batches',
                                 total=dataloader.get_num_batch('train', batch_size)):

        # Get Batch Data
        dict_batch_tensor = get_batch_data(args, dict_batch_array)

        # Optimization
        optimizer.zero_grad()

        # Forward Propagation
        net_out, _, reg = forward_propagation(net, dict_batch_tensor)

        # Calculate Loss
        loss, loss_reg = calculate_loss(loss_fn=criterion,
                                        net_out=net_out,
                                        dict_targets=dict_batch_tensor,
                                        reg=reg,
                                        beta=args.beta)

        # Get Network Outputs Statistics
        if n_batches == 0:
            net_out_min = torch.min(net_out).item()
            net_out_max = torch.max(net_out).item()
        else:
            min_cand = torch.min(net_out)
            max_cand = torch.max(net_out)
            if min_cand < net_out_min:
                net_out_min = min_cand.item()
            if max_cand > net_out_max:
                net_out_max = max_cand.item()

        # Backward propagation
        loss.backward()

        # Gradient clipping
        if args.clip_grad_norm_max != 0:
            nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm_max)

        # Update parameters
        optimizer.step()

        # Increment monitoring variables
        loss.detach()
        batch_loss = loss.item()
        epoch_loss += batch_loss  # Accumulate loss
        if reg is not None:
            epoch_regularizer += loss_reg.detach().item()
        n_batches += 1  # Accumulate count so we can calculate mean later

        # Collect Meter Data
        dict_meter_data['net_out'].append(net_out.detach().cpu())
        for k, v in dict_batch_tensor.items():
            if k == 'features':
                continue
            try:
                dict_meter_data[k].append(v.detach().cpu())
            except KeyError:
                dict_meter_data[k] = []

        # Garbage collection to free VRAM
        del dict_batch_tensor, dict_batch_array, loss, reg, net_out

    # Average loss and regularizer values across batches
    epoch_loss /= n_batches
    epoch_loss = epoch_loss
    epoch_regularizer /= n_batches

    # Collect outputs and targets
    if meter is not None:
        meter = add_meter_data(meter, dict_meter_data)

    # Get network statistics
    stat = {'loss': epoch_loss, 'reg': epoch_regularizer, 'net_out_min': net_out_min, 'net_out_max': net_out_max}
    if get_net_out_stat is not None:
        stat = get_net_out_stat(args, stat, dict_meter_data)
    return net, meter, stat
