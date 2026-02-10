###########################################################################################################
# 3ET+ Event-Based Eye Tracking - Pupil Center Regression
###########################################################################################################


def add_args(parser):
    # Basic Settings
    parser.add_argument('--filename', default='', help='Filename to save model and log to.')
    parser.add_argument('--phase', default='pretrain', help='Phase of a step')
    # Dataset
    parser.add_argument('--data_dir', default=None, help='Directory path that saves datasets')
    parser.add_argument('--trainfile', default=None, help='HDF5 File of training set')
    parser.add_argument('--valfile', default=None, help='HDF5 File of validation set')
    parser.add_argument('--testfile', default=None, help='HDF5 File of testing set')
    parser.add_argument('--zero_padding', default='head',
                        help='Method of padding zeros to samples in a batch')
    # 3ET+ Specific Settings
    parser.add_argument('--cnn_frontend', default=1, type=int,
                        help='Enable CNN spatial feature extractor for voxel grids')
    parser.add_argument('--spatial_factor', default=0.3333333333333333, type=float,
                        help='Spatial downsampling factor (240x180 -> 80x60)')
    parser.add_argument('--n_time_bins', default=3, type=int,
                        help='Number of temporal bins in voxel grid')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of frames per sequence')
    parser.add_argument('--train_stride', default=15, type=int,
                        help='Stride for training sequence slicing')
    parser.add_argument('--val_stride', default=30, type=int,
                        help='Stride for val/test sequence slicing')
    parser.add_argument('--sensor_width', default=240, type=int, help='Event camera sensor width')
    parser.add_argument('--sensor_height', default=180, type=int, help='Event camera sensor height')
    parser.add_argument('--pixel_tolerances', default='5,10,15', type=str,
                        help='Comma-separated pixel tolerances for p-accuracy')
    parser.add_argument('--temp_subsample_factor', default=1.0, type=float,
                        help='Temporal subsample factor for labels')
    # CNN input dimensions (derived from spatial_factor: 240*1/3=80, 180*1/3=60)
    parser.add_argument('--cnn_input_height', default=60, type=int,
                        help='CNN input height (sensor_height * spatial_factor)')
    parser.add_argument('--cnn_input_width', default=80, type=int,
                        help='CNN input width (sensor_width * spatial_factor)')
    parser.add_argument('--cnn_pool_height', default=5, type=int,
                        help='CNN adaptive pool output height (default 5 for 80x60 input)')
    parser.add_argument('--cnn_pool_width', default=4, type=int,
                        help='CNN adaptive pool output width (default 4 for 80x60 input)')
    # Hyperparameters
    parser.add_argument('--seed', default=4, type=int,
                        help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=64, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of epochs to train for.')
    parser.add_argument('--n_epochs_retrain', default=5, type=int, help='Number of epochs for retrain phase.')
    parser.add_argument('--loss', default='weighted_mse', help='Loss function.')
    parser.add_argument('--opt', default='ADAMW', help='Which optimizer to use (ADAM or SGD)')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-5, type=float, help='End learning rate')
    parser.add_argument('--decay_factor', default=0.5, type=float, help='Learning rate decay factor')
    parser.add_argument('--patience', default=5, type=float, help='Patience for LR scheduler')
    parser.add_argument('--beta', default=0, type=float, help='Regularization coefficient')
    parser.add_argument('--clip_grad_norm_max', default=100, type=float, help='Gradient clipping max norm')
    # General Network Settings
    parser.add_argument('--hid_type', default='LSTM', help='RNN layer type')
    parser.add_argument('--hid_type_pretrain', default='LSTM',
                        help='Pretrain model type to be load')
    parser.add_argument('--hid_layers', default=1, type=int, help='Number of RNN layers')
    parser.add_argument('--hid_size', default=256, type=int, help='Size of RNN layers')
    parser.add_argument('--hid_dropout', default=0, type=float,
                        help='Dropout rate of the RNN layer')
    parser.add_argument('--fc_type', default='FC', help='FC layer type')
    parser.add_argument('--fc_extra_size', default=0, type=int,
                        help='Whether add FC layer after the RNN layer')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Factor of weight decay')
    # Delta Network Settings
    parser.add_argument('--hardsigmoid', default=0, type=int, help='Use hardsigmoid')
    parser.add_argument('--hardtanh', default=0, type=int, help='Use hardtanh')
    parser.add_argument('--qa', default=0, type=int, help='Quantize activations')
    parser.add_argument('--aqi', default=8, type=int, help='Activation integer bits')
    parser.add_argument('--aqf', default=8, type=int, help='Activation fractional bits')
    parser.add_argument('--qw', default=0, type=int, help='Quantize weights')
    parser.add_argument('--wqi', default=1, type=int, help='Weight integer bits')
    parser.add_argument('--wqf', default=7, type=int, help='Weight fractional bits')
    parser.add_argument('--nqi', default=2, type=int, help='LUT output integer bits')
    parser.add_argument('--nqf', default=4, type=int, help='LUT output fractional bits')
    parser.add_argument('--qa_fc_extra', default=1, type=int, help='Quantize extra FC activations')
    parser.add_argument('--aqi_fc_extra', default=8, type=int, help='Extra FC activation integer bits')
    parser.add_argument('--aqf_fc_extra', default=8, type=int, help='Extra FC activation fractional bits')
    parser.add_argument('--qa_fc_final', default=1, type=int, help='Quantize final FC activations')
    parser.add_argument('--aqi_fc_final', default=8, type=int, help='Final FC activation integer bits')
    parser.add_argument('--aqf_fc_final', default=8, type=int, help='Final FC activation fractional bits')
    parser.add_argument('--th_x', default=0.1, type=float, help='Delta threshold for input')
    parser.add_argument('--th_h', default=0.1, type=float, help='Delta threshold for hidden')
    # Gradient quantization (for compatibility)
    parser.add_argument('--qg', default=0, type=int, help='Quantize gradients')
    parser.add_argument('--gqi', default=8, type=int, help='Gradient integer bits')
    parser.add_argument('--gqf', default=8, type=int, help='Gradient fractional bits')
    # DeltaLSTM
    parser.add_argument('--cbtd', default=0, type=int, help='Use Column-Balanced Weight Dropout')
    parser.add_argument('--gamma_rnn', default=0.0, type=float, help='Target sparsity of RNN layer')
    parser.add_argument('--gamma_fc', default=0.0, type=float, help='Target sparsity of FC Layer')
    parser.add_argument('--alpha_anneal_epoch', default=30, type=int, help='Epochs for alpha to reach 1')
    parser.add_argument('--balance_pe', default=1, type=float, help='Balance workload of PEs in CBTD')
    parser.add_argument('--num_array', default=1, type=int, help='Number of MAC Arrays (M)')
    parser.add_argument('--num_array_pe', default=4, type=int, help='Number of PEs per MAC Array (N)')
    # MSQ Bit Sparsification
    parser.add_argument('--msq_lambda', default=0.0, type=float,
                        help='MSQ LSB regularization weight (0=disabled)')
    parser.add_argument('--msq_k', default=2, type=int,
                        help='Number of LSBs to penalize (default 2)')
    parser.add_argument('--msq_target', default='rnn', type=str,
                        help='Layers for MSQ: rnn, fc, or all')
    parser.add_argument('--msq_anneal', default=0, type=int,
                        help='Anneal MSQ lambda over alpha_anneal_epoch (0=constant, 1=linear ramp)')
    # BitWave Sign-Magnitude
    parser.add_argument('--bitwave', default=0, type=int,
                        help='Use sign-magnitude for bit sparsity analysis (0=two\'s complement, 1=sign-magnitude)')
    # RVT Configuration (only used when --hid_type RVT)
    parser.add_argument('--rvt_channels', default='32,64', type=str,
                        help='RVT output channels per stage (comma-separated)')
    parser.add_argument('--rvt_dim_head', default=32, type=int,
                        help='RVT attention head dimension')
    parser.add_argument('--rvt_drop_prob', default=0.6, type=float,
                        help='RVT MLP dropout probability')
    parser.add_argument('--rvt_expansion_ratio', default=2, type=int,
                        help='RVT MLP expansion ratio')
    # JaneEyeNet Configuration (only used when --hid_type JaneEyeNet)
    parser.add_argument('--jen_kernel_size', default=3, type=int,
                        help='JaneEyeNet conv kernel size')
    parser.add_argument('--jen_output_channels', default=32, type=int,
                        help='JaneEyeNet output channels')
    parser.add_argument('--jen_stride', default=2, type=int,
                        help='JaneEyeNet conv stride')
    parser.add_argument('--jen_drop_prob', default=0.6, type=float,
                        help='JaneEyeNet MLP dropout probability')
    parser.add_argument('--jen_expansion_ratio', default=2, type=int,
                        help='JaneEyeNet MLP expansion ratio')
    # TBPTT (Truncated Backpropagation Through Time)
    parser.add_argument('--tbptt', default=0, type=int,
                        help='TBPTT window size (0=disabled, 15=JaneEye-Net default)')
    # LR Scheduler
    parser.add_argument('--lr_scheduler', default='plateau', type=str,
                        help='LR scheduler type: plateau or exponential')
    parser.add_argument('--lr_gamma', default=0.98, type=float,
                        help='Gamma for ExponentialLR scheduler')
    # Data Augmentation
    parser.add_argument('--voxel_ch_norm', default=0, type=int,
                        help='Per-channel non-zero voxel grid normalization')
    parser.add_argument('--augment_hflip', default=0.0, type=float,
                        help='Horizontal flip probability (0=disabled)')
    parser.add_argument('--augment_noise', default=0.0, type=float,
                        help='Gaussian noise probability (0=disabled)')
    parser.add_argument('--augment_noise_min', default=0.05, type=float,
                        help='Min noise scaling factor')
    parser.add_argument('--augment_noise_max', default=2.2, type=float,
                        help='Max noise scaling factor')
    # Best Model Criterion
    parser.add_argument('--best_model_criterion', default='euclidean_dist', type=str,
                        help='Best model criterion: euclidean_dist, p3, p5, p10')
    # Training Process
    parser.add_argument('--step', default='train', help='Which step to start from')
    parser.add_argument('--run_through', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether eval val set during training')
    parser.add_argument('--score_val', default=1, type=int, help='Whether score val set during training')
    parser.add_argument('--eval_test', default=0, type=int, help='Whether eval test set during training')
    parser.add_argument('--score_test', default=0, type=int, help='Whether score test set during training')
    parser.add_argument('--eval_sp', default=1, type=int, help='Whether evaluate sparsity')
    parser.add_argument('--iter_mode', default='batch', help='Dynamic batch size.')
    parser.add_argument('--normalization', default=0, type=int, help='Normalize features')
    parser.add_argument('--pretrain_model', default=None, help='Custom pretrained model')
    parser.add_argument('--use_cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--gpu_device', default=0, type=int, help='Select GPU')
    # Regression output
    parser.add_argument('--n_classes', default=2, type=int, help='Number of outputs (2 for x,y regression)')
    return parser
