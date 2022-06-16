###########################################################################################################
# TIMIT
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
    parser.add_argument('--labelfile', default='label_list_30.csv',
                        help='CSV File of keyword & Label pairs')
    parser.add_argument('--zero_padding', default='head',
                        help='Method of padding zeros to samples in a batch')
    # Feature Extraction
    parser.add_argument('--feat_name', default='', help='A string append to the dataset file names')
    parser.add_argument('--frame_size', default=0.025, type=float, help='Frame size of signals')
    parser.add_argument('--frame_stride', default=0.010, type=float, help='Frame stride of signals')
    parser.add_argument('--n_filt', default=40, type=int, help='Number of log filter banks')
    parser.add_argument('--nfft', default=1024, type=int, help='Number of log filter banks')
    parser.add_argument('--gain_ctrl', default=0, type=int,
                        help='Real-time frame normalization for audio gain control')
    parser.add_argument('--gradient', default=1, type=int,
                        help='Append gradient of filter banks to features')
    # Hyperparameters
    parser.add_argument('--seed', default=4, type=int,
                        help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=256, type=int, help='Batch size for evaluation.')
    parser.add_argument('--n_epochs', default=50, type=int, help='Number of epochs to train for.')
    parser.add_argument('--loss', default='ctc', help='Loss function.')
    parser.add_argument('--opt', default='ADAMW', help='Which optimizer to use (ADAM or SGD)')
    parser.add_argument('--lr_schedule', default=1, type=int, help='Whether enable learning rate scheduling')
    parser.add_argument('--lr', default=5e-4, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--decay_factor', default=0.5, type=float, help='Learning rate')
    parser.add_argument('--patience', default=5, type=float, help='Learning rate')
    parser.add_argument('--beta', default=0, type=float,
                        help='Best model used in testing, either "per", or "vloss"')
    parser.add_argument('--clip_grad_norm_max', default=100, type=float,
                        help='Best model used in testing, either "per", or "vloss"')
    # General Network Settings
    parser.add_argument('--hid_type', default='LSTM', help='Classification layer type')
    parser.add_argument('--hid_type_pretrain', default='LSTM',
                        help='Pretrain model type to be load')
    parser.add_argument('--hid_layers', default=2, type=int, help='Number of classification layers')
    parser.add_argument('--hid_size', default=1024, type=int, help='Size of classification layers')
    parser.add_argument('--hid_dropout', default=0, type=float,
                        help='Dropout rate of the classification layer')
    parser.add_argument('--fc_type', default='FC', help='FC layer type')
    parser.add_argument('--fc_extra_size', default=1024, type=int,
                        help='Whether add FC layer after the classfication layer')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Factor of weight decay')
    # Convolutional Network Settings
    parser.add_argument('--context_window_size', default=256, type=int, help='Context window size')
    # Delta Network Settings
    parser.add_argument('--hardsigmoid', default=0, type=int, help='Use hardsigmoid')
    parser.add_argument('--hardtanh', default=0, type=int, help='Use hardtanh')
    parser.add_argument('--qa', default=1, type=int,
                        help='Number of fraction bits after activation decimal point')
    parser.add_argument('--aqi', default=8, type=int,
                        help='Number of integer bits before activation decimal point')
    parser.add_argument('--aqf', default=8, type=int,
                        help='Number of fraction bits after activation decimal point')
    parser.add_argument('--qw', default=1, type=int,
                        help='Number of fraction bits after weight decimal point')
    parser.add_argument('--wqi', default=1, type=int,
                        help='Number of integer bits before weight decimal point')
    parser.add_argument('--wqf', default=7, type=int,
                        help='Number of fraction bits after weight decimal point')
    parser.add_argument('--nqi', default=2, type=int,
                        help='Number of integer bits before LUT output decimal point')
    parser.add_argument('--nqf', default=4, type=int,
                        help='Number of fraction bits after LUT output decimal point')  # 5
    parser.add_argument('--qa_fc_extra', default=1, type=int,
                        help='Number of fraction bits after activation decimal point for the extra FC layer')
    parser.add_argument('--aqi_fc_extra', default=8, type=int,
                        help='Number of integer bits before activation decimal point for the extra FC layer')
    parser.add_argument('--aqf_fc_extra', default=8, type=int,
                        help='Number of fraction bits after activation decimal point for the extra FC layer')
    parser.add_argument('--qa_fc_final', default=1, type=int,
                        help='Number of fraction bits after activation decimal point for the final FC layer')
    parser.add_argument('--aqi_fc_final', default=8, type=int,
                        help='Number of integer bits before activation decimal point for the final FC layer')
    parser.add_argument('--aqf_fc_final', default=8, type=int,
                        help='Number of fraction bits after activation decimal point for the final FC layer')
    parser.add_argument('--th_x', default=0.1, type=float, help='Whether quantize the network weights')
    parser.add_argument('--th_h', default=0.1, type=float, help='Whether quantize the network weights')
    # DeltaLSTM
    parser.add_argument('--cbtd', default=1, type=int,
                        help='Whether use Column-Balanced Weight Dropout')
    parser.add_argument('--gamma_rnn', default=0.94, type=float, help='Target sparsity of RNN layer')
    parser.add_argument('--gamma_fc', default=0.94, type=float, help='Target sparsity of FC Layer')
    parser.add_argument('--alpha_anneal_epoch', default=30, type=int, help='Number of epochs for alpha to reach 1')
    parser.add_argument('--balance_pe', default=1, type=float, help='Balance the workload of PEs in CBTD')
    parser.add_argument('--num_array', default=8, type=int, help='Number of MAC Arrays (M)')
    parser.add_argument('--num_array_pe', default=64, type=int, help='Number of PEs per MAC Array (N)')
    # Scoring Settings
    parser.add_argument('--smooth', default=1, type=int, help='Whether smooth the posterior over time')
    parser.add_argument('--smooth_window_size', default=60, type=int, help='Posterior smooth window size')
    parser.add_argument('--confidence_window_size', default=80, type=int,
                        help='Confidence score window size')
    parser.add_argument('--fire_threshold', default=0, type=float,
                        help='Threshold for train (1) firing a decision')
    # Training Process
    parser.add_argument('--step', default='train', help='Which step to start from')
    parser.add_argument('--run_through', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--eval_val', default=1, type=int, help='Whether eval val set during training')
    parser.add_argument('--score_val', default=1, type=int, help='Whether score val set during training')
    parser.add_argument('--eval_test', default=1, type=int, help='Whether eval test set during training')
    parser.add_argument('--score_test', default=1, type=int, help='Whether score test set during training')
    parser.add_argument('--eval_sp', default=1, type=int, help='Whether run through rest steps')
    parser.add_argument('--iter_mode', default='batch', help='Dynamic batch size.')
    parser.add_argument('--normalization', default=1, type=int, help='Custom pretrained model')
    parser.add_argument('--pretrain_model', default=None, help='Custom pretrained model')
    parser.add_argument('--use_cuda', default=1, type=int, help='Use GPU yes/no')
    parser.add_argument('--gpu_device', default=0, type=int, help='Select GPU')
    # TIMIT
    parser.add_argument('--phn', default=48, type=int, help='Number of phonemes used in training')
    return parser
