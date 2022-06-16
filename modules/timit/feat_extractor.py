import ast
import os
import numpy as np
import pandas as pd
from modules.timit import log
from utils.feature.filter_bank import compute_filter_bank
from tqdm import tqdm
from scipy.io import wavfile
from utils.util_feat import write_dataset


class FeatExtractor:
    def __init__(self):
        self.description_name = 'description.csv'

    def extract(self, args):
        # Load dataframe
        df_description = pd.read_csv(os.path.join('data', args.dataset_name, self.description_name))

        # Initialization
        np.random.seed(2)

        X_train = []
        X_val = []
        X_test = []

        y_train = []
        y_val = []
        y_test = []

        y_train_39 = []
        y_val_39 = []
        y_test_39 = []

        y_train_48 = []
        y_val_48 = []
        y_test_48 = []

        y_train_61 = []
        y_val_61 = []
        y_test_61 = []

        # Get File ID
        _, trainfile, valfile, testfile = log.gen_dataset_name(args)

        # Loop over dataframe
        for row in tqdm(df_description.itertuples(), total=df_description.shape[0]):
            # Get features
            sample_rate, sample = wavfile.read(row.path)
            features = compute_filter_bank(signal=sample,
                                           sample_rate=sample_rate,
                                           frame_size=args.frame_size,
                                           frame_stride=args.frame_stride,
                                           n_filt=args.n_filt,
                                           gain_ctrl=args.gain_ctrl,
                                           gradient=args.gradient,
                                           NFFT=args.nfft,
                                           MFCC=False)

            if row.group == 'train':
                X_train.append(features)
                y_train_39.extend([ast.literal_eval(row.label39)])
                y_train_48.extend([ast.literal_eval(row.label48)])
                y_train_61.extend([ast.literal_eval(row.label61)])
            elif row.group == 'val':
                X_val.append(features)
                y_val_39.extend([ast.literal_eval(row.label39)])
                y_val_48.extend([ast.literal_eval(row.label48)])
                y_val_61.extend([ast.literal_eval(row.label61)])
            elif row.group == 'test':
                X_test.append(features)
                y_test_39.extend([ast.literal_eval(row.label39)])
                y_test_48.extend([ast.literal_eval(row.label48)])
                y_test_61.extend([ast.literal_eval(row.label61)])

        def get_dataset(x, y_39, y_48, y_61, phn):
            features = np.concatenate(x, axis=0).astype(np.float32)
            # features = np.array(x, dtype=object).astype(np.float32)
            feature_lengths = np.asarray([sample.shape[0] for sample in x]).astype(np.int32)
            targets_39 = np.concatenate(y_39).astype(np.int32)
            targets_48 = np.concatenate(y_48).astype(np.int32)
            targets_61 = np.concatenate(y_61).astype(np.int32)
            target_lengths = np.asarray([len(target) for target in y_39]).astype(np.int32)
            n_features = features.shape[-1]
            if phn == 39:
                n_classes = np.max(targets_39).astype(np.int32) + 1
            elif phn == 48:
                n_classes = np.max(targets_48).astype(np.int32) + 1
            elif phn == 61:
                n_classes = np.max(targets_61).astype(np.int32) + 1
            else:
                raise RuntimeError('You can only use phoneme number of 39, 48 or 61.')
            dict_dataset = {'features': features, 'feature_lengths': feature_lengths, 'target_lengths': target_lengths,
                            'targets_39': targets_39, 'targets_48': targets_48, 'targets_61': targets_61,
                            'n_features': n_features, 'n_classes': n_classes}
            return dict_dataset

        # Process Datasets
        dataset_train = get_dataset(X_train, y_train_39, y_train_48, y_train_61, args.phn)
        dataset_val = get_dataset(X_val, y_val_39, y_val_48, y_val_61, args.phn)
        dataset_test = get_dataset(X_test, y_test_39, y_test_48, y_test_61, args.phn)

        # Write Dataset
        write_dataset(os.path.join('feat', args.dataset_name, trainfile), dataset_train)
        write_dataset(os.path.join('feat', args.dataset_name, valfile), dataset_val)
        write_dataset(os.path.join('feat', args.dataset_name, testfile), dataset_test)

        print("Feature stored in: ", os.path.join('feat', args.dataset_name))
        print("Feature Extraction Completed...")
        print(" ")


def compute_fbank(signal, sample_rate, frame_size=0.025, frame_stride=0.01, NFFT=512, nfilt=40, MFCC=0):
    PRINT_INFO = 0

    # Initialization
    np.random.seed(2)

    sample_rate = float(sample_rate)

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32, copy=False)]

    #batch_idx = 0
    #batch_size = 8
    #norm_frames = np.asarray([]).reshape(0, frames.shape[1])
    #num_batch = np.ceil(num_frames / float(batch_size)).astype(int)
    #for i in range(0, num_batch):
    #    if i == (num_batch - 1):
    #        batch_curr = frames[batch_size * i:, :]
    #    else:
    #        batch_curr = frames[batch_size * i:batch_size * (i + 1), :]
    #    stack_curr = batch_curr.reshape(-1)
    #    mean = np.mean(stack_curr)
    #    std = np.std(stack_curr, ddof=1)
    #    batch_curr -= mean
    #    if (std != 0):
    #        batch_curr /= std
    #    norm_frames = np.vstack((norm_frames, batch_curr))
#
    #frames = norm_frames

    # Windowing
    frames *= np.hamming(frame_length)

    # Fourier-Transform and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20*np.log(filter_banks)  # dB
    #energy = np.sum(np.square(mag_frames), axis=1).reshape(-1, 1)
    #filter_banks = np.hstack((filter_banks, energy))
    temp_derivative_first_order = np.gradient(filter_banks, edge_order=1, axis=0)
    temp_derivative_second_order = np.gradient(temp_derivative_first_order, edge_order=2, axis=0)
    # print(temp_derivative_first_order-temp_derivative_second_order)
    features = np.hstack((filter_banks, temp_derivative_first_order, temp_derivative_second_order))

    #features = filter_banks
    if (PRINT_INFO == 1):
        print("Length of Signal: " + str(signal_length))
        print("Length of Frames: " + str(frame_length))
        print("Step of Frames:   " + str(frame_step))
        print("Number of Frames: " + str(num_frames))
        print("Length of Padded Signal: " + str(pad_signal_length))
        print(indices.shape)
        print(indices)
        print("Frames:")
        print(frames)

    if (MFCC == 1):
        num_ceps = 39
        cep_lifter = 22
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift  # *
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        # print(mfcc.shape)
        features = mfcc
    #print(features.shape)
    return features