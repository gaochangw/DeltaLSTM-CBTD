import editdistance
import numpy as np
import pandas as pd
from tqdm import tqdm

def gen_meter_args(args, n_classes, **kwargs):
    dict_meter_args = {'n_classes': n_classes, 'source_phn': args.phn, 'smooth': args.smooth,
                       'smooth_window_size': args.smooth_window_size,
                       'confidence_window_size': args.confidence_window_size, 'zero_padding': args.zero_padding,
                       'fire_threshold': args.fire_threshold, 'blank': 0, 'idx_silence': 0, 'threshold': 0}
    if args.dataset_name == 'timit':
        dict_meter_args['phn'] = args.phn
    return dict_meter_args


class Meter:
    def __init__(self, dict_meter_args):
        self.guessed_labels = []
        self.target_labels = []
        self.blank = dict_meter_args['blank']
        self.source_phn = dict_meter_args['source_phn']

        # Create PHN-48 to PHN-39 mapping dict
        phn_map_48_39 = pd.read_csv('./data/timit/phn_map_48_39.csv')
        self.dict_48_39_int = {0: 0}
        for idx, x in enumerate(phn_map_48_39['phn-48-int']):
            self.dict_48_39_int[x + 1] = int(
                phn_map_48_39['phn-39-int'][idx]) + 1  # Create PHN-48 to PHN-39 conversion dict

        # Create PHN-61 to PHN-39 mapping dict
        phn_map_61_39 = pd.read_csv('./data/timit/phn_map_61_48_39.csv')
        self.dict_61_39_int = {0: 0}
        for idx, x in enumerate(phn_map_61_39['phn-61-int']):
            self.dict_61_39_int[x + 1] = int(
                phn_map_61_39['phn-39-int'][idx]) + 1  # Create PHN-48 to PHN-39 conversion dict

    def clear_data(self):
        # Clear Data Buffers
        self.guessed_labels = []
        self.target_labels = []

    def extend_data(self, outputs, targets, target_lengths):
        guessed_batch_labels = self.convert_prediction_to_transcription(outputs, self.blank, source_phn=self.source_phn)
        self.guessed_labels.extend(guessed_batch_labels)

        # remove the easier batch labels step once the hdf5 are fixed
        easier_batch_labels = self.convert_from_ctc_to_easy_labels(targets,
                                                                   target_lengths)  # ease access to warp-ctc labels
        target_batch_labels = [vec2str(label) for label in easier_batch_labels]  # prepare string
        self.target_labels.extend(target_batch_labels)

        return self.guessed_labels, self.target_labels

    def get_metrics(self, dict_stat):
        phone_error_rate = calculate_error_rates(self.target_labels, self.guessed_labels)
        dict_stat['per'] = phone_error_rate
        dict_stat['lr_criterion'] = phone_error_rate
        self.guessed_labels = []
        self.target_labels = []
        return dict_stat

    def get_metrics_preserve(self):
        phone_error_rate = calculate_error_rates(self.target_labels, self.guessed_labels)
        return phone_error_rate

    def clear(self):
        self.guessed_labels = []
        self.target_labels = []

    def greedy_decoder_map_phn(self, sample_prediction):
        guess_vec = np.argmax(sample_prediction, axis=1)
        return guess_vec

    def convert_prediction_to_transcription(self, net_out: list, blank, source_phn=48):
        """
        Convert Prediction to Transcription
        :param net_out: list of network outputs of all batches
        :param blank:
        :param decoder:
        :param beam_width:
        :param source_phn:
        :return:
        """
        guessed_labels_of_all_batches = []
        # Iterate through batches
        for prediction in tqdm(net_out):
            # Shape of prediction: (N, T, F)
            # Use greedy decoder
            guessed_labels = [self.greedy_decoder_map_phn(phrase) for phrase in prediction]
            for guess_vec in guessed_labels:
                if source_phn == 61:
                    # Remove label 'q' ()
                    guess_vec[np.where(guess_vec == 58)] = 0
                    for i in range(0, guess_vec.shape[0]):
                        guess_vec[i] = self.dict_61_39_int[guess_vec[i]]
                elif source_phn == 48:
                    for i in range(0, guess_vec.shape[0]):
                        guess_vec[i] = self.dict_48_39_int[guess_vec[i]]
                guess_vec_elim = vec2str(eliminate_duplicates_and_blanks(guess_vec, blank))
                guessed_labels_of_all_batches.append(guess_vec_elim)

        return guessed_labels_of_all_batches

    def convert_from_ctc_to_easy_labels(self, targets: list, target_lengths: list):
        labels = []
        # Iterate through batches
        for target, target_length in zip(targets, target_lengths):
            curr_idx = 0
            curr_label = 0
            while curr_idx < len(target):
                curr_len = target_length[curr_label]
                label_list = target[curr_idx:curr_idx + curr_len]
                labels.append([item for item in label_list])
                curr_idx += curr_len
                curr_label += 1
        return labels


def vec2str(guess_vec):
    guessed_label = '-'.join([str(item) for item in guess_vec])

    return guessed_label


def eliminate_duplicates_and_blanks(guess_vec, blank):
    rv = []
    # Remove duplicates
    for item in guess_vec:
        if (len(rv) == 0 or item != rv[-1]):
            rv.append(item)

    # Remove blanks (warp ctc label: label 0, tensorflow: last label)
    final_rv = []
    for item in rv:
        if item != blank:
            final_rv.append(item)
    return final_rv


def calculate_error_rates(target_labels, guessed_labels):
    # Phone error rate
    chars_wrong = 0
    total_chars = 0
    for idx, target in enumerate(target_labels):
        guess_chars = guessed_labels[idx].split('-')
        target_chars = target.split('-')
        errors = int(editdistance.eval(target_chars, guess_chars))
        chars_wrong += errors
        total_chars += len(target_chars)
    CER = float(chars_wrong) / total_chars

    return CER
