import os
import sys
from glob import glob
import pandas as pd
from tqdm import tqdm
from sphfile import SPHFile


class DataPrepare:
    def __init__(self, dataset_path, output_path):
        # Data Path
        self.dataset_path = dataset_path
        self.output_path = output_path

        # Create a dict for label to index conversion
        self.labels = pd.read_csv(os.path.join(self.output_path, 'label_list.csv'))
        self.dict_label2int61 = {}
        self.dict_label2int48 = {}
        self.dict_label2int39 = {}
        for idx, x in enumerate(self.labels['label61']):
            self.dict_label2int61[x] = idx
        for idx, x in enumerate(self.labels['label48']):
            self.dict_label2int48[x] = idx
        for idx, x in enumerate(self.labels['label39']):
            self.dict_label2int39[x] = idx

        self.val_list = ['MCCS0', 'MRCZ0', 'MESD0', 'MMDH0', 'MNLS0',
                         'MPWM0', 'MJES0', 'FKMS0', 'MPAM1', 'MKCH0',
                         'MAJC0', 'FEDW0', 'FMCM0', 'FCAL1', 'MGJF0',
                         'FCMH1', 'MJTC0', 'MRKO0', 'FUTB0', 'FJRE0',
                         'MAHH0', 'MDAW1', 'MDBB0', 'FNMR0', 'FLBW0',
                         'MCSH0', 'MJDM1', 'MERS0', 'MRGG0', 'MJBR0',
                         'FJWB0', 'FREW0', 'MRPC0', 'MABW0', 'MCMB0',
                         'MCTT0', 'MDRM0', 'MRJO0', 'FCRH0', 'MLJB0',
                         'FCAU0', 'MDWA0', 'FJAS0', 'FRNG0', 'FCMH0',
                         'MFGK0', 'MRMS1', 'FGWR0', 'MRJM3', 'MTDT0']

    def phn_text_to_int_str(self, str_phn_text, dict_label2int):
        list_phn_text = str_phn_text.split()
        list_int_str = [int(dict_label2int[x]) for x in list_phn_text]
        # int_str = ' '.join(list_int_str)
        return list_int_str

    def map_phn_text(self, str_phn_text, type, path_target):
        phn_map_61_48_39 = pd.read_csv(os.path.join(path_target, 'phn_map_61_48_39.csv'))
        dict_61_48 = {}
        dict_61_39 = {}
        for idx, x in enumerate(phn_map_61_48_39['phn-61']):
            dict_61_48[x] = phn_map_61_48_39['phn-48'][idx]  # Create PHN-61 to PHN-48 conversion dict
            dict_61_39[x] = phn_map_61_48_39['phn-39'][idx]  # Create PHN-61 to PHN-39 conversion dict

        phn_map_48_39 = pd.read_csv(os.path.join(path_target, 'phn_map_48_39.csv'))
        dict_48_39 = {}
        for idx, x in enumerate(phn_map_48_39['phn-48']):
            dict_48_39[x] = phn_map_48_39['phn-39'][idx]  # Create PHN-48 to PHN-39 conversion dict

        list_phn_text = str_phn_text.split()
        if type == 1:  # Type 1 - PHN--61 to PHN-48 conversion
            list_mapped_phn_text = [dict_61_48[x] for x in list_phn_text]
        elif type == 2:  # Type 2 - PHN--61 to PHN-39 conversion
            list_mapped_phn_text = [dict_61_39[x] for x in list_phn_text]
        elif type == 3:  # Type 3 - PHN--48 to PHN-39 conversion
            list_mapped_phn_text = [dict_48_39[x] for x in list_phn_text]
        else:
            print("ERROR: Only type 1~3 are supportted...")
            sys.exit(1)

        # Remove space characters
        while ' ' in list_mapped_phn_text:
            list_mapped_phn_text.remove(' ')

        # Convert list of phone text into a single string seperated by space chars
        str_mapped_phn_text = ' '.join(list_mapped_phn_text)

        return str_mapped_phn_text

    def collect(self, **kwargs):
        # Standard parameters
        columns = ['group', 'speaker', 'utt', 'key', 'modality', 'path', 'label61', 'label48', 'label39', 'phn61',
                   'phn48', 'phn39']

        # Get train and test folders
        train = os.path.join(self.dataset_path, 'TRAIN')
        test = os.path.join(self.dataset_path, 'TEST')

        # Show Dataset Path
        # print("Train Data Path: ", train)
        # print("Test Data Path: ", test)

        # Get core test set according to lists
        with open(os.path.join(self.output_path, 'test_list.txt')) as f:
            list_test = f.readlines()
            list_test = [x.strip() for x in list_test]

        # Loop over folders path_dataset
        train_group = []
        val_group = []
        test_group = []
        description_list = []
        for group, folder in zip(['train', 'test'], [train, test]):
            all_file_paths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.WAV'))]
            all_file_paths = sorted(all_file_paths)
            key_idx = 0

            for idx, file_path in tqdm(enumerate(all_file_paths), total=len(all_file_paths), desc='Convert'):
                file_name = file_path.split('/')[-1].split('.')[0]
                speaker_id = file_path.split('/')[-2]

                sph = SPHFile(file_path)
                sph.write_wav(filename=file_path.replace(".WAV", ".WAVN"))

                if file_name.find('SA') == -1:  # Remove SA utterance from dataset
                    if group == 'train':
                        train_group.append(file_path)
                    elif speaker_id in list_test:
                        test_group.append(file_path)
                    elif speaker_id in self.val_list:
                        val_group.append(file_path)
                    else:
                        continue

        # Train group
        for idx, file_path in tqdm(enumerate(train_group), total=len(train_group), desc='Train'):
            file_name = file_path.split('/')[-1].split('.')[0]
            speaker_id = file_path.split('/')[-2]

            # Get utterance phone label text
            with open(file_path.split('.')[0] + '.PHN') as f:
                transcript = f.readlines()
                transcript = [x.strip() for x in transcript]
                phn61 = [x.split(' ')[-1] for x in transcript]
                phn61 = ' '.join(phn61)

            # Map PHN-61 to PHN-48 & PHN-39
            phn48 = self.map_phn_text(phn61, 1, self.output_path)
            phn39 = self.map_phn_text(phn61, 2, self.output_path)

            # Get label int strings
            label61 = self.phn_text_to_int_str(phn61, self.dict_label2int61)
            label48 = self.phn_text_to_int_str(phn48, self.dict_label2int48)
            label39 = self.phn_text_to_int_str(phn39, self.dict_label2int39)

            key_idx += 1
            row = {}
            row['group'] = 'train'
            row['speaker'] = speaker_id
            row['utt'] = file_name
            row['key'] = 'train_' + str(key_idx)
            row['modality'] = 'audio'
            row['path'] = file_path.replace(".WAV", ".WAVN")
            row['label61'] = label61
            row['label48'] = label48
            row['label39'] = label39
            row['phn61'] = phn61
            row['phn48'] = phn48
            row['phn39'] = phn39
            description_list.append(row)

        # Val group
        for idx, file_path in tqdm(enumerate(val_group), total=len(val_group), desc='Val'):
            file_name = file_path.split('/')[-1].split('.')[0]
            speaker_id = file_path.split('/')[-2]

            # Get utterance phone label text
            with open(file_path.split('.')[0] + '.PHN') as f:
                transcript = f.readlines()
                transcript = [x.strip() for x in transcript]
                phn61 = [x.split(' ')[-1] for x in transcript]
                phn61 = ' '.join(phn61)

            # Map PHN-61 to PHN-48 & PHN-39
            phn48 = self.map_phn_text(phn61, 1, self.output_path)
            phn39 = self.map_phn_text(phn61, 2, self.output_path)

            # Get label int strings
            label61 = self.phn_text_to_int_str(phn61, self.dict_label2int61)
            label48 = self.phn_text_to_int_str(phn48, self.dict_label2int48)
            label39 = self.phn_text_to_int_str(phn39, self.dict_label2int39)

            key_idx += 1
            row = {}
            row['group'] = 'val'
            row['speaker'] = speaker_id
            row['utt'] = file_name
            row['key'] = 'val_' + str(key_idx)
            row['modality'] = 'audio'
            row['path'] = file_path.replace(".WAV", ".WAVN")
            row['label61'] = label61
            row['label48'] = label48
            row['label39'] = label39
            row['phn61'] = phn61
            row['phn48'] = phn48
            row['phn39'] = phn39
            description_list.append(row)

        # Test group
        for idx, file_path in tqdm(enumerate(test_group), total=len(test_group), desc='Test'):
            file_name = file_path.split('/')[-1].split('.')[0]
            speaker_id = file_path.split('/')[-2]

            # Get utterance phone label text
            with open(file_path.split('.')[0] + '.PHN') as f:
                transcript = f.readlines()
                transcript = [x.strip() for x in transcript]
                phn61 = [x.split(' ')[-1] for x in transcript]
                phn61 = ' '.join(phn61)

            # Map PHN-61 to PHN-48 & PHN-39
            phn48 = self.map_phn_text(phn61, 1, self.output_path)
            phn39 = self.map_phn_text(phn61, 2, self.output_path)

            # Get label int strings
            label61 = self.phn_text_to_int_str(phn61, self.dict_label2int61)
            label48 = self.phn_text_to_int_str(phn48, self.dict_label2int48)
            label39 = self.phn_text_to_int_str(phn39, self.dict_label2int39)

            key_idx += 1
            row = {}
            row['group'] = 'test'
            row['speaker'] = speaker_id
            row['utt'] = file_name
            row['key'] = 'test_' + str(key_idx)
            row['modality'] = 'audio'
            row['path'] = file_path.replace(".WAV", ".WAVN")
            row['label61'] = label61
            row['label48'] = label48
            row['label39'] = label39
            row['phn61'] = phn61
            row['phn48'] = phn48
            row['phn39'] = phn39
            description_list.append(row)

        df = pd.DataFrame(description_list, columns=columns)
        description_file = os.path.join(self.output_path, 'description.csv')
        df.to_csv(description_file, index=False)
