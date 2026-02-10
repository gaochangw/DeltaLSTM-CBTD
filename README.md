# DeltaLSTM-CBTD
PyTorch Implementation of DeltaLSTM (./nnmodels/deltalstm.py) and CBTD (./utils/util.py)

# Prerequisite
Requires Miniconda with Python 3.13 and PyTorch (latest stable) installed in the base environment.

Install additional dependencies:
```
pip install numpy matplotlib pandas tqdm h5py scipy scikit-learn tensorboard editdistance sphfile
```

#  Run
Navigate to the project folder and run experiments with the main.py file by specifying the target dataset and step.
Please run the following commands to pretrain an LSTM network and retrain on DeltaLSTM:
1. Data Preparation:
```
python main.py --dataset_name timit --step data_prep
```
2. Feature Extraction
```
python main.py --dataset_name timit --step feat_ext
```
3. Pretrain on LSTM
```
python main.py --dataset_name timit --step train --phase pretrain --hid_type LSTM
```
4. Test LSTM
```
python main.py --dataset_name timit --step test --phase pretrain --hid_type LSTM
```
5. Retrain on DeltaLSTM
```
python main.py --dataset_name timit --step train --phase retrain --hid_type DeltaLSTM --hid_type_pretrain LSTM
```
6. Test DeltaLSTM
```
python main.py --dataset_name timit --step test --phase retrain --hid_type DeltaLSTM
```