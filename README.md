# DeltaLSTM-CBTD
PyTorch Implementation of DeltaLSTM (./nnmodels/deltalstm.py) and CBTD (./utils/util.py)

# Prerequisite
Install Miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Create an environment using the following command:
```
conda create -n pt python=3.8 numpy matplotlib pandas tqdm h5py \
    scipy jupyter scikit-learn tensorboard editdistance \
    pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

```
Activate the environment before running the script.
```
conda activate pt
```

Install Pip Packages.
```
pip install sphfile
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