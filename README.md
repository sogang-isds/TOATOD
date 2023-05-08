# Task-Optimized Adatper for an End-to-End Task Oriented Dialog

## Prerequisite

### Install Requirements
```bash
pip install -r requirements.txt
```

### wandb setup
```bash
pip install wandb
wandb login
wandb init
```

### Download & Preprocess Data
#### MultiWOZ 2.1
```bash
cd data/multiwoz21
bash data_preparation.sh
```
#### MultiWOZ 2.2
- MultiWOZ 2.2 Data Preprocessing
First, You have to download the dataset from MultiWOZ 2.2. And then, you have to convert the dataset to the format of MultiWOZ 2.1.
```bash
cd data/multiwoz22
bash data_preparation.sh
```
#### NLU task
```bash
cd data/banking77
bash banking77_preparation.sh

cd ../clinc150
bash clinc150_preparation.sh

cd ../hwu64
bash ../hwu64_preparation.sh
```

### Download Pretrained Model
```bash
cd checkpoints

wget https://pptod.s3.amazonaws.com/Pretrain/small.zip
unzip small.zip
rm small.zip

wget https://pptod.s3.amazonaws.com/Pretrain/base.zip
unzip base.zip
rm base.zip
```

## Train & Eval of E2E TOD
### small
```bash
cd E2E_TOD
bash small_run_21.sh
bash small_run_22.sh
```

### base
```bash
cd E2E_TOD
bash base_run_21.sh
bash base_run_22.sh
```

## Train & Eval of NLU(IC)
```bash
cd IC
bash run.sh
```