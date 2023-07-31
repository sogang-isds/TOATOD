# Task-Optimized Adatper for an End-to-End Task Oriented Dialog

This repository contains the code for the paper [Task-Optimized Adapter for an End-to-End Task Oriented Dialog](https://arxiv.org/abs/2305.02468) (Findings of ACL 2023).

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
#### NLU (IC) task
```bash
cd data/banking77
bash banking77_preparation.sh

cd ../clinc150
bash clinc150_preparation.sh

cd ../hwu64
bash hwu64_preparation.sh
```

### Download Pre-trained Weights
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

## Train & Eval of NLU (IC)
```bash
cd IC
bash run.sh
```


## TOATOD Checkpoints

If you want to test our best models, download the checkpoint files from the following links.

|    Task     |   Dataset    |                                                                Model (small)                                                                 |                                                                 Model (base)                                                                 |
|:-----------:|:------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
|   E2E TOD   | MultiWOZ 2.1 | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EjURZdvSD25OjKJ3_OJl35wBlked50lDy55ZSFL9amNY5A?e=MhnEfu) | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EtVHb-COv8hCpcMJ0lyBygABMF_P1WaLysn_6Uos9QYzlQ?e=zoopeX) |
|             | MultiWOZ 2.2 | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EjURZdvSD25OjKJ3_OJl35wBlked50lDy55ZSFL9amNY5A?e=MhnEfu) | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EtVHb-COv8hCpcMJ0lyBygABMF_P1WaLysn_6Uos9QYzlQ?e=zoopeX) |
|     NLU     |  banking77   | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EpVHo_TELeJEn6ifNQLguNIBHEodcDs02v3tO-A_I6H5-A?e=2aryLT) | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/Equ6Ayt0vHtCsiYflCzyOl8BkQocXb4vY0m5T1ePRUPnGw?e=B19Gs7) |
|             |   clinc150   | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EtcFvjBiTqNFgH3fxtcYU3UBLApPPwM5qhg74xz_F68IEQ?e=yseELd) | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EgjbVlfBjLlNlWiJ7xvK1fEB-UKqPJsCJBO4mlKrW1whRg?e=snGIvQ) |
|             |    hwu64     | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/ErbWXhoGlTJNuUWXmLgRKy0B87obgT3-GQetzkVhvb2iDg?e=CbXlDh) | [Link](https://sogang365-my.sharepoint.com/:f:/g/personal/jhlee22_o365_sogang_ac_kr/EsWc9Bs64WpDgU_em8_lSccBu5O4VHDXGnPjqfSWXUJLXw?e=Khyg5d) |
 

### Reference

We heavily referenced the code from [PPTOD](https://github.com/awslabs/pptod) and would like to express our gratitude to the Amazon crew.

```
@article{su2021multitask,
   author = {Yixuan Su and
             Lei Shu and
             Elman Mansimov and
             Arshit Gupta and
             Deng Cai and
             Yi{-}An Lai and
             Yi Zhang},
   title     = {Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System},
   booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)",

   year      = {2022},
   url       = {https://arxiv.org/abs/2109.14739}
}
```
