# DANN-DDI
This repository provides a deep attention neural network framework named DANN-DDI to predict potential interactions between drug-drug pairs in multiple drug feature networks.This framework contains the code to implement the method and data on the interactions between 841 drugs.
We implement this model based on tensorflow, which enables this model to be trained with GPUs.

## Environments
- Python 3.6.8 :: Anaconda, Inc.
- Tensorflow 1.14.0

## Usage
​When using this code, you need to clone this repo and load all the files in the folder into your running environment first. Then, you should enter the root directory and run the following code:
```
    cd src
    python GenerateEmbeddings.py
	python DANN-DDI.py
```
where the file `GenerateEmbeddings.py` is to generate representation vectors of the drugs by using SDNE which is typical representation learning method, and `DANN-DDI.py` is using representation vectors to train this model and predict potential interactions between drug-drug pairs.