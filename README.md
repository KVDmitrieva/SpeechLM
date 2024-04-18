# AS project

## Installation guide

Run
```shell
pip install -r ./requirements.txt
```
to install all libs.

## Train running guide
By default, config assumes that it is used in kaggle with [ASVSpoof](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset/). 
In this case you can speed up dataset preparation by downloading data index with:
```shell
chmod +x setup.sh & ./setup.sh
```

If you use other trainig sources, you have to change `data_dir` path in config. 


In order to recreate results, use `rawnet.json`:
```shell
python3 train.py -c hw_as/configs/rawnet.json 
```

## Test running guide
Download model checkpoint with `test_setup.sh`:
```shell
chmod +x test_setup.sh & ./test_setup.sh
```
Run test with
```shell
python3 test.py \
   -c default_test_model/rawnet.json \
   -r default_test_model/model.pth
```
it will evaluate model on audio files in `test_dir`. 
You can provide another test dir path with option `-f` or evaluate model on a single audio with option `-t`.