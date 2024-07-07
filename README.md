# ML-CrAIST : Multi-scale Low-high Frequency Information-based Cross Attention with Image Super-resolving Transformer

**The official repository with Pytorch**

## Installation

**Python 3.9.12**

- create virtual environment
``` bash
python3 -m venv ./venv_utsav
```

- activte virtual environment
``` bash
source venv_utsav/bin/activate
```

- install dependencies  
``` bash
pip3 install torch torchvision opencv-python matplotlib pyyaml tqdm tensorboardX tensorboard einops thop
```

## train  
``` bash
python train.py -v "CrAIST_X2_V1" -p train --train_yaml "trainSR_X2_DIV2K.yaml"
python train.py -v "CrAIST_X3_V1" -p train --train_yaml "trainSR_X3_DIV2K.yaml"
python train.py -v "CrAIST_X4_V1" -p train --train_yaml "trainSR_X4_DIV2K.yaml"
```

## fine-tune  
``` bash
python train.py -v "CrAIST_X2_V1" -p finetune --ckpt 79
```

## test
**Use version "CrAIST_X2_V1" for large model and "CrAIST_X2_48" for lighter model.**
``` bash
python test.py -v "CrAIST_X2_V1" --checkpoint_epoch 414 -t tester_Matlab --test_dataset_name "Urban100"
python test.py -v "CrAIST_X3_V1" --checkpoint_epoch 584 -t tester_Matlab --test_dataset_name "Urban100"
python test.py -v "CrAIST_X4_V1" --checkpoint_epoch 682 -t tester_Matlab --test_dataset_name "Urban100"
```

- provide dataset path in env/env.json file  
- other configurations are done using yaml files  

