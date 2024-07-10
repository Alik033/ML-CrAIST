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

## Train  
``` bash
python train.py -v "CrAIST_X2_V1" -p train --train_yaml "trainSR_X2_DIV2K.yaml"
python train.py -v "CrAIST_X3_V1" -p train --train_yaml "trainSR_X3_DIV2K.yaml"
python train.py -v "CrAIST_X4_V1" -p train --train_yaml "trainSR_X4_DIV2K.yaml"
```

## Fine-tune  
``` bash
python train.py -v "CrAIST_X2_V1" -p finetune --ckpt 79
```

## Test
**Use version "CrAIST_X2_V1" for ML-CrAIST model (Ours) and "CrAIST_X2_48" for lighter model (Ours-Li).**

-- | Ours | --  |  | -- | Ours-Li | --
--- | --- | --- | --- | --- | --- | ---
Scale | Version | Epoch | |Scale | Version | Epoch
2x | CrAIST_X2_V1 | 414 | |2x | CrAIST_X2_48 | 761
3x | CrAIST_X3_V1 | 584 | |3x | CrAIST_X2_48 | 911
4x | CrAIST_X4_V1 | 682 | |4x | CrAIST_X2_48 | 766

- e.g.,
``` bash
python test.py -v "CrAIST_X2_V1" --checkpoint_epoch 414 -t tester_Matlab --test_dataset_name "Urban100"
```
- provide dataset path in env/env.json file  
- other configurations are done using yaml files  
## Acknowledgement
- [https://github.com/Francis0625/Omni-SR](https://github.com/Francis0625/Omni-SR)
