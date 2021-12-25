# Instance-Level Forward Correction
This is the code for the paper: [Confidence Scores Make Instance-dependent Label-noise Learning Possible](https://arxiv.org/abs/2001.03772)
Antonin Berthon, Bo Han, Gang Niu, Tongliang Liu, Masashi Sugiyama

To be presented at [ICML 2021](https://icml.cc/Conferences/2021).


If you find this code useful in your research then please cite

```
@inproceedings{berthon2021confidence,
  title={Confidence scores make instance-dependent label-noise learning possible},
  author={Berthon, Antonin and Han, Bo and Niu, Gang and Liu, Tongliang and Sugiyama, Masashi},
  booktitle={ICML},
  pages={825--836},
  year={2021}
}
```

## Setup
All the experiments were ran on NVIDIA Tesla K80 GPUs
from [Google Colab](https://research.google.com/colaboratory/) notebooks.

## Data
As described in the paper, we experiment on a synthetic dataset as well as CIFAR10 and SVHN by generating some instance-dependant label noise. Please contact _berthon.antonin[at]gmail[dot]com_ to receive a download link.

## Synthetic, CIFAR10 and SVHN experiments
You can run the experiments on synthetic, CIFAR10 and SVHN experiments on the following [Google Colab Notebook](https://colab.research.google.com/drive/1bFDmVUlpYM70rN94WK0nw82-Gbe3Urp9?usp=sharing).

## Clothing1M experiment
An example of training of ILFC and benchmark models on Clothing1M can be found in the following [Google Colab Notebook](https://colab.research.google.com/drive/1NDQQJE25Wus630xJ9UjvjDx8si9vw2dl?usp=sharing). 

Train ILFC on the Clothing1M dataset:
```
python experiments.ILFC_clothing.py --seed 123  --import_data_path <path to clothing1m dataset> \
--noisy_model_epochs 0 --mom_decay_start 10 --warm_start 3 --bs 64 --method mean \
--model resnet18 --nb_epoch 60 --noisy_model resnet18 --lr 0.0001 
--train_limit 2000 --optim "Adam" \
--result_dir <results export path> \
--model_export <model export path> 
# Optional
# --noisy_model_import <path to naive model checkpoint> 
# --noisy_model_export <path to naive model checkpoint> 
```

Train a benchmark method on the Clothing1M dataset:
```
python experiments.COMP_clothing.py --seed 1 --import_data_path <path to clothing&m dataset>" \
--nb_epoch 10 --lr 0.0001 --comp_model <One of {"MAE", "LQ", "F"}> \
--mom_decay_start 5 --bs 64 --train_limit 6000 \
--result_dir <results export path> \
--model_export <model export path> 
# --model_import <model import path> 
```
