# [ICML 2023] _Decentralized SGD and Average-direction SAM are Asymptotically Equivalent_

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.02913-b31b1b.svg)](https://arxiv.org/abs/2306.02913)
[![blog](https://img.shields.io/badge/blogpost-%20-blue?style=social&logo=disqus)](https://raiden-zhu.github.io/resources/)
[![twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Faaron_lou%2Fstatus%2F1646528998594482176%3Fs%3D20)](https://twitter.com/Raiden13238619/status/1683152896307896322?s=20)
[![slides](https://img.shields.io/badge/Slides-blue)](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM/blob/main/Slides_ICML_2023_Decentralized_SGD_and_Average_direction_SAM_are_Asymptotically_Equivalent%20.pdf)
[![poster](https://img.shields.io/badge/Poster-green)](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM/blob/main/Poster_ICML2023_D_SGD_as_SAM.pdf)


### The Best of All Worlds: Embracing Decentralization for Improved Communication Efficiency, Privacy, and Generalization

The repository contains the offical implementation of the paper

> [**ICML 2023**] **Decentralized SGD and Average-direction SAM are Asymptotically Equivalent**

<!-- Please kindly refer to our [**arXiv version**](https://arxiv.org/abs/2306.02913) for the latest updates and more detailed information. -->

## Overview
**Motivating question:** The Best of All Worlds?
Can we guarantee communication effiency, privacy and generalizablity all at once?
Our recent ICML 2023 paper proves that **decentralized training** might be the anwer!

**TLDR**: The first work on the surprising sharpness-aware minimization nature of decentralized learning. We provide a completely new perspective to understand decentralization, which helps to bridge the gap between theory and practice in decentralized learning.

**Abstract**: Decentralized stochastic gradient descent (D-SGD) allows collaborative learning on massive devices simultaneously without the control of a central server. However, existing theories claim that decentralization invariably undermines generalization. In this paper, we challenge the conventional belief and present a completely new perspective for understanding decentralized learning. We prove that D-SGD implicitly minimizes the loss function of an average-direction Sharpness-aware minimization (SAM) algorithm under general non-convex non-$\beta$-smooth settings. This surprising asymptotic equivalence reveals an intrinsic regularization-optimization trade-off and three advantages of decentralization: (1) there exists a free uncertainty evaluation mechanism in D-SGD to improve posterior estimation; (2) D-SGD exhibits a gradient smoothing effect;  and (3) the sharpness regularization effect of D-SGD does not decrease as total batch size increases, which justifies the potential generalization benefit of D-SGD over centralized SGD (C-SGD) in large-batch scenarios. 

![image](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM/blob/main/files/An%20illustration%20of%20centralized%20SGD%20and%20decentralized%20SGD.png)

![image](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM/blob/main/files/The%20validation%20accuracy%20comparison%20of%20C-SGD%20and%20D-SGD%20(ring%20topology)%20on%20CIFAR-10.png)

## Environment Setup
Requisite packages can be installed directly from the `requirements.txt`.
```
pip install -r requirements.txt
```

## Example of usage

#### Train ResNet-18 on CIFAR-10 using D-SGD and C-SGD with 1024 total batch sizes.

```
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "csgd" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
```

#### Train ResNet-18 on CIFAR-10 using D-SGD and C-SGD with 8192 total batch sizes.

```
python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "csgd" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

python main.py --dataset_name "CIFAR10" --image_size 56 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 60 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
```

More detailed scripts can be found in the "scripts" folder.

The 3D local loss landscape visualization is based on [![visualization](https://img.shields.io/badge/Github-Loss%20Landscape%20Visualization-blue)](https://github.com/tomgoldstein/loss-landscape).

![image](https://github.com/Raiden-Zhu/ICML-2023-DSGD-and-SAM/blob/main/files/Minima%203D%20visualization%20of%20ResNet-18%20trained%20on%20CIFAR-10%20using%20C-SGD%20and%20D-SGD%20(ring%20topology).png)

## Citing this repository

Please cite our paper if you find this repo useful in your work:

```

@InProceedings{pmlr-v202-zhu23e,
  title = 	 {Decentralized {SGD} and Average-direction {SAM} are Asymptotically Equivalent},
  author =       {Zhu, Tongtian and He, Fengxiang and Chen, Kaixuan and Song, Mingli and Tao, Dacheng},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {43005--43036},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/zhu23e/zhu23e.pdf},
  url = 	 {https://proceedings.mlr.press/v202/zhu23e.html},
  abstract = 	 {Decentralized stochastic gradient descent (D-SGD) allows collaborative learning on massive devices simultaneously without the control of a central server. However, existing theories claim that decentralization invariably undermines generalization. In this paper, we challenge the conventional belief and present a completely new perspective for understanding decentralized learning. We prove that D-SGD implicitly minimizes the loss function of an average-direction Sharpness-aware minimization (SAM) algorithm under general non-convex non-$\beta$-smooth settings. This surprising asymptotic equivalence reveals an intrinsic regularization-optimization trade-off and three advantages of decentralization: (1) there exists a free uncertainty evaluation mechanism in D-SGD to improve posterior estimation; (2) D-SGD exhibits a gradient smoothing effect; and (3) the sharpness regularization effect of D-SGD does not decrease as total batch size increases, which justifies the potential generalization benefit of D-SGD over centralized SGD (C-SGD) in large-batch scenarios.}
}

```

## Contact

Please feel free to contact via email (<raiden@zju.edu.cn>) or Wechat (RaidenT_T) if you have any questions.
