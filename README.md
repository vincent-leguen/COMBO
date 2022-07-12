# COMBO - Complementing Brightness Constancy with Deep Networks for Optical Flow Prediction
[Vincent Le Guen](https://www.linkedin.com/in/vincentleguen/), [Clément Rambour](http://cedric.cnam.fr/~rambourc/), and [Nicolas Thome](http://cedric.cnam.fr/~thomen/)

Code for our ECCV 2022 paper "Complementing Brightness Constancy with Deep Networks for Optical Flow Prediction": https://arxiv.org/abs/2207.03790

<img src="https://github.com/vincent-leguen/COMBO/blob/main/combo_model.jpg" width="1000">

## Abstract
State-of-the-art methods for optical flow estimation rely on deep learning, which require complex sequential training schemes to reach optimal performances on real-world data. In this work, we introduce the COMBO deep network that explicitly exploits the brightness constancy (BC) model used in traditional methods. Since BC is an approximate physical model violated in several situations, we propose to train a physically-constrained network complemented with a data-driven network. We introduce a unique and meaningful flow decomposition between the physical prior and the data-driven complement, including an uncertainty quantification of the BC model. We derive a joint training scheme for learning the different components of the decomposition ensuring an optimal cooperation, in a supervised but also in a semi-supervised context. Experiments show that COMBO can improve performances over state-of-the-art supervised networks, e.g. RAFT, reaching state-of-the-art results on several benchmarks. We highlight how COMBO can leverage the BC model and adapt to its limitations. Finally, we show that our semi-supervised method can significantly simplify the training procedure. 

## Code


If you find this code useful for your research, please cite our paper:

```
@incollection{leguen22combo,
title = {Complementing Brightness Constancy with Deep Networks for Optical Flow Prediction},
author = {Le Guen, Vincent and Rambour, Clément and Thome, Nicolas},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2022}
}
```
