# MachineMem

This is the official PyTorch implementation of "What Images are More Memorable to Machines?" This repo currently supports the training and testing of MachineMem predictor and instructions for GANalyze. Codes of MachineMem measurer will be released around March 2023. 

[What Images are More Memorable to Machines?](https://arxiv.org/abs/2201.12078)<br>
[Junlin Han](https://junlinhan.github.io/), [Huangying Zhan](https://huangying-zhan.github.io/), Jie Hong, Pengfei Fang, [Hongdong Li](http://users.cecs.anu.edu.au/~hongdong/), [Lars Petersson](https://people.csiro.au/P/L/Lars-Petersson),  [Ian Reid](https://cs.adelaide.edu.au/~ianr/)<br>
DATA61-CSIRO and Australian National University and University of Adelaide<br>

```
@inproceedings{han2022machinemem,
  title={What Images are More Memorable to Machines?},
  author={Junlin Han and Huangying Zhan and Jie Hong and Pengfei Fang and Hongdong Li and Lars Petersson and Ian Reid},
  booktitle={arXiv preprint arXiv:2211.07625},
  year={2022}
}
```
   
## Prerequisites

This repo aims is based on [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Following their instructions to install the environments and prepare the datasets.

[timm](https://github.com/rwightman/pytorch-image-models) is also required, simply run

```
pip install timm
```
## Images augmented with YOCO
For each quadruplet, we show the original input image, augmented image from image-level augmentation, and two images from different cut dimensions produced by YOCO.
<img src='imgs/visu.png' align="middle" width=800>


## Contact
junlin.han@data61.csiro.au or junlinhcv@gmail.com

If you tried YOCO in other tasks/datasets/augmentations, please feel free to let me know the results. They will be collected and presented in this repo, regardless of positive or negative. Many thanks!

## Acknowledgments
Our code is developed based on [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet) and [MoCo](https://github.com/facebookresearch/moco). We thank anonymous reviewers for their invaluable feedback!
