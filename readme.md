## CPINN

<p align="center"> 
<a href="https://lbesson.mit-license.org/">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" />
</a>
<a href="http://doi.org/10.48550/arXiv.2204.11144">
  <img src="http://img.shields.io/badge/DOI-10.48550/arXiv.2204.11144-B31B1B.svg" />
</a>
</p>

This repository contains reproducers for the ICLR 2023 paper [Competitive Physics Informed Neural Networks](https://arxiv.org/abs/2204.11144) by Zeng, Kothari, Bryngelson, and Schäfer.

The paper can be cited as
```bibtex
@inproceedings{zeng23,
  author = {Zeng, Q. and Kothari, Y. and Bryngelson, S. H. and Sch{\"a}fer, F.},
  title = {Competitive physics informed networks},
  booktitle = {International Conference on Learning Representations (ICLR)},
  note = {arXiv:2204.11144},
  address = {Kigali, Rwanda},
  year = {2023},
  link = {https://openreview.net/pdf?id=z9SIj-IM7tn},
}
```

### What is CPINN? The abstract.

Neural networks can be trained to solve partial differential equations (PDEs) by using the PDE residual as the loss function. This strategy is called "physics-informed neural networks" (PINNs), but it currently cannot produce high-accuracy solutions, typically attaining about 0.1% relative error. We present an adversarial approach that overcomes this limitation, which we call competitive PINNs (CPINNs). CPINNs train a discriminator that is rewarded for predicting mistakes the PINN makes. The discriminator and PINN participate in a zero-sum game with the exact PDE solution as an optimal strategy. This approach avoids squaring the large condition numbers of PDE discretizations, which is the likely reason for failures of previous attempts to decrease PINN errors even on benign problems. Numerical experiments on a Poisson problem show that CPINNs achieve errors four orders of magnitude smaller than the best-performing PINN. We observe relative errors on the order of single-precision accuracy, consistently decreasing with each epoch. To the authors' knowledge, this is the first time this level of accuracy and convergence behavior has been achieved. Additional experiments on the nonlinear Schrödinger, Burgers', and Allen-Cahn equation show that the benefits of CPINNs are not limited to linear problems.

### Other code used in our experiments

In addition to the PyTorch optimizers, we use [ExtraGradient](https://github.com/facebookresearch/GAN-optimization-landscape/blob/main/lib/optim/extragradient.py) and [Competitive Gradient Descent](https://github.com/devzhk/cgds-package) optimizers.
The Weak Adversarial Network cases follow from the [Weak Adversarial Networks repo.](https://github.com/yaohua32/wan)
The training and testing data are obtained via the code in the [original PINNs repo.](https://github.com/maziarraissi/PINNs)

### Running the experiments

The system environment requirement is in `requirements.txt`. 

#### Poisson equation experiments 

* `python Poisson/train_1Dis_CGD.py` for the CGD training (CPINN)
* `python Poisson/train_1Dis_ExtraAdam.py` for the ExtraAdam training (CPINN)
* `python Poisson/train_1Dis_ExtraSGD.py` for the ExtraSGD training (CPINN)
* `python Poisson/train_1Dis_GACGD.py` for the GMRES-based ACGD training (CPINN)
* `python Poisson/train_Adam.py -rng 12` for the Adam training (PINN)
* `python Poisson/train_SGD.py` for the SGD training (PINN)
* `python WAN/train_WAN_copy_comb_activation_Adam_resample_no_log.py` for the WAN training with Adam/AdaGrad (WAN)
* `python WAN/train_WAN_2D_Poisson_comb_activation_algo1_no_log.py` for the WAN training with ACGD (WAN)

#### Schrodinger equation experiments 

* `python Schrodinger/train_Schrodinger_1_Dis_CPINN_GACGD.py -lrmin 0.001 -lrmax 0.001 -pinn 4 100 -dis 4 200` for the GMRES-based ACGD training (CPINN)
* `python3 Schrodinger/train_Schrodinger_Adam.py -lr 0.0001 -pinn 4 100` for the Adam training (PINN)

#### Burgers' equation experiments 

* `python Burger/BurgerTrainGACGD.py -lrmin 0.001 -lrmax 0.001 -disLayer 8 -disNeu 60 -pinnLayer 8 -pinnNeu 60`for GMRES-based ACGD training (CPINN)
* `python Burger/BurgerTrainAdam.py -lr 0.001 -pinnLayer 8 -pinnNeu 60` (PINN)

#### Allen-Cahn equation experiments 

* `python AC/TrainAC_1Dis_GACGD_adaptive.py -PINN_neurons 128 -dis_neurons 256` (CPINN)
* `python AC/TrainAC_Adam_Adaptive.py -adaptiveTol 0.0000001` (PINN)
