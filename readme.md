# CPINN
## Overview
This repository contains code examples for the paper [Competitive Physics Informed Neural Networks](https://arxiv.org/abs/2204.11144) in Pytorch.

CPINN uses an adversarial architecture to train Physics Informed Neural Networks (PINNs) against a discriminator in a zero-sum minimax game to reach higher accuracy beyond PINNs' capability.

In addition to the PyTorch optimizers, we use [ExtraGradient](https://github.com/facebookresearch/GAN-optimization-landscape/blob/main/lib/optim/extragradient.py) and [Competitive Gradient Descent](https://github.com/devzhk/cgds-package) optimizers, both avaliable under MIT license.

We obtain the training and testing data from the original [PINNs](https://github.com/maziarraissi/PINNs) directory, avaliable under MIT license.

## Run The Code
The system environment requirement is in requirements.txt
(You might need to change "python3" to "python", depending on your system)
In the same directory as this readme is in, use the following commands to run the experiments for Poisson equation:

`python3 Poisson/train_1Dis_CGD.py` for the CGD training (CPINN)

`python3 Poisson/train_1Dis_ExtraAdam.py`for the ExtraAdam training (CPINN)

`python3 Poisson/train_1Dis_ExtraSGD.py`for the ExtraSGD training (CPINN)

`python3 Poisson/train_1Dis_GACGD.py`for the GMRES-based ACGD training (CPINN)

`python3 Poisson/train_Adam.py -rng 12`for the Adam training (PINN)

`python3 Poisson/train_SGD.py`for the SGD training (PINN)

`python3 WAN/train_WAN_copy_comb_activation_Adam_resample_no_log.py` for the WAN training with Adam/AdaGrad (WAN)

`python3 WAN/train_WAN_2D_Poisson_comb_activation_algo1_no_log.py` for the WAN training with ACGD (WAN)


use the following commands to run the experiments for the Schrodinger's equation:

`python3 Schrodinger/train_Schrodinger_1_Dis_CPINN_GACGD.py -lrmin 0.001 -lrmax 0.001 -pinn 4 100 -dis 4 200` for the GMRES-based ACGD training (CPINN)

`python3 Schrodinger/train_Schrodinger_Adam.py -lr 0.0001 -pinn 4 100` for the Adam training (PINN)

use the following commands to run the experiments for the Burgers' equation:

`python3 Burger/BurgerTrainGACGD.py -lrmin 0.001 -lrmax 0.001 -disLayer 8 -disNeu 60 -pinnLayer 8 -pinnNeu 60`for GMRES-based ACGD training (CPINN)

`python3 Burger/BurgerTrainAdam.py -lr 0.001 -pinnLayer 8 -pinnNeu 60`(PINN)

use the following commands to run the experiments for the Allen-Cahn equation:

`python3 AC/TrainAC_1Dis_GACGD_adaptive.py -PINN_neurons 128 -dis_neurons 256`(CPINN)

`python3 AC/TrainAC_Adam_Adaptive.py -adaptiveTol 0.0000001` (PINN)
