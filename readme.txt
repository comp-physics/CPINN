The system environment requirement is in requirements.txt
(You might need to change "python3" to "python", depending on your system)
In the same directory as this readme is in, use the following commands to run the experiments for Poisson equation:

"python3 Poisson\train_1Dis_CGD.py"
"python3 Poisson\train_1Dis_ExtraAdam.py"
"python3 Poisson\train_1Dis_ExtraSGD.py"
"python3 Poisson\train_1Dis_GACGD.py"
"python3 Poisson\train_Adam.py -rng 12"
"python3 Poisson\train_SGD.py"


use the following commands to run the experiments for the Schrodinger's equation:

"python3 Schrodinger/train_Schrodinger_1_Dis_CPINN_GACGD.py -lrmin 0.001 -lrmax 0.001 -pinn 4 100 -dis 4 200"
"python3 Schrodinger/train_Schrodinger_Adam.py -lr 0.0001 -pinn 4 100"

use the following commands to run the experiments for the Burgers' equation:

"python3 Burger/BurgerTrainGACGD.py -lrmin 0.001 -lrmax 0.001 -disLayer 8 -disNeu 60 -pinnLayer 8 -pinnNeu 60"
"python3 Burger/BurgerTrainAdam.py -lr 0.001 -pinnLayer 8 -pinnNeu 60"

use the following commands to run the experiments for the Allen-Cahn equation:

"python3 AC/TrainAC_1Dis_GACGD_adaptive.py -PINN_neurons 128 -dis_neurons 256"
"python3 AC/TrainAC_Adam_Adaptive.py -adaptiveTol 0.0000001"