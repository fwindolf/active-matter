#!/bin/sh
#SBATCH --time=48:00:00      # 2d
#SBATCH --mem=16G  	     # 16G of memory
#SBATCH --gres=gpu
#SBATCH --constraint=pascal

module purge
module load CUDA cuDNN
module load anaconda3/5.1.0-gpu

source activate $WRKDIR/conda/envs/am

which python

conda list | grep -E "keras|tensorflow|opencv"

cd $WRKDIR/am2018

# Parameters for training
m=seg_net
dp="$WRKDIR/data/2700/tau_20_var_150 $WRKDIR/data/2700/tau_20_var_200  $WRKDIR/data/2700/tau_50_var_120 $WRKDIR/data/2700/tau_50_var_150 $WRKDIR/data/1200/tau_20_var_120 $WRKDIR/data/1200/tau_20_var_150 $WRKDIR/data/1200/tau_20_var_180 $WRKDIR/data/1200/tau_50_var_100  $WRKDIR/data/1200/tau_50_var_120 $WRKDIR/data/1200/tau_50_var_150"
s=stacked
dh=480
dw=480
dz=3
ds=0.4

tc=1
tr=0.1
tb=8
te=200
ts=0.15

echo "--------------------------------------------------------"
echo "PARAMETERS"
echo "MODEL     : $m"
echo "DATA_PATHS: $dp"
echo "STRUCTURE : $s"
echo "INPUT_DIMS: ($dh, $dw, $dz)"
echo "CROPS     : $tc with pre-scale of $ds"
echo "LR        : $tr"
echo "BATCHSIZE : $tb"
echo "EPOCHS    : $te"
echo "SPLIT     : $ts"
echo "--------------------------------------------------------"

# Start training
srun python train.py -m $m -dp $dp -l -s $s -dh $dh -dw $dw -dz $dz -ds $ds -tr $tr -tb $tb -te $te -ts $ts -tc $tc