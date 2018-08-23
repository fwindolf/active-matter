#!/bin/sh
#SBATCH --time=48:00:00      # 2d
#SBATCH --mem=16G  	         # 16G of memory
#SBATCH --gres=gpu
#SBATCH --constraint=pascal

module purge
module load CUDA cuDNN
module load anaconda3/5.1.0-gpu

source activate $WRKDIR/conda/envs/am

which python

conda list | grep -E "keras|tensorflow|opencv"

# Copy data file to temp disk and unpack
mkdir /tmp/$SLURM_JOB_ID
trap "rm -rf /tmp/$SLURM_JOB_ID; exit" TERM EXIT # Set trap for unscheduled exit

cp $WRKDIR/data.tar /tmp/$SLURM_JOB_ID
cd /tmp/$SLURM_JOB_ID
tar xf data.tar 

# Go to work dir and start training
cd $WRKDIR/am2018

# Parameters for training
datadir=/tmp/$SLURM_JOB_ID

# Parameters for training
m=lstm_shallow
dp="$datadir/data/2700/tau_20_var_150 $datadir/data/2700/tau_20_var_200 $datadir/data/2700/tau_50_var_120 $datadir/data/2700/tau_50_var_150 $datadir/data/1200/tau_20_var_120 $datadir/data/1200/tau_20_var_180 $datadir/data/1200/tau_50_var_100 $datadir/data/1200/tau_50_var_150"
l=true
s=sequence
dh=448
dw=448
dn=1
dz=4
da=0.8
tr=0.1
tb=16
te=1
ts=0.1
tc=0
tl=dice

echo "--------------------------------------------------------"
echo "PARAMETERS"
echo "MODEL     : $m"
echo "LABELED   : $l"
echo "DATA_PATHS: $dp"
echo "STRUCTURE : $s"
echo "INPUT_DIMS: ($dh, $dw, $dn) with stacksize $dz"
echo "LR        : $tr"
echo "BATCHSIZE : $tb"
echo "EPOCHS    : $te"
echo "SPLIT     : $ts"
echo "CROPS     : $tc with area $da"
echo "LOSS      : $tl"
echo "--------------------------------------------------------"

# Start training
if [ "$l" = true ] ; then
    python train.py -m $m -l -dp $dp -s $s -dh $dh -dw $dw -dn $dn -dz $dz -da $da -tr $tr -tb $tb -te $te -ts $ts -tc $tc -tl $tl
else
    python train.py -m $m -dp $dp -s $s -dh $dh -dw $dw -dn $dn -dz $dz -da $da -tr $tr -tb $tb -te $te -ts $ts -tc $tc -tl $tl
fi

mv /tmp/$SLURM_JOB_ID/output $WRKDIR/am2018