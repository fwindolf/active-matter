source activate keras-tensorflow-1.6

datadir=/l/projects/data/AM2018_MIXED
#datadir=/l/projects/data/AM2018_SIMULATION/data

m=lstm_shallow
dt=mixed
#dt=text
dp="$datadir/0deg_-100V/ $datadir/0deg_-120V/ $datadir/30deg_-100V/ $datadir/60deg_-100V/ $datadir/60deg_-120V/ $datadir/60deg_-95V/"
#dp="$datadir/1200/tau_20_var_120/ $datadir/1200/tau_20_var_150/ $datadir/1200/tau_20_var_180/ $datadir/1200/tau_50_var_100/ $datadir/1200/tau_50_var_120/  $datadir/1200/tau_50_var_150/ $datadir/2700/tau_20_var_150/ $datadir/2700/tau_20_var_200/ $datadir/2700/tau_50_var_120/ $datadir/2700/tau_50_var_150/"
l=true
s=sequence
dh=448
dw=448
dn=1
dz=4
dc=1
da=0.8
tr=0.1
tb=2
te=30
ts=0.1
tc=0
tl=dice

echo "--------------------------------------------------------"
echo "PARAMETERS"
echo "MODEL     : $m"
echo "LABELED   : $l"
echo "DATA_TYPE : $dt"
echo "DATA_PATHS: $dp"
echo "STRUCTURE : $s"
echo "INPUT_DIMS: ($dh, $dw, $dn) with stacksize $dz"
echo "CLASSES   : $dc"
echo "LR        : $tr"
echo "BATCHSIZE : $tb"
echo "EPOCHS    : $te"
echo "SPLIT     : $ts"
echo "CROPS     : $tc with area $da"
echo "LOSS      : $tl"
echo "--------------------------------------------------------"

# Start training
if [ "$l" = true ] ; then
    python train.py -m $m -l -dt $dt -dp $dp -s $s -dh $dh -dw $dw -dn $dn -dz $dz -dc $dc -da $da -tr $tr -tb $tb -te $te -ts $ts -tc $tc -tl $tl
else
    python train.py -m $m -dt $dt -dp $dp -s $s -dh $dh -dw $dw -dn $dn -dz $dz -dc $dc -da $da -tr $tr -tb $tb -te $te -ts $ts -tc $tc -tl $tl
fi
