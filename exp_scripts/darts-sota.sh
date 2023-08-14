#!/bin/bash
#SBATCH --job-name=darts_search
#SBATCH --output=../logged_outputs/darts_search.%j.log
#SBATCH --partition gpu-a100
#SBATCH --ntasks-per-node=2
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

module load anaconda3
source /opt/packages/anaconda3/etc/profile.d/conda.sh

conda activate darts_pt

script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}

space=${space:-s5}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'gpu:' $gpu

cd ../sota/cnn
python train_search.py \
    --method darts \
    --search_space $space --dataset $dataset \
    --seed $seed --save $id --gpu $gpu \
    # --expid_tag debug --fast \
wait
conda deactivate
## bash darts-sota.sh