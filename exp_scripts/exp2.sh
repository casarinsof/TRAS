#!/bin/bash
#SBATCH --job-name=C100_search
#SBATCH --output=../logged_outputs/C100_search.%j.log
#SBATCH --partition gpu-low
#SBATCH --ntasks-per-node=8
#SBATCH --nodes 1
#SBATCH --gres=gpu:a100_3g.39gb
#SBATCH --mem=32G
#SBATCH --time=12-12:00:00

module load python/3.9.15-aocc-3.2.0-linux-ubuntu22.04-zen2
module load python/py-pip-22.2.2-gcc-12.1.0-linux-ubuntu22.04-zen2

source /data/vision_group/venvs/nas/bin/activate

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
    --epochs 50 --save 'ALL_OP'
    # --expid_tag debug --fast \
wait
conda deactivate
## bash darts-sota.sh