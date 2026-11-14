#!/bin/sh
#BSUB -J seg_02516
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

mkdir -p logs

module load python3/3.11.9
unset PYTHONPATH
source ~/venv_02516/bin/activate

python3 train.py --dataset ph2 --model encdec --loss bce --epochs 50 --batch-size 8 --image-size 256
python3 train.py --dataset drive --model encdec --loss bce --epochs 100 --batch-size 2 --image-size 512
