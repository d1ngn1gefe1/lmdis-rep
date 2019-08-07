#!/usr/bin/env bash

sbatch --job-name=lmdis_cat \
       --output=results/cat_10/out.txt \
       --error=results/cat_10/err.txt \
       --nodes=1 \
       --gres=gpu:1 \
       --time=40:00:00 \
       --cpus-per-task 10 \
       --partition=learnfair \
       --wrap="srun python exp-ae-cat-10.py"