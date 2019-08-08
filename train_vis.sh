#!/usr/bin/env bash

sbatch --job-name=lmdis_vis_zebra \
       --output=results/vis_zebra_10/out.txt \
       --error=results/vis_zebra_10/err.txt \
       --nodes=1 \
       --gres=gpu:1 \
       --time=40:00:00 \
       --cpus-per-task 10 \
       --partition=learnfair \
       --wrap="srun python exp-ae-vis-10.py"