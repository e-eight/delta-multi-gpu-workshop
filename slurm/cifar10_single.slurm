#!/bin/bash
#SBATCH --job-name=cifar10_single_demo
#SBATCH --output=outputs/cifar10_single_%j.out
#SBATCH --error=outputs/cifar10_single_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1               
#SBATCH --cpus-per-task=8               
#SBATCH --time=00:30:00
#SBATCH --account=xxxx-delta-gpu        # ← REPLACE with your Delta project
#SBATCH --constraint=scratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xxxxx@xxxxxxxx.xxx  # <- change this

mkdir -p outputs

module load pytorch-conda
conda activate base

# Print job info
echo "=========================================="
echo "Job started: $(date)"
echo "Node: -$(hostname)"
echo "GPUs available: $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Modules loaded:"
echo "$(module list)"
echo "=========================================="

ROOT="$(dirname $PWD)"
time srun python $ROOT/src/cifar10_single.py 

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
