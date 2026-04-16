#!/bin/bash
#SBATCH --job-name=cifar10_fabric_demo
#SBATCH --output=outputs/cifar10_fabric_%j.out
#SBATCH --error=outputs/cifar10_fabric_%j.err
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2             
#SBATCH --gpus=2              
#SBATCH --cpus-per-task=8               
#SBATCH --time=00:30:00
#SBATCH --account=bccu-delta-gpu        # ← REPLACE with your Delta project
#SBATCH --constraint=scratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soham@illinois.edu  # <- change this


export LOGLEVEL=INFO

ROOT="$(dirname $PWD)"
mkdir -p outputs $ROOT/src/logs $ROOT/src/checkpoints

module load pytorch-conda
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load aws-ofi-nccl # loads the nccl built with the AWS nccl plugin for Slingshot11

# Print job info
echo "=========================================="
echo "Job started: $(date)"
echo "Node: -$(hostname)"
echo "GPUs available: $SLURM_GPUS_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Modules loaded:"
echo "$(module list)"
echo "=========================================="

conda activate base

time srun fabric run \
     --accelerator="gpu" \
     --devices=$SLURM_NTASKS_PER_NODE \
     --precision="bf16-mixed" \
     $ROOT/src/cifar10_fabric.py \
     --checkpoint-dir "$ROOT/src/checkpoints" \
     --log-dir "$ROOT/src/logs" \
     --data-dir "$ROOT/src/data"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
