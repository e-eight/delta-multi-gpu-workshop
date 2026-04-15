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
#SBATCH --account=xxxx-delta-gpu        # ← REPLACE with your Delta project
#SBATCH --constraint=scratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=xxxxx@xxxxxxx.xxx  # <- change this


export LOGLEVEL=INFO

mkdir -p outputs logs checkpoints

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

ROOT="$(dirname $PWD)"
time srun fabric run $ROOT/src/cifar10_fabric.py \
     --accelerator="gpu" \
     --devices=$SLURM_NTASKS_PER_NODE \
     --precision="bf16-mixed"

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
