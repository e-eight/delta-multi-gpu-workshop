#!/bin/bash
#SBATCH --job-name=cifar10_ddp_demo
#SBATCH --output=outputs/cifar10_ddp_%j.out
#SBATCH --error=outputs/cifar10_ddp_%j.err
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             
#SBATCH --gpus-per-node=2              
#SBATCH --cpus-per-task=8               
#SBATCH --time=00:30:00
#SBATCH --account=bccu-delta-gpu        # ← REPLACE with your Delta project
#SBATCH --constraint=scratch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=soham@illinois.edu  # <- change this


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

export LOGLEVEL=INFO

ROOT="$(dirname $PWD)" 
mkdir -p outputs $ROOT/src/logs $ROOT/src/checkpoints

module load pytorch-conda
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn # particular to Delta
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

srun torchrun \
     --nnodes=$SLURM_NNODES \
     --nproc_per_node=$SLURM_GPUS_PER_NODE \
     --rdzv_id=$RANDOM \
     --rdzv_backend=c10d \
     --rdzv_endpoint="$head_node_ip:29500" \
     $ROOT/src/cifar10_ddp.py \
     --checkpoint-dir "$ROOT/src/checkpoints" \
     --log-dir "$ROOT/src/logs" \
     --data-dir "$ROOT/src/data"
     
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
