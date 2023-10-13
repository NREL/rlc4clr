#!/bin/bash
#SBATCH --account=rlc4clr
#SBATCH --time=4:00:00
#SBATCH --job-name=es_stg1
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1


module purge
module load anaconda3
conda activate /projects/drl4dsr/xzhang2/conda_envs/rlc4clr


#worker_num=2 # Must be one less that the total number of nodes
worker_num=$(( $SLURM_JOB_NUM_NODES - 1 ))
total_cpus=$(( $SLURM_JOB_NUM_NODES * $SLURM_CPUS_ON_NODE - 1 ))
echo "worker_num="$worker_num
echo "total_cpus="$total_cpus

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

echo "nodes: "$nodes
echo "nodes_array: "$nodes_array

node1=${nodes_array[0]}
echo "node1: "$node1

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
echo "ip_prefix: "$ip_prefix
echo "suffix: "$suffix
echo "ip_head: "$ip_head
echo "redis_password: "$redis_password

# export ip_head # Exporting for latter access by trainer.py

echo "starting head"
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password --temp-dir="/tmp/scratch/ray"& # Starting the head
sleep 30

echo "starting workers"
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=${node2}"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password --temp-dir="/tmp/scratch/ray"& # Starting the workers
  sleep 5
done

sleep 20

echo "Start training"

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

python -u train_stg1.py --redis-password $redis_password --worker-num $total_cpus --ip-head $ip_head
