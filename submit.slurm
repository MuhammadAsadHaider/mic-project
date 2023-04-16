```
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=37:00:00
#SBATCH --job-name=DDGAN_SWINUNETR
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user haider24@knights.ucf.edu
#SBATCH --output=/home/cap6412.student10/output.out

# Load modules
module purge
#module load anaconda
module load cuda/cuda-11.6
module load gcc/gcc-9.1.0
module load oneapi/mkl

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO

cd /home/cap6412.student10/MIC-PROJECT
source venv/bin/activate

srun python train_ddgan.py --num_epoch 100 --num_process_per_node 2 --num_proc_node 2 --data_dir swin_unitr --json_list swin_unitr/brats21_folds.json
```