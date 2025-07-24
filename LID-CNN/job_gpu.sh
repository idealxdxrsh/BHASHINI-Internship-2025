#!/bin/bash
#SBATCH --job-name="lid_gpu_train"      # Name of the job
#SBATCH -p gpu_h100_4                   # Partition to submit to. Use a GPU partition.
#SBATCH --gres=gpu:1                    # Request 1 GPU resource
#SBATCH -N 1                            # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks (usually 1 for a single Python script)
#SBATCH --cpus-per-task=8               # Number of CPUs for the task
#SBATCH --mem=32G                       # Memory per node
#SBATCH -t 0-08:00:00                   # Runtime in D-HH:MM:SS
#SBATCH -o slurm.%j.out                 # File to which STDOUT will be written
#SBATCH -e slurm.%j.err                 # File to which STDERR will be written
#SBATCH --mail-user=f20231167@goa.bits-pilani.ac.in # Your email address
#SBATCH --mail-type=ALL                 # Receive all email notification

# -- Environment Setup --
echo "Purging existing modules..."
module purge

# Load Anaconda and the NVIDIA HPC SDK for CUDA libraries
# Use 'module avail' or 'spack find' to get the exact versions on Sharanga
echo "Loading required modules..."
module load anaconda3-2022.05-gcc-11.2.0-od5lltp  # Replace with actual version
module load nvhpc/22.3       # This provides CUDA toolkit libraries

# Activate the conda environment you created in Step 2
echo "Activating Conda environment..."
source activate lid_env

# -- Execute the training script --
echo "Starting Python training script..."
python train_lid_gpu.py
echo "Script finished."

# Deactivate the environment
conda deactivate
