#!/bin/bash
#SBATCH --job-name="VICRegSSL"
#SBATCH -o /example/path/logs/stdout.txt
#SBATCH -e /example/path/logs/stderr.txt
#SBATCH --time=14-00:00:00
#SBATCH -p gpu
#SBATCH -w gnode01

ENV_NAME="name of your conda environment"
PWD_APP="/example/path/"
IMAGE_PATH="/home/example/path/singularity_img/bcxtt_env.sif"

srun singularity exec --bind $PWD_APP:/app --nv $IMAGE_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && \
conda activate $ENV_NAME && \
python /app/vicreg/main_vicreg.py \
--data-dir /app/data/ssl_images/ssl_images_npy \
--exp-dir /app/outputs_cluster1 \
--epochs 300 \
--batch-size 512 \
--base-lr 0.2 \
--mlp 2048-2048-2048 \
--sim-coeff 15.0 \
--std-coeff 25.0 \
--cov-coeff 10.0 \
--device cuda"
