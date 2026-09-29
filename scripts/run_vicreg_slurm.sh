#!/bin/bash
# Self-supervised VICReg pretraining of a ResNet-50 on unlabeled thermal images (SLURM + Singularity).
# Requires a clone of https://github.com/facebookresearch/vicreg in ./vicreg and the
# unlabeled images as .npy files. Adjust the paths below to your cluster.
#SBATCH --job-name="VICRegSSL"
#SBATCH -o logs/stdout.txt
#SBATCH -e logs/stderr.txt
#SBATCH --time=14-00:00:00
#SBATCH -p gpu

ENV_NAME="thermal-bc"
PWD_APP="/path/to/breast_cancer_detection"
IMAGE_PATH="/path/to/singularity_image.sif"

srun singularity exec --bind $PWD_APP:/app --nv $IMAGE_PATH \
bash -c "source /opt/conda/etc/profile.d/conda.sh && \
conda activate $ENV_NAME && \
python /app/vicreg/main_vicreg.py \
--data-dir /app/data/ssl_images/ssl_images_npy \
--exp-dir /app/vicreg/checkpoints \
--epochs 300 \
--batch-size 512 \
--base-lr 0.2 \
--mlp 2048-2048-2048 \
--sim-coeff 15.0 \
--std-coeff 25.0 \
--cov-coeff 10.0 \
--device cuda"
