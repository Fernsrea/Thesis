#!/bin/bash
#SBATCH --job-name=build_lightningdit
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --nodes=1

CONTAINER_IMAGE=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.08-py3.sqsh
CONTAINER_SAVE=/netscratch/$USER/my_lightningdit_skin_image.sqsh
WORKDIR=$(pwd)
MOUNT_DIRS=/netscratch/$USER:/netscratch/$USER,$WORKDIR:$WORKDIR
ENV_YAML=/netscratch/$USER/lightningditskin_env.yaml

srun \
  --container-image=$CONTAINER_IMAGE \
  --container-save=$CONTAINER_SAVE \
  --container-workdir=$WORKDIR \
  --container-mounts=$MOUNT_DIRS \
  /bin/bash -c "
    set -e
    echo '[SETUP] Initializing conda...';
    source /opt/conda/etc/profile.d/conda.sh;
    conda activate base || true;
    conda update -n base conda -y || true;

    echo '[SETUP] Creating conda environment...';
    conda env create -f $ENV_YAML || true;

    echo '[SETUP] Verifying environment...';
    conda run -n lightningditskin echo '✅ Environment ready inside container';

    echo '[SETUP] Cleaning up image...';
    conda clean --all -y;
    rm -rf /var/lib/apt/lists/* || true;

    echo '[SETUP] Image setup complete.';
  "
