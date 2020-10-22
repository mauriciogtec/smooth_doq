#!/bin/bash
#SBATCH -A A-ib1
#SBATCH -p rtx
#SBATCH -t 4:0:0
#SBATCH -N 3
#SBATCH -n 12

module load launcher_gpu
module load intel/18

cd $SCRATCH/smooth_doq/

source $HOME/.bashrc
conda activate ./sdoq

export LAUNCHER_WORKDIR=`pwd`
export LAUNCHER_JOB_FILE=$LAUNCHER_WORKDIR/tacc/gan_jobs3.txt
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun