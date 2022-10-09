#!/bin/bash
#SBATCH -N 1
#SBATCH -p RT_study
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=2D_PIC
#SBATCH --comment=2D_PIC
mpirun -np 16 ./pic_2d_openmp