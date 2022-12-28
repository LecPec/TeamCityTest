#!/bin/bash
#SBATCH -N 4
#SBATCH -p broadwell-suppz
#SBATCH --ntasks-per-node=5
#SBATCH --time=10:00
#SBATCH --job-name=2D_PIC
#SBATCH --comment=2D_PIC
mpirun -np 20 ./pic_2d_openmp