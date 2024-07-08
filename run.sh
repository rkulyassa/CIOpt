#!/usr/bin/env bash
#SBATCH --job-name="ci-opt-test"
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH --account=tpl104
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --export=ALL

module reset
### ssl from centos7 container
export LD_LIBRARY_PATH=/home/vbhvsh0/software/OpenSSL/ssl-v10/lib:$LD_LIBRARY_PATH

### openmm
export LD_LIBRARY_PATH=/home/vbhvsh0/software/openmm/lib:$LD_LIBRARY_PATH

### protobuf compiled in centos7 env with gcc 4.8.5
export LD_LIBRARY_PATH=/home/vbhvsh0/software/protobuf/v3.14.0/lib:$LD_LIBRARY_PATH

### Modules from Expanse
module load cuda11.7/toolkit
module load intel/19.1.3.304/vecir2b 
module load intel-mpi/2019.10.317/uwgziob
module load intel-mkl/2020.4.304/2bp4pd3-omp

### Terachem Libs
export TeraChem=/home/vbhvsh0/Terachem//TeraChem
export NBOEXE=/home/vbhvsh0/Terachem//TeraChem/bin/nbo6.i4.exe
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/vbhvsh0/Terachem//TeraChem/lib:$LD_LIBRARY_PATH
export PATH=/home/vbhvsh0/Terachem//TeraChem/bin:$PATH

### PRELOAD MKL
export LD_PRELOAD=$INTEL_MKLHOME/mkl/lib/intel64/libmkl_rt.so

python ciopt.py