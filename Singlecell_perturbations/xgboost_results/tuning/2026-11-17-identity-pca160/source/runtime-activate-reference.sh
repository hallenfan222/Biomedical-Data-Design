# Source this file inside an allocated compute session.
module purge
module load gpu/0.15.4
module load cuda12.2/toolkit/12.2.2
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PATH=/expanse/lustre/projects/jhu159/zfan11/bdd-runtime/env/bin:$PATH
export CUPY_CACHE_DIR=/expanse/lustre/projects/jhu159/zfan11/bdd-runtime/cupy-cache
