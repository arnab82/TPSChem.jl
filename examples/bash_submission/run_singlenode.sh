#!/bin/bash
#SBATCH -J single_node_cepa
#SBATCH -p general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abachhar@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=95:00:00
#SBATCH --mem=220G
#SBATCH -A r01859

export NTHREAD=64
sleep 10
hostname
module reset
export PATH=$HOME/.juliaup/bin:$PATH
export JULIA_DEPOT_PATH="/N/u/abachhar/BigRed200/.julia"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── CEPA pathway (full reference: examples/CEPA_JOB_KEYWORDS.md) ─────────────
# All of these are defaults only; set any of them before sbatch to override.
#
#   TPSCHEM_BUILD_HQQ   how H is applied on the Q space. Storing it is worth a
#               lot -- roughly 300x faster per apply -- and the solve costs under
#               100 MB on top, so budget ~80% of the node (176 GB) for the matrix.
#               Whether one fits at dim_q ~ 181000 depends on the fill, which is
#               worth measuring (one coarse-threshold run prints it).
#       packed  lower triangle only, dim_q(dim_q+1)/2 * 8 B = 131 GB here. Fits,
#               fill-independent, and measured both the smallest and the fastest
#               of the stored containers. The default.
#       sparse  CSC, dim_q^2 * fill * 16 B = 179 GB at 35% fill. Only smaller than
#               packed below 25% fill, and its COO build peaks at ~24 B/entry
#               before compressing, so it can die in the builder even when the
#               finished matrix would have fitted.
#       direct  dense, dim_q^2 * 8 B = 263 GB. Stores both triangles of a
#               symmetric matrix, so packed supersedes it. Does not fit here.
#       matvec  nothing stored, entries recomputed per apply, memory
#               O(nthreads * nroots * dim_q) ~ 0.5 GB. Always fits, and roughly
#               two orders of magnitude slower per apply. The fallback.
#       fois    H applied through the full FOIS matvec. Cheapest in memory and
#               by far the slowest -- roughly 12x slower per apply than matvec.
#   TPSCHEM_SOLVER      pcg | minres | krylov.
#       pcg     Jacobi-preconditioned CG; falls back to MINRES automatically on
#               any solve where the shifted operator is not positive definite.
#       minres  no definiteness assumption; safe for excited roots.
#       krylov  legacy KrylovKit path, single right-hand side, cannot block.
#   TPSCHEM_BLOCK_ROOTS true solves all roots in one pass sharing one H apply
#               (~3x on matvec/fois); false does one pass per root.
#   TPSCHEM_MINRES_MAXITER  Krylov iterations per solve. At matvec size an
#               iteration costs minutes, so this is a real wall-clock cap.
export TPSCHEM_BUILD_HQQ="${TPSCHEM_BUILD_HQQ:-packed}"
export TPSCHEM_SOLVER="${TPSCHEM_SOLVER:-pcg}"
export TPSCHEM_BLOCK_ROOTS="${TPSCHEM_BLOCK_ROOTS:-true}"
export TPSCHEM_MINRES_MAXITER="${TPSCHEM_MINRES_MAXITER:-300}"
export TPSCHEM_CEPA_SHIFT="${TPSCHEM_CEPA_SHIFT:-acpf}"
export TPSCHEM_CEPA_MIT="${TPSCHEM_CEPA_MIT:-30}"
export TPSCHEM_BLAS_THREADS="${TPSCHEM_BLAS_THREADS:-1}"

echo "CEPA pathway:   solver=$TPSCHEM_SOLVER  storage=$TPSCHEM_BUILD_HQQ  block_roots=$TPSCHEM_BLOCK_ROOTS"
echo "                shift=$TPSCHEM_CEPA_SHIFT  cepa_mit=$TPSCHEM_CEPA_MIT  cg_maxiter=$TPSCHEM_MINRES_MAXITER"
echo "                julia threads=$NTHREAD  blas threads=$TPSCHEM_BLAS_THREADS"

export INFILE=$1
export data_file1=$2
export data_file2=$3
export OUTFILE="${INFILE}.out"
export WORKDIR=$(pwd)

# FIX 1: define TMPDIR explicitly
export TMPDIR="/N/scratch/abachhar/${SLURM_JOB_ID}"

# FIX 2: mkdir with space
mkdir -p "$TMPDIR"

echo "TMPDIR=$TMPDIR"
echo "INFILE=$INFILE"
echo "OUTFILE=$OUTFILE"
echo "WORKDIR=$WORKDIR"

cp $INFILE $TMPDIR/
cp $data_file1 $TMPDIR/
cp $data_file2 $TMPDIR/
[ -n "$3" ] && cp $3 $TMPDIR/
[ -n "$4" ] && cp $4 $TMPDIR/
[ -n "$5" ] && cp $5 $TMPDIR/
[ -n "$6" ] && cp $3 $TMPDIR/
[ -n "$7" ] && cp $4 $TMPDIR/
[ -n "$8" ] && cp $5 $TMPDIR/
cd $TMPDIR

touch $OUTFILE
while true; do
    rsync -av $OUTFILE $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.out"
    sleep 60
done &
RSYNC_PID=$!

export JULIAENV="/N/u/abachhar/BigRed200/multinode-tpsci"

# instantiate only if Manifest is missing or stale
julia --project=$JULIAENV -e '
    import Pkg
    Pkg.instantiate()
    Pkg.precompile()
    println("Environment ready")
'

julia --project=$JULIAENV -t $NTHREAD $INFILE $data_file1 $data_file2 >& $OUTFILE

# cleanup
kill $RSYNC_PID
cp $OUTFILE $WORKDIR/"${INFILE}.out"
rm $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.out"
mkdir -p $WORKDIR/"${INFILE}.scr"
cp -r * $WORKDIR/"${INFILE}.scr"
cd $WORKDIR
rm -rf $TMPDIR
exit