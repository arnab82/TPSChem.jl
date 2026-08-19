# =============================================================================
# SPT sharding scaling benchmark.
#
# Times the never-gather SPT primitives across the workers SLURM gave you, and
# checks each against the single-node result so a fast-but-wrong path cannot
# pass quietly (that has happened here: the sharded matvec was once 2.7x SLOWER
# than single-node and nothing noticed, because the tests compared numbers and
# never time).
#
# Measures:
#   build_sigma_sharded        destination partition,  O(P |state|) traffic
#   build_sigma_sharded_2d     2D worker grid,         O(sqrt(P) |state|)
#   sharded FOIS               never-gather first-order space
#   sharded PT1                partition = :dest and :grid
#
# Run the diagnostics driver first -- it tells you whether the grid is expected
# to pay off for your coupling graph before you spend node-hours here.
#
# Launch:
#   export TPSCHEM_INPUT_JLD2=/path/to/problem.jld2
#   sbatch examples/multinode/spt/run_spt_scaling.slurm
# or locally:
#   julia -p 4 --project examples/multinode/spt/spt_scaling_driver.jl problem.jld2
# =============================================================================

using Distributed, LinearAlgebra, Printf, JLD2
@everywhere using TPSChem
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "spt_common.jl"))

const SKIP_SINGLE = env_bool("TPSCHEM_SKIP_SINGLE_NODE", false)
const BLAS_THREADS = env_int("TPSCHEM_BLAS_THREADS", 1)
const NBODY = env_int("TPSCHEM_NBODY", 4)
const FOIS_THRESH = env_float("TPSCHEM_FOIS_THRESH", 1e-4)

pids = init_multinode_workers!()
data = load_problem_data()
ch, co = spt_operators(data)
v = spt_reference_state(data)
BLAS.set_num_threads(BLAS_THREADS)

@printf("\n%s\n", "="^78)
@printf(" SPT sharding benchmark: %d workers, state dim=%d, %d Fock sectors, %.1f MB\n",
        length(pids), length(v), length(v.data), state_mb(v))
@printf("%s\n", "="^78)

# ---- single-node reference (skip with TPSCHEM_SKIP_SINGLE_NODE=1 if it does
#      not fit on the master) ----------------------------------------------
t_single = NaN; ref_norm = NaN
if !SKIP_SINGLE
    TPSChem.cache_hamiltonian(v, v, co, ch)
    sig = deepcopy(v); TPSChem.zero!(sig)
    TPSChem.build_sigma!(sig, v, co, ch; cache=true, verbose=0)   # warm
    TPSChem.zero!(sig)
    global t_single = @elapsed TPSChem.build_sigma!(sig, v, co, ch; cache=true, verbose=0)
    global ref_norm = sum(TPSChem.orth_dot(sig, sig))
    TPSChem.flush_cache(ch)
    @printf("\n%-34s %8.2f s   |sig|=%.10f\n", "single-node build_sigma!", t_single, ref_norm)
end

dv = TPSChem.distribute_spt_state(deepcopy(v); workers=pids, strategy=:hash)
results = Tuple{String,Float64,Float64}[]

for (label, fn) in (("sharded matvec (destination)", TPSChem.build_sigma_sharded),
                    ("sharded matvec (2D grid)",     TPSChem.build_sigma_sharded_2d))
    TPSChem._tpsci_sharded_cache_operator_problem!(co, ch; workers=pids,
                                                   blas_threads=BLAS_THREADS)
    d = fn(dv, co, ch; nbody=NBODY, workers=pids, blas_threads=BLAS_THREADS)
    TPSChem.destroy!(d)                                           # warm
    t = @elapsed d = fn(dv, co, ch; nbody=NBODY, workers=pids, blas_threads=BLAS_THREADS)
    n = sum(LinearAlgebra.diag(TPSChem.overlap(d, d)))
    TPSChem.destroy!(d)
    push!(results, (label, t, n))
    @printf("%-34s %8.2f s   |sig|=%.10f%s\n", label, t, n,
            isnan(ref_norm) ? "" : @sprintf("   rel=%.2e", abs(n-ref_norm)/abs(ref_norm)))
    flush(stdout)
end

if length(results) == 2
    @printf("%-34s %8.2fx\n", "  -> 2D vs destination", results[1][2]/results[2][2])
    isnan(t_single) || @printf("%-34s %8.2fx  (%d workers)\n",
                               "  -> 2D vs single node", t_single/results[2][2], length(pids))
end

# ---- FOIS -------------------------------------------------------------------
t_fois_single = NaN
if !SKIP_SINGLE
    TPSChem.build_compressed_1st_order_state(v, co, ch; nbody=NBODY, thresh=FOIS_THRESH)
    global t_fois_single = @elapsed f1 = TPSChem.build_compressed_1st_order_state(
        v, co, ch; nbody=NBODY, thresh=FOIS_THRESH)
    @printf("\n%-34s %8.2f s   dim=%d\n", "single-node FOIS", t_fois_single, length(f1))
end
# :hash + existing_owners keeps the FOIS ownership consistent with the
# reference, which compute_pt1_wavefunction_sharded requires.
fois_kw = (nbody=NBODY, thresh=FOIS_THRESH, workers=pids, blas_threads=BLAS_THREADS,
           strategy=:hash, existing_owners=dv.owners)
df = TPSChem.build_compressed_1st_order_state_sharded(dv, co, ch; fois_kw...)
TPSChem.destroy!(df)
t_fois = @elapsed df = TPSChem.build_compressed_1st_order_state_sharded(dv, co, ch; fois_kw...)
@printf("%-34s %8.2f s   dim=%d%s\n", "sharded FOIS", t_fois, length(df),
        isnan(t_fois_single) ? "" : @sprintf("   %.2fx", t_fois_single/t_fois))

# ---- PT1 --------------------------------------------------------------------
for part in (:dest, :grid)
    p1, e2, ec = TPSChem.compute_pt1_wavefunction_sharded(df, dv, co, ch;
                    H0="Hcmf", workers=pids, blas_threads=BLAS_THREADS,
                    partition=part, verbose=0)
    TPSChem.destroy!(p1)                                           # warm
    t = @elapsed begin
        p1b, e2b, ecb = TPSChem.compute_pt1_wavefunction_sharded(df, dv, co, ch;
                    H0="Hcmf", workers=pids, blas_threads=BLAS_THREADS,
                    partition=part, verbose=0)
        TPSChem.destroy!(p1b)
    end
    @printf("%-34s %8.2f s   ecorr[1]=%.12f\n", "sharded PT1 ($part)", t, ec[1])
    flush(stdout)
end

TPSChem.destroy!(df); TPSChem.destroy!(dv)
@printf("\n%s\ndone\n", "="^78)
