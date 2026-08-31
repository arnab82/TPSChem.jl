#
# Multinode CEPA pathway benchmark: all roots in one solver pass vs one pass per
# root.
#
#   CEPA_MN_WORKERS=2 CEPA_MN_THREADS=4 julia --project=. \
#       examples/multinode/cepa_blocked_roots_bench_multinode.jl
#
# Two tables, mirroring examples/cepa_blocked_roots_bench.jl:
#   1. One sharded Hamiltonian apply, one root vs a block of R, for both storage
#      tiers. This isolates what blocking changes across nodes: the stored H, the
#      Fock-sector routing and the number of remote round trips are identical for
#      every root, so batching them costs one apply instead of R.
#   2. End-to-end `do_tps_sharded_cepa` per (tier, solver, block_roots), with wall
#      time and peak RSS on the master and on every worker.
#
# Each end-to-end configuration runs in a fresh master that spawns fresh workers,
# because `Sys.maxrss()` is a process high-water mark — measured in one process,
# an earlier configuration's peak would leak into every later one.
#
# Threads per worker are pinned (CEPA_MN_THREADS, default 4) and BLAS to 1, so the
# comparison is apples-to-apples.
#
using Distributed
using Printf

const NWORK   = parse(Int, get(ENV, "CEPA_MN_WORKERS", "2"))
const NTHREAD = parse(Int, get(ENV, "CEPA_MN_THREADS", "4"))
const THRESH  = parse(Float64, get(ENV, "CEPA_MN_THRESH", "1e-4"))
const NROOTS  = parse(Int, get(ENV, "CEPA_MN_NROOTS", "4"))
const MROOTS  = parse(Int, get(ENV, "CEPA_MN_M", "100"))
const DATA    = get(ENV, "CEPA_MN_DATA",
                    joinpath(@__DIR__, "..", "..", "test", "data_cmf_13_cr2_morokuma.jld2"))
const CHILD   = get(ENV, "CEPA_MN_CHILD", "")
const CEPA_MIT = parse(Int, get(ENV, "CEPA_MN_MIT", "30"))
# Comma-separated shift names to run, e.g. CEPA_MN_SHIFTS=acpf,aqcc to add rows to
# an already-measured table without repeating the expensive CEPA-0 ones.
const SHIFTS  = split(get(ENV, "CEPA_MN_SHIFTS", "cepa,acpf,aqcc"), ",")

if nprocs() == 1
    addprocs(NWORK; exeflags=["--project=$(Base.active_project())", "-t$NTHREAD"])
end

@everywhere using TPSChem
@everywhere using TPSChem.QCBase
@everywhere using TPSChem.RDM
@everywhere using LinearAlgebra
@everywhere BLAS.set_num_threads(1)
using JLD2

function setup()
    d = JLD2.load(DATA)
    ints, clusters, d1 = d["ints"], d["clusters"], d["d1"]
    ref_fock = FockConfig([(3,0),(3,3),(0,3)])
    clustered_ham = TPSChem.extract_ClusteredTerms(ints, clusters)
    cb = TPSChem.compute_cluster_eigenbasis_spin(ints, clusters, d1, [3,3,3], ref_fock,
                                                 max_roots=MROOTS, verbose=0)
    cluster_ops = TPSChem.compute_cluster_ops(cb, ints)
    TPSChem.add_cmf_operators!(cluster_ops, cb, ints, d1.a, d1.b)
    ci = TPSChem.TPSCIstate(clusters, ref_fock, R=NROOTS)
    ci = TPSChem.add_spin_focksectors(ci)
    return ci, cluster_ops, clustered_ham
end

worker_rss() = Dict(pid => remotecall_fetch(Sys.maxrss, pid) for pid in workers())

# ── Child mode: one end-to-end configuration ─────────────────────────────────
if !isempty(CHILD)
    tier, solver, block_roots, shift = split(CHILD, ",")
    ci, cluster_ops, clustered_ham = setup()
    # Solve the reference locally and hand over e0: the sharded Davidson cannot
    # start from the raw spin-Fock expansion (its initial guess is rank deficient),
    # and the reference solve is not what this benchmark is timing anyway.
    e0, ci = TPSChem.tps_ci_direct(ci, cluster_ops, clustered_ham, conv_thresh=1e-10)
    GC.gc()
    r = @timed TPSChem.do_tps_sharded_cepa(deepcopy(ci), cluster_ops, clustered_ham;
                                           cepa_shift=String(shift), cepa_mit=CEPA_MIT,
                                           nbody=4, e0=e0,
                                           thresh_foi=THRESH, tol=1e-8,
                                           thresh_sigma=0.0,
                                           h_storage=Symbol(tier),
                                           solver=Symbol(solver),
                                           block_roots=(block_roots == "true"),
                                           workers=workers(), verbose=0)
    e, q = r.value
    TPSChem.destroy!(q)
    wr = worker_rss()
    @printf("RESULT\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n",
            tier, solver, block_roots, shift, r.time, r.bytes/2^30,
            Sys.maxrss()/2^30, maximum(values(wr))/2^30,
            join((@sprintf("%.10f", x) for x in e), ","))
    exit(0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Parent
# ─────────────────────────────────────────────────────────────────────────────
println("="^100)
@printf(" Multinode CEPA blocked-roots benchmark — %s\n", basename(DATA))
@printf(" workers = %i   threads/worker = %i   BLAS = 1   R = %i   thresh_foi = %.0e\n",
        nworkers(), NTHREAD, NROOTS, THRESH)
println("="^100)

ci, cluster_ops, clustered_ham = setup()
e0, ref_vec = TPSChem.tps_ci_direct(deepcopy(ci), cluster_ops, clustered_ham, conv_thresh=1e-10)
dref = TPSChem.distribute_tpsci_state(deepcopy(ref_vec); workers=workers(), strategy=:balanced)
qspace = TPSChem.open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                     nbody=4, thresh=THRESH, prescreen=false,
                                     workers=workers(), verbose=0)
TPSChem.project_out!(qspace, dref)
sig = TPSChem.open_matvec_sharded(dref, cluster_ops, clustered_ham;
                                  nbody=4, thresh=0.0, prescreen=false,
                                  workers=workers(), verbose=0)
hblock = TPSChem.restrict_to_basis_sharded(sig, qspace)
TPSChem.destroy!(sig)
@printf("\n dim_q = %i\n", length(qspace))

hlocal = TPSChem.collect_tpsci_state(hblock)
hroots = []
for i in 1:NROOTS
    tmp = TPSChem.distribute_tpsci_state(TPSChem.extract_chosen_root(hlocal, i);
                                         workers=workers(), strategy=:balanced)
    push!(hroots, TPSChem.restrict_to_basis_sharded(tmp, qspace))
    TPSChem.destroy!(tmp)
end
eshifts = Float64[e0[i] for i in 1:NROOTS]

best(f, n) = minimum(begin GC.gc(); @elapsed f() end for _ in 1:n)

@printf("\n ┌ Table 1: one sharded H apply — single root vs block of %i ───────────────────────────┐\n", NROOTS)
@printf(" %-12s %12s %12s %10s %12s %12s\n",
        "tier", "1-root (s)", "block (s)", "speedup", "per-root (s)", "build (s)")
for tier in (:blocks, :matrixfree)
    local op, tbuild
    tbuild = @elapsed op = tier == :blocks ?
        TPSChem.build_block_h_sharded(qspace, cluster_ops, clustered_ham;
                                      workers=workers(), verbose=0) :
        TPSChem.MatrixFreeShardedH(cluster_ops, clustered_ham, workers(), true, 1)
    t1 = best(2) do
        for i in 1:NROOTS
            TPSChem.destroy!(TPSChem.apply_sharded_H(op, hroots[i]; eshift=eshifts[i]))
        end
    end
    tR = best(() -> TPSChem.destroy!(TPSChem.apply_sharded_H_block(op, hblock, eshifts)), 2)
    @printf(" %-12s %12.4f %12.4f %9.2fx %12.4f %12.2f\n",
            tier, t1/NROOTS, tR, t1/tR, tR/NROOTS, tbuild)
    flush(stdout)
    tier == :blocks && TPSChem.destroy!(op)
end
@printf(" └%s┘\n", "─"^86)
@printf(" 1-root column is the per-root cost; speedup = (%i sequential applies) / (one block apply)\n", NROOTS)

for h in hroots; TPSChem.destroy!(h); end
TPSChem.destroy!(hblock); TPSChem.destroy!(qspace); TPSChem.destroy!(dref)

# ── Table 2 ──────────────────────────────────────────────────────────────────
# CEPA-0 runs a single macro-iteration, so the solve -- the only part blocking
# touches -- is at its smallest relative to the one-time FOIS and H_qq build. ACPF
# iterates the shift, repeating the solve while the build stays one-time, so it is
# the case where blocking should show up even for the cheap stored-block apply.
configs = [("blocks",     "minres", "false", "cepa"), ("blocks",     "minres", "true", "cepa"),
           ("blocks",     "pcg",    "false", "cepa"), ("blocks",     "pcg",    "true", "cepa"),
           ("matrixfree", "minres", "false", "cepa"), ("matrixfree", "minres", "true", "cepa"),
           ("blocks",     "minres", "false", "acpf"), ("blocks",     "minres", "true", "acpf"),
           ("blocks",     "pcg",    "false", "acpf"), ("blocks",     "pcg",    "true", "acpf"),
           ("blocks",     "minres", "false", "aqcc"), ("blocks",     "minres", "true", "aqcc")]
configs = [c for c in configs if c[4] in SHIFTS]

@printf("\n ┌ Table 2: end to end (fresh master + workers each, cepa_mit=%i) ──────────────────────┐\n", CEPA_MIT)
@printf(" %-11s %-7s %-10s %-6s %10s %10s %11s %11s\n",
        "tier", "solver", "blocked", "shift", "time (s)", "alloc GiB", "master GiB", "worker GiB")
rows = Any[]
for (tier, slv, mr, sh) in configs
    cmd = `$(Base.julia_cmd()) --project=$(dirname(dirname(@__DIR__))) $(@__FILE__)`
    env = copy(ENV)
    env["CEPA_MN_CHILD"] = "$tier,$slv,$mr,$sh"
    env["JULIA_NUM_THREADS"] = "1"
    out = try
        read(setenv(cmd, env), String)
    catch err
        @printf(" %-11s %-7s %-10s %-6s   FAILED\n", tier, slv, mr, sh)
        continue
    end
    lines = split(out, '\n')
    idx = findfirst(l -> startswith(l, "RESULT"), lines)
    if idx === nothing
        @printf(" %-11s %-7s %-10s %-6s   NO RESULT\n", tier, slv, mr, sh)
        continue
    end
    f = split(lines[idx], '\t')
    @printf(" %-11s %-7s %-10s %-6s %10.2f %10.3f %11.3f %11.3f\n",
            f[2], f[3], f[4], f[5], parse(Float64,f[6]), parse(Float64,f[7]),
            parse(Float64,f[8]), parse(Float64,f[9]))
    push!(rows, (tier, slv, mr, sh, parse.(Float64, split(f[10], ','))))
    flush(stdout)
end
@printf(" └%s┘\n", "─"^86)

# Compare within a shift family: acpf and aqcc converge to different energies than
# cepa-0, so a single global reference would be meaningless.
for family in unique(r[4] for r in rows)
    fam = [r for r in rows if r[4] == family]
    isempty(fam) && continue
    ref = fam[1][5]
    @printf("\n Energy agreement, shift=%s (vs %s/%s/blocked=%s):\n",
            family, fam[1][1], fam[1][2], fam[1][3])
    for (tier, slv, mr, sh, e) in fam
        @printf("   %-11s %-7s blocked=%-6s  max|ΔE| = %.2e\n", tier, slv, mr,
                maximum(abs.(e .- ref)))
    end
end
println()
