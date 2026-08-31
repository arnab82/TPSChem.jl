#
# CEPA solver pathway benchmark: multiroot vs one-root-at-a-time, single node.
#
#   JULIA_NUM_THREADS=4 julia --project=. examples/cepa_multiroot_bench.jl
#
# Two tables:
#   1. Operator throughput — one H_qq apply, one root vs a block of R, for each
#      storage strategy. This isolates the thing blocking actually changes: the
#      work every root shares (streaming H_qq, recomputing its entries, screening
#      FOIS terms) is done once instead of R times.
#   2. End-to-end CEPA-0 — full `do_fois_cepa` for each (solver, storage,
#      multiroot) combination, with wall time, allocations and peak RSS.
#
# Every configuration in table 2 runs in a fresh subprocess, because `Sys.maxrss()`
# is a high-water mark for the whole process: measured in-process, one configuration's
# peak would leak into every later one.
#
# Thread count is pinned (CEPA_BENCH_THREADS, default 4) so the comparison is
# apples-to-apples; BLAS is pinned to 1 thread for the same reason.
#
using TPSChem
using TPSChem.QCBase
using TPSChem.RDM
using LinearAlgebra
using SparseArrays
using Printf
using JLD2

const DATA    = get(ENV, "CEPA_BENCH_DATA",   joinpath(@__DIR__, "..", "test", "data_cmf_13_cr2_morokuma.jld2"))
const THRESH  = parse(Float64, get(ENV, "CEPA_BENCH_THRESH", "1e-4"))
const NROOTS  = parse(Int,     get(ENV, "CEPA_BENCH_NROOTS", "4"))
const MROOTS  = parse(Int,     get(ENV, "CEPA_BENCH_M",      "100"))
const NTHREAD = parse(Int,     get(ENV, "CEPA_BENCH_THREADS","4"))
const CHILD   = get(ENV, "CEPA_BENCH_CHILD", "")
# A :fois solve applies H through the whole FOIS every iteration -- tens of seconds
# each here -- so an unbounded run is hours. Cap the budget: with the same cap on
# both sides, multiroot and one-at-a-time do exactly the same number of applies and
# the comparison measures precisely what blocking changes.
const MAXIT   = parse(Int, get(ENV, "CEPA_BENCH_MAXITER", "300"))
const FOISIT  = parse(Int, get(ENV, "CEPA_BENCH_FOIS_MAXITER", "6"))

BLAS.set_num_threads(1)

# ── System setup, shared by parent and children ──────────────────────────────
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

# Poll the live heap while `f` runs, so we see the peak and not just the endpoint.
function with_peak_heap(f)
    peak = Ref(Base.gc_live_bytes())
    done = Ref(false)
    task = @async while !done[]
        peak[] = max(peak[], Base.gc_live_bytes())
        sleep(0.05)
    end
    r = f()
    done[] = true
    wait(task)
    return r, peak[]
end

# ── Child mode: run one end-to-end configuration and report one line ─────────
if !isempty(CHILD)
    solver, storage, multiroot = split(CHILD, ",")
    ci, cluster_ops, clustered_ham = setup()
    GC.gc()
    rss0 = Sys.maxrss()
    (r, peak) = with_peak_heap() do
        @timed TPSChem.do_fois_cepa(deepcopy(ci), cluster_ops, clustered_ham;
                                    thresh_foi=THRESH, nbody=4, tol=1e-8,
                                    thresh_sigma=0.0, cg_maxiter=MAXIT,
                                    solver=Symbol(solver),
                                    build_hqq=Symbol(storage),
                                    multiroot=(multiroot == "true"),
                                    verbose=0)
    end
    e = r.value[1]
    @printf("RESULT\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n",
            solver, storage, multiroot, r.time, r.bytes/2^30,
            peak/2^30, max(Sys.maxrss(), rss0)/2^30,
            join((@sprintf("%.10f", x) for x in e), ","))
    exit(0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Parent
# ─────────────────────────────────────────────────────────────────────────────
@printf("\n")
println("═"^96)
@printf(" CEPA multiroot benchmark — %s\n", basename(DATA))
@printf(" julia threads = %i   BLAS threads = 1   R = %i   thresh_foi = %.0e   M = %i\n",
        Threads.nthreads(), NROOTS, THRESH, MROOTS)
println("═"^96)
Threads.nthreads() == NTHREAD ||
    @printf(" !! running on %i threads, expected %i — set JULIA_NUM_THREADS=%i\n",
            Threads.nthreads(), NTHREAD, NTHREAD)

ci, cluster_ops, clustered_ham = setup()
e0, ref_vec = TPSChem.tps_ci_direct(deepcopy(ci), cluster_ops, clustered_ham, conv_thresh=1e-10)
q = TPSChem.open_matvec_thread(deepcopy(ref_vec), cluster_ops, clustered_ham, nbody=4, thresh=THRESH)
TPSChem.project_out!(q, ci)
dim_q = length(q)
q1 = TPSCIstate(q, R=1)
@printf("\n dim_q = %i\n", dim_q)

# ── Table 1: operator throughput ─────────────────────────────────────────────
@printf("\n")
@printf(" ┌ Table 1: one H_qq apply — single root vs block of %i ─────────────────────────────────┐\n", NROOTS)
@printf(" %-10s %12s %12s %10s %12s %12s\n",
        "storage", "1-root (s)", "block (s)", "speedup", "per-root (s)", "build (s)")

x  = randn(dim_q)
X  = randn(NROOTS, dim_q)
X1 = reshape(x, 1, dim_q)
best(f, n) = minimum(begin GC.gc(); @elapsed f() end for _ in 1:n)

results1 = Any[]
for storage in (:sparse, :direct, :matvec, :fois)
    local op1, opR, tbuild
    tbuild = @elapsed begin
        if storage == :sparse
            A = TPSChem.build_H_qq_sparse(q1, cluster_ops, clustered_ham)
            op1 = TPSChem.ThreadedSymSpMV(A); opR = op1
            @printf(" %-10s nnz = %i (%.2f%% fill, %.2f GiB)\n", "sparse",
                    nnz(A), 100*nnz(A)/dim_q^2, (nnz(A)*16 + 8dim_q)/2^30)
        elseif storage == :direct
            A = TPSChem.build_H_qq(q1, cluster_ops, clustered_ham)
            op1 = TPSChem.ThreadedSymDenseMV(A); opR = op1
        elseif storage == :matvec
            op1 = TPSChem.StructuredHqq(q1, cluster_ops, clustered_ham; nroots=1)
            opR = TPSChem.StructuredHqq(q1, cluster_ops, clustered_ham; nroots=NROOTS)
        else
            op1 = TPSChem.FoisHqq(q, cluster_ops, clustered_ham; nbody=4, nroots=1)
            opR = TPSChem.FoisHqq(q, cluster_ops, clustered_ham; nbody=4, nroots=NROOTS)
        end
    end
    Y1 = zeros(1, dim_q); YR = zeros(NROOTS, dim_q)
    reps = storage == :fois ? 2 : 3
    t1 = best(() -> TPSChem.mul_block!(Y1, op1, X1), reps)
    tR = best(() -> TPSChem.mul_block!(YR, opR, X), reps)
    push!(results1, (storage, t1, tR, tbuild))
    @printf(" %-10s %12.4f %12.4f %9.2fx %12.4f %12.2f\n",
            storage, t1, tR, NROOTS*t1/tR, tR/NROOTS, tbuild)
    flush(stdout)
end
@printf(" └%s┘\n", "─"^86)
@printf(" speedup = %i x (1-root time) / (block time): 1.0 means blocking bought nothing\n", NROOTS)

# ── Table 2: end-to-end, one subprocess per configuration ────────────────────
configs = [("minres", "sparse", "false"), ("minres", "sparse", "true"),
           ("minres", "matvec", "false"), ("minres", "matvec", "true"),
           ("minres", "fois",   "false"), ("minres", "fois",   "true"),
           ("pcg",    "sparse", "false"), ("pcg",    "sparse", "true"),
           ("krylov", "fois",   "false")]

@printf("\n ┌ Table 2: end-to-end CEPA-0 (fresh process each, peak RSS is real) ───────────────────┐\n")
@printf(" %-8s %-8s %-10s %10s %10s %10s %10s\n",
        "solver", "storage", "multiroot", "time (s)", "alloc GiB", "heap GiB", "RSS GiB")
rows = Any[]
for (slv, st, mr) in configs
    cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) $(@__FILE__)`
    env = copy(ENV)
    env["CEPA_BENCH_CHILD"] = "$slv,$st,$mr"
    env["JULIA_NUM_THREADS"] = string(NTHREAD)
    env["CEPA_BENCH_MAXITER"] = string(st == "fois" ? FOISIT : MAXIT)
    out = try
        read(setenv(cmd, env), String)
    catch err
        @printf(" %-8s %-8s %-10s   FAILED (%s)\n", slv, st, mr, sprint(showerror, err))
        continue
    end
    line = findfirst(l -> startswith(l, "RESULT"), split(out, '\n'))
    if line === nothing
        @printf(" %-8s %-8s %-10s   NO RESULT\n", slv, st, mr)
        continue
    end
    f = split(split(out, '\n')[line], '\t')
    @printf(" %-8s %-8s %-10s %10.2f %10.3f %10.3f %10.3f\n",
            f[2], f[3], f[4], parse(Float64,f[5]), parse(Float64,f[6]),
            parse(Float64,f[7]), parse(Float64,f[8]))
    push!(rows, (slv, st, mr, parse.(Float64, split(f[9], ','))))
    flush(stdout)
end
@printf(" └%s┘\n", "─"^86)

# ── Energies must agree across every pathway ─────────────────────────────────
if !isempty(rows)
    ref = rows[1][4]
    @printf("\n Energy agreement (vs %s/%s/multiroot=%s):\n", rows[1][1], rows[1][2], rows[1][3])
    for (slv, st, mr, e) in rows
        @printf("   %-8s %-8s multiroot=%-6s  max|ΔE| = %.2e\n",
                slv, st, mr, maximum(abs.(e .- ref)))
    end
end
println()
