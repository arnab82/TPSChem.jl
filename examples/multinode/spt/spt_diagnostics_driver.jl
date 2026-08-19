# =============================================================================
# SPT sharding diagnostics -- run this FIRST.
#
# Decides, for YOUR system, whether the 2D grid matvec is worth using, before
# you spend node-hours on a scaling run.  Two measurements:
#
#   1. Fock-coupling density.  The 2D grid pays only when a worker's slice of
#      sectors does not already touch essentially every other sector.  On h12
#      the density is 6.8% with mean bra in-degree 142 of 2090, which is why
#      re-partitioning by *contribution* moves exactly the same bytes (it is
#      the dual of the same cut) while the *grid* does not.
#   2. Load spread.  The sharded FOIS balances by the count of contributing ket
#      sectors.  On h12 that is within 1.03-1.12x of a work-weighted assignment,
#      i.e. not worth changing -- check whether that holds for you.
#
# Launch (no workers needed; this is metadata-only and runs on one process):
#   julia --project examples/multinode/spt/spt_diagnostics_driver.jl input.jld2
# =============================================================================

using LinearAlgebra, Printf, Statistics, JLD2, TPSChem
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(@__DIR__, "spt_common.jl"))

const P_LIST = haskey(ENV, "TPSCHEM_P_LIST") ?
    parse.(Int, split(ENV["TPSCHEM_P_LIST"], ",")) : [4, 9, 16, 25, 36]

data = load_problem_data()
ch = get_clustered_ham(data)
state = spt_reference_state(data)

focks = collect(keys(state.data))
S = length(focks)
elems = Dict(f => sum(length(t.core[1]) for (_, t) in state.data[f]) for f in focks)
bytes = Dict(f => sum(sum(sizeof(c) for c in t.core) for (_, t) in state.data[f]) for f in focks)
total = sum(values(bytes))
@printf("\nstate: dim=%d  Fock sectors=%d  cores=%.1f MB\n", length(state), S, total*1e-6)

# ---- 1. coupling ------------------------------------------------------------
nbr = [Int[] for _ in 1:S]
deg = zeros(Int, S)
for (ib, fb) in enumerate(focks), (ik, fk) in enumerate(focks)
    haskey(ch, fb - fk) || continue
    push!(nbr[ik], ib); deg[ib] += 1
end
pairs = sum(length.(nbr))
@printf("coupling: %d of %d pairs (%.1f%% dense)   bra in-degree mean %.0f median %d max %d\n",
        pairs, S*S, 100*pairs/(S*S), mean(deg), median(deg), maximum(deg))

@printf("\n%-5s %14s %14s %14s %10s\n", "P", "destination", "2D grid", "ring", "grid wins?")
for P in P_LIST
    P > S && continue
    qr = isqrt(P); while qr > 1 && P % qr != 0; qr -= 1; end
    qc = P ÷ qr
    own = Dict(f => (i-1) % P  for (i, f) in enumerate(focks))
    rg  = Dict(f => (i-1) % qr for (i, f) in enumerate(focks))
    cg  = Dict(f => (i-1) % qc for (i, f) in enumerate(focks))
    pull = 0
    for p in 0:P-1
        mine = Set(ib for (ib, fb) in enumerate(focks) if own[fb] == p)
        need = Set{Int}()
        for (ik, fk) in enumerate(focks)
            own[fk] == p && continue
            any(ib -> ib in mine, nbr[ik]) && push!(need, ik)
        end
        pull += sum(bytes[focks[ik]] for ik in need; init=0)
    end
    grid = 0
    for i in 0:qr-1, j in 0:qc-1
        ks = Set{Int}(); bs = Set{Int}()
        for (ik, fk) in enumerate(focks)
            cg[fk] == j || continue
            for ib in nbr[ik]
                rg[focks[ib]] == i || continue
                push!(ks, ik); push!(bs, ib)
            end
        end
        grid += sum(bytes[focks[ik]] for ik in ks; init=0)
        grid += sum(bytes[focks[ib]] for ib in bs; init=0)
    end
    @printf("%-5d %11.1f MB %11.1f MB %11.1f MB %10s   (grid %dx%d)\n",
            P, pull*1e-6, grid*1e-6, P*total*1e-6,
            grid < pull ? "YES $(round(pull/grid, digits=2))x" : "no", qr, qc)
end
@printf("\npeak transient per worker: destination stays ~O(|state|) however many\n")
@printf("workers you add; 2D grid is O(|state|/sqrt(P)); ring is O(|state|/P).\n")

# ---- 2. FOIS load spread ----------------------------------------------------
cand = Dict{Any,Vector{Any}}()
for fk in focks, (ftrans, _) in ch
    fb = ftrans + fk
    TPSChem._spt_valid_fock(fb, state.clusters) || continue
    push!(get!(() -> [], cand, fb), fk)
end
work = Dict(fb => sum(elems[fk]*length(ch[fb-fk]) for fk in ks; init=0) for (fb, ks) in cand)
function spread(P, w)
    load = zeros(Int, P); got = zeros(Int, P)
    for (fb, ks) in cand
        p = argmin(load); load[p] += w(fb, ks); got[p] += work[fb]
    end
    return maximum(got)/mean(got)
end
@printf("\nFOIS assignment (%d candidate output sectors)\n", length(cand))
@printf("%-5s %26s %22s\n", "P", "current (count of kets)", "work-weighted")
for P in P_LIST
    P > length(cand) && continue
    @printf("%-5d %22.2fx %21.2fx\n", P,
            spread(P, (fb, ks) -> length(ks)), spread(P, (fb, ks) -> work[fb]))
end
@printf("\nIf 'current' is close to 'work-weighted', the count proxy is fine and\n")
@printf("re-weighting the assignment will not buy you anything.\n")
