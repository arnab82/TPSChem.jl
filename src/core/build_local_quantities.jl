using .ActiveSpaceSolvers
using .BlockDavidson




"""
    get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)

For a given `ClusterSubspace`, `tss`, return the subspace remaining
"""
function get_ortho_compliment(tss::ClusterSubspace, cb::ClusterBasis)
#={{{=#
    data = OrderedDict{Tuple{UInt8,UInt8}, UnitRange{Int}}()
    for (fock,basis) in cb
    
        if haskey(tss.data,fock)
            first(tss.data[fock]) == 1 || error(" p-space doesn't include ground state?")
            newrange = last(tss[fock])+1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        else
            newrange = 1:size(cb[fock],2)
            if length(newrange) > 0
                data[fock] = newrange
            end
        end
    end

    return ClusterSubspace(tss.cluster, data)
#=}}}=#
end


"""
    tdm_spinfree_2rdm(cb::ClusterBasis; verbose=0)

Compute the local spin-free two-body transition density

    Gamma[p,q,r,s,u,v] = sum_{sigma,tau} <u|p'_sigma q'_tau s_tau r_sigma|v>

for each Fock-neutral sector of a cluster basis. The first four dimensions are
local orbital indices in the same convention as `compute_2rdm`; the final two
dimensions are bra and ket cluster-basis state indices.
"""
function tdm_spinfree_2rdm(cb::ClusterBasis; verbose=0)
    verbose == 0 || println("")
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    for (fock, basis) in cb
        nstates = size(basis, 2)
        T = eltype(basis.vectors)
        block = zeros(T, norbs, norbs, norbs, norbs, nstates, nstates)

        for ket_idx in 1:nstates
            ket_vec = basis.vectors[:, ket_idx]
            for bra_idx in 1:nstates
                bra_vec = basis.vectors[:, bra_idx]
                _, _, rdm2aa, rdm2bb, rdm2ab =
                    ActiveSpaceSolvers.FCI.compute_rdm1_rdm2(basis.ansatz, bra_vec, ket_vec)

                for s_orb in 1:norbs, r_orb in 1:norbs, q_orb in 1:norbs, p_orb in 1:norbs
                    block[p_orb, q_orb, r_orb, s_orb, bra_idx, ket_idx] =
                        rdm2aa[p_orb, r_orb, q_orb, s_orb] +
                        rdm2bb[p_orb, r_orb, q_orb, s_orb] +
                        rdm2ab[p_orb, r_orb, q_orb, s_orb] +
                        rdm2ab[q_orb, s_orb, p_orb, r_orb]
                end
            end
        end

        dicti[(fock, fock)] = block
    end
    return dicti
end



"""
    compute_cluster_ops(cluster_bases::Vector{ClusterBasis}, ints)

Build the standard local operator tables used by TPSCI. This intentionally
does not build the expensive local spin-free 2-RDM table `Ppqsr`; use
`compute_cluster_ops_2rdm` when calling `compute_2rdm`.
"""
function _compute_cluster_ops_for_cluster(cb, ints::InCoreInts{T}; verbose=1) where {T}
#={{{=#
        ci = cb.cluster
        ops = ClusterOps(ci, T=T)

        verbose < 1 || display(ci)
        flush(stdout)
       
        ops["H"] = TPSChem.tdm_H(cb, subset(ints, ci.orb_list), verbose=0)
        ops["A"], ops["a"] = TPSChem.tdm_A(cb,"alpha")
        ops["B"], ops["b"] = TPSChem.tdm_A(cb,"beta")
        ops["AA"], ops["aa"] = TPSChem.tdm_AA(cb,"alpha")
        ops["BB"], ops["bb"] = TPSChem.tdm_AA(cb,"beta")
        ops["Aa"] = TPSChem.tdm_Aa(cb,"alpha")
        ops["Bb"] = TPSChem.tdm_Aa(cb,"beta")
        ops["Ab"], ops["Ba"] = TPSChem.tdm_Ab(cb)
        # remove BA and ba account for these terms 
        ops["AB"], ops["ba"], ops["BA"], ops["ab"] = TPSChem.tdm_AB(cb)
        ops["AAa"], ops["Aaa"] = TPSChem.tdm_AAa(cb,"alpha")
        ops["BBb"], ops["Bbb"] = TPSChem.tdm_AAa(cb,"beta")
        ops["ABa"], ops["Aba"] = TPSChem.tdm_ABa(cb,"alpha")
        ops["ABb"], ops["Bba"] = TPSChem.tdm_ABa(cb,"beta")
        #ops["ABa"], ops["Aba"], ops["BAa"], ops["Aab"] = TPSChem.tdm_ABa(cb,"alpha")
        #ops["ABb"], ops["Bba"], ops["BAb"], ops["Bab"] = TPSChem.tdm_ABa(cb,"beta")
       
        # spin operators
        
        #
        # S+
        op = Dict{Tuple,Array}()
        for (fock,mat) in ops["Ab"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat[p,p,i,j]
                    end
                end
            end
        end
        ops["S+"] = op
        
        #
        # S-
        op = Dict{Tuple,Array}()
        for (fock,mat) in ops["Ba"]
            dims = size(mat)
            op[fock] = zeros(dims[2:4]...)
            for j in 1:dims[4]
                for i in 1:dims[3]
                    for p in 1:dims[2]
                        op[fock][p,i,j] = mat[p,p,i,j]
                    end
                end
            end
        end
        ops["S-"] = op
        
        #
        # Sz
        op = Dict{Tuple,Array}()
        #
        # loop over fock-space transitions
        for (fock,basis) in cb
            focktrans = (fock,fock)

            sz = (fock[1] - fock[2]) / 2.0
            op[focktrans] = sz*Matrix(1.0I, size(cb[fock],2), size(cb[fock],2))
            op[focktrans] = reshape(op[focktrans],1,size(op[focktrans],1),size(op[focktrans],2))

        end
        ops["Sz"] = op


        #
        # S2
        ops["S2"] = TPSChem.tdm_S2(cb, subset(ints, ci.orb_list), verbose=0)


        to_delete = [
                     #"AAa",
                     #"Aaa",
                     #"BBb",
                     #"Bbb",
                     #
                     #"ABa",
                     #"Aba",
                     ##"BAa",
                     ##"Aab",
                     #
                     #"ABb",
                     #"Bba",
                     ##"BAb",
                     ##"Bab",
                     #"Aa",
                     #"Bb",
                     #"Ab",
                     #"Ba",
                     #"AB",
                     #"ba",
                     #"BA",
                     #"ab",
                     #"AA",
                     #"BB",
                     #"aa",
                     #"bb"
                     ]
        for op in to_delete
            for (ftran,array) in ops[op]
                ops[op][ftran] .*= 0
            end
        end


        # Compute single excitation operator
        tmp = Dict{Tuple,Array}()
        for (fock,basis) in cb
            tmp[(fock,fock)] = (ops["Aa"][(fock,fock)] + ops["Bb"][(fock,fock)])
        end
        ops["E1"] = tmp

        

        #
        # reshape data into 3index quantities: e.g., (pqr, I, J)
        for opstring in keys(ops)
            opstring != "H" || continue
            opstring != "S2" || continue
            for ftrans in keys(ops[opstring])
                data = ops[opstring][ftrans]
                dim1 = prod(size(data)[1:(length(size(data))-2)])
                dim2 = size(data)[length(size(data))-1]
                dim3 = size(data)[length(size(data))-0]
                ops[opstring][ftrans] = copy(reshape(data, (dim1,dim2,dim3)))
            end
        end
    return ops
end
    #=}}}=#

function compute_cluster_ops(cluster_bases, ints::InCoreInts{T}) where {T}
#={{{=#
    clusters = Vector{MOCluster}()
    for ci in cluster_bases
        push!(clusters, ci.cluster)
    end

    cluster_ops = Vector{ClusterOps{T}}(undef, length(clusters))
    for ci in clusters
        cluster_ops[ci.idx] = _compute_cluster_ops_for_cluster(cluster_bases[ci.idx], ints)
        # Each cluster's build churns far more than it retains (the FCI kernels
        # allocate large working tensors that do not end up in the returned
        # dicts), so reclaim that before starting the next cluster rather than
        # letting it pile up until Julia's incremental GC gets to it on its own.
        GC.gc(false)
    end
    return cluster_ops
end
    #=}}}=#


"""
    add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)

Add the exact local spin-free 2-RDM transition table `Ppqsr` to an existing
`cluster_ops` object. This is only needed for physically complete
`compute_2rdm` calculations.
"""
function add_spinfree_2rdm_ops!(cluster_ops::Vector{ClusterOps{T}},
                                cluster_bases) where {T}
    for cb in cluster_bases
        ci = cb.cluster
        display(ci)
        flush(stdout)

        Ppqsr = TPSChem.tdm_spinfree_2rdm(cb)
        for ftrans in keys(Ppqsr)
            data = Ppqsr[ftrans]
            dim1 = prod(size(data)[1:(ndims(data)-2)])
            dim2 = size(data, ndims(data)-1)
            dim3 = size(data, ndims(data))
            Ppqsr[ftrans] = copy(reshape(data, (dim1, dim2, dim3)))
        end
        cluster_ops[ci.idx]["Ppqsr"] = Ppqsr
    end
    return cluster_ops
end


"""
    compute_cluster_ops_2rdm(cluster_bases, ints)

Build the standard TPSCI local operators plus the expensive `Ppqsr` table
required by `compute_2rdm`. Use this instead of `compute_cluster_ops` only
when you need a 2-RDM.
"""
function compute_cluster_ops_2rdm(cluster_bases, ints::InCoreInts{T}) where {T}
    cluster_ops = TPSChem.compute_cluster_ops(cluster_bases, ints)
    return TPSChem.add_spinfree_2rdm_ops!(cluster_ops, cluster_bases)
end


"""
    _tdm_build_concurrency()

Number of fock-space transitions to build *at once* during the cluster-operator
build -- independent of `Threads.nthreads()`.

Each transition in flight holds a large transient tensor (a 3-body operator on
an 11-orbital cluster is ~0.4 GB at M=200) while the FCI kernel underneath
computes it, on top of the arrays it returns.  Handing every one of the node's
Julia threads a transition at once, as the first version of this did, bounds
final retained memory correctly but not *peak* memory: it can have as many of
those transients live simultaneously as there are threads, which is a real
difference from the serial build this replaced -- one transition's garbage at a
time there, GC keeping pace.

Defaulting to a smaller number of concurrent transitions keeps that peak
bounded while still using the whole node: the freed core budget goes to BLAS
instead (`_ops_build_blas_threads` divides by this same cap), and the chunk
functions call `GC.gc(false)` between clusters to reclaim what each cluster's build
churned before the next one starts.  Override with `TPSCHEM_OPS_BUILD_THREADS`.
"""
function _tdm_build_concurrency()
    v = get(ENV, "TPSCHEM_OPS_BUILD_THREADS", "")
    n = tryparse(Int, v)
    cap = (n === nothing || n < 1) ? 16 : n
    return clamp(cap, 1, Threads.nthreads())
end

"""
    _tdm_thread_map(jobs, f)

Apply `f` to each element of `jobs`, running at most `_tdm_build_concurrency()`
of them at once, and return the results in `jobs` order.

Measured on real cluster data at 8 threads, three ways of bounding concurrency
below `length(jobs)` all cost about the same, and all cost more than the
original unbounded `Threads.@threads for i in eachindex(jobs)`: pulling jobs
one at a time from a shared atomic counter (720s vs a 549s baseline), fixed
round-robin chunks run with explicit `:static` scheduling (669s), and the same
chunks run with `:dynamic` (671s, matching bare `Threads.@threads`'s default
since Julia 1.8). The three are close enough to each other, and far enough from
the baseline, that the gap is evidently in the *chunking* itself rather than in
which scheduler runs the chunks -- the exact mechanism is not identified.

Since the cost is in chunking, not in the concurrency limit itself, the fix is
to only pay it when the cap actually binds: when `_tdm_build_concurrency() >=
length(jobs)`, every "chunk" would hold exactly one job anyway, so this skips
straight to the plain per-item loop that measured 549s. Chunking only engages
when there are genuinely more jobs than the cap allows in flight -- the regime
this whole cap exists for (transitions per builder can be several times the
default cap of 16), and the one this file's tests have not been able to
reproduce on an 8-thread laptop where the cap never binds below `nthreads()`.
"""
function _tdm_thread_map(jobs::Vector, f)
    n = length(jobs)
    n == 0 && return Any[]
    vals = Vector{Any}(undef, n)
    cap = _tdm_build_concurrency()
    if cap >= n
        Threads.@threads for i in 1:n
            vals[i] = f(jobs[i])
        end
        return vals
    end
    chunks = [Int[] for _ in 1:cap]
    for i in 1:n
        push!(chunks[mod1(i, cap)], i)
    end
    Threads.@threads for c in 1:cap
        for i in chunks[c]
            vals[i] = f(jobs[i])
        end
    end
    return vals
end

"""
    _tdm_fock_pairs(cb, offset, range)

Collect the `(bra, ket)` fock-sector pairs a tdm builder will actually touch.
Separating enumeration from computation is what lets the expensive part be
threaded; the `haskey` guard is the same one the serial loops applied inline.
"""
function _tdm_fock_pairs(cb, offset::Tuple{Int,Int}, range)
    jobs = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    for na in range
        for nb in range
            fockket = (na, nb)
            fockbra = (na + offset[1], nb + offset[2])
            if haskey(cb, fockbra) && haskey(cb, fockket)
                push!(jobs, (fockbra, fockket))
            end
        end
    end
    return jobs
end

"""
    tdm_H(cb::ClusterBasis; verbose=0)

Compute local Hamiltonian `<s|H|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_H(cb::ClusterBasis, ints; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    verbose == 0 || display(cb.cluster)
    focks = collect(keys(cb.basis))
    vals = _tdm_thread_map(focks, function (fock)
                               basis = cb[fock]
                               Hmap = LinearMap(ints, basis.ansatz)
                               return cb[fock]' * Matrix((Hmap * cb[fock]))
                           end)
    for (i, fock) in enumerate(focks)
        focktrans = (fock,fock)
        verbose == 0 || display(cb[fock].ansatz)
        dicti[focktrans] = vals[i]

        if verbose > 0
            for e in 1:size(cb[fock],2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
            end
        end
    end
    return dicti
#=}}}=#
end


"""
"""
function tdm_S2(cb::ClusterBasis, ints; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(cb.cluster)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    verbose == 0 || display(cb.cluster)
    focks = collect(keys(cb.basis))
    vals = _tdm_thread_map(focks, function (fock)
                               basis = cb[fock]
                               return cb[fock]' * apply_S2_matrix(basis.ansatz, cb[fock].vectors)
                           end)
    for (i, fock) in enumerate(focks)
        focktrans = (fock,fock)
        verbose == 0 || display(cb[fock].ansatz)
        dicti[focktrans] = vals[i]

        if verbose > 0
            for e in 1:size(cb[fock],2)
                @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
            end
        end
    end
    return dicti
#=}}}=#
end


"""
    tdm_A(cb::ClusterBasis; verbose=0)

Compute `<s|p'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_A(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    spin_case in ("alpha", "beta") || throw(DomainError(spin_case))
    jobs = _tdm_fock_pairs(cb, spin_case == "alpha" ? (1,0) : (0,1), 0:norbs)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               d = spin_case == "alpha" ?
                                   compute_operator_c_a(basis_bra, basis_ket) :
                                   compute_operator_c_b(basis_bra, basis_ket)
                               return (d, permutedims(d, [1,3,2]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)] = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AA(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AA(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    spin_case in ("alpha", "beta") || throw(DomainError(spin_case))
    jobs = _tdm_fock_pairs(cb, spin_case == "alpha" ? (2,0) : (0,2), 0:norbs)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               d = spin_case == "alpha" ?
                                   compute_operator_cc_aa(basis_bra, basis_ket) :
                                   compute_operator_cc_bb(basis_bra, basis_ket)
                               return (d, permutedims(d, [2,1,4,3]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)] = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Aa(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    spin_case in ("alpha", "beta") || throw(DomainError(spin_case))
    jobs = _tdm_fock_pairs(cb, (0,0), 0:norbs)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               return spin_case == "alpha" ?
                                      compute_operator_ca_aa(basis_bra, basis_ket) :
                                      compute_operator_ca_bb(basis_bra, basis_ket)
                           end)
    for (i, job) in enumerate(jobs)
        dicti[(job[1],job[2])] = vals[i]
    end
    return dicti
#=}}}=#
end


"""
    tdm_Ab(cb::ClusterBasis; verbose=0)

Compute `<s|p'q|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_Ab(cb::ClusterBasis; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    jobs = _tdm_fock_pairs(cb, (1,-1), -1:norbs+1)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               d = compute_operator_ca_ab(basis_bra, basis_ket)
                               return (d, permutedims(d, [2,1,4,3]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)] = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_AB(cb::ClusterBasis; verbose=0)

Compute `<s|p'q'|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space, where
`p'` is alpha and `q'` is beta.

Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AB(cb::ClusterBasis; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    dictj = Dict{Tuple,Array}()
    dictj_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    jobs = _tdm_fock_pairs(cb, (1,1), -2:norbs+2)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               di = compute_operator_cc_ab(basis_bra, basis_ket)
                               dj = -permutedims(di, [2,1,3,4])
                               return (di, permutedims(di, [2,1,4,3]),
                                       dj, permutedims(dj, [2,1,4,3]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)]     = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
        dictj[(fockbra,fockket)]     = vals[i][3]
        dictj_adj[(fockket,fockbra)] = vals[i][4]
    end
    return dicti, dicti_adj, dictj, dictj_adj
#=}}}=#
end


"""
    tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t` 
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`
"""
function tdm_AAa(cb::ClusterBasis, spin_case; verbose=0)
#={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    spin_case in ("alpha", "beta") || throw(DomainError(spin_case))
    jobs = _tdm_fock_pairs(cb, spin_case == "alpha" ? (1,0) : (0,1), 0:norbs)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               d = spin_case == "alpha" ?
                                   compute_operator_cca_aaa(basis_bra, basis_ket) :
                                   compute_operator_cca_bbb(basis_bra, basis_ket)
                               return (d, permutedims(d, [3,2,1,5,4]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)] = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
    end
    return dicti, dicti_adj
#=}}}=#
end


"""
    tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)

Compute `<s|p'q'r|t>` between all cluster states, `s` and `t`
from accessible sectors of a cluster's fock space.
- `spin_case`: alpha or beta
Returns `Dict[((na,nb),(na,nb))] => Array`

Only `(dicti, dicti_adj)` are returned. An earlier version also built `dictj`
(`= -permutedims(dicti, ...)`) and its adjoint, but `_compute_cluster_ops_for_cluster`
-- the only caller -- has only ever kept two of the four return values
(`ops["ABa"], ops["Aba"] = tdm_ABa(...)`); Julia's tuple destructuring silently
drops extra values rather than erroring, so `dictj`/`dictj_adj` were computed in
full (one `permutedims` and one negation per fock transition, each the same
size as `dicti`) and then immediately discarded. That doubled both the transient
memory live during the build and the wall time spent on this builder for no
retained benefit -- removed here since nothing has ever read them.
"""
function tdm_ABa(cb::ClusterBasis, spin_case; verbose=0)
    #={{{=#
    verbose == 0 || println("")
    verbose == 0 || display(ci)
    norbs = length(cb.cluster)

    dicti = Dict{Tuple,Array}()
    dicti_adj = Dict{Tuple,Array}()
    #
    # loop over fock-space transitions
    spin_case in ("alpha", "beta") || throw(DomainError(spin_case))
    # NB: the spin -> sector offset here is the reverse of tdm_A/tdm_AAa --
    # "alpha" raises nb and "beta" raises na.  Preserved from the serial loop.
    jobs = _tdm_fock_pairs(cb, spin_case == "alpha" ? (0,1) : (1,0), -2:norbs+2)
    vals = _tdm_thread_map(jobs, function (job)
                               basis_bra = cb[job[1]]
                               basis_ket = cb[job[2]]
                               di = spin_case == "alpha" ?
                                    compute_operator_cca_aba(basis_bra, basis_ket) :
                                    compute_operator_cca_abb(basis_bra, basis_ket)
                               return (di, permutedims(di, [3,2,1,5,4]))
                           end)
    for (i, job) in enumerate(jobs)
        fockbra, fockket = job
        dicti[(fockbra,fockket)]     = vals[i][1]
        dicti_adj[(fockket,fockbra)] = vals[i][2]
    end
    return dicti, dicti_adj
    #=}}}=#
end



"""
    function add_cmf_operators!(ops::Vector{ClusterOps}, bases::Vector{ClusterBasis}, ints, Da, Db; verbose=0)

Add effective local hamiltonians (local CASCI) type hamiltonians to a `ClusterOps` type for each `Cluster'
"""
function _add_cmf_operator_for_cluster!(ops_i, cb, ints, Da, Db; verbose=0)
    #={{{=#
        ci = cb.cluster
        verbose == 0 || println()
        verbose == 0 || display(ci)
        norbs = length(cb.cluster)
        
        ints_i = subset(ints, ci.orb_list, Da, Db)
        #ints_i = form_casci_ints(ints, ci, Da, Db)


        dicti = Dict{Tuple,Array}()
        
        #
        # loop over fock-space transitions
        verbose == 0 || display(cb.cluster)
        focks = collect(keys(cb.basis))
        vals = _tdm_thread_map(focks, function (fock)
                                   basis = cb[fock]
                                   Hmap = LinearMap(ints_i, basis.ansatz)
                                   return cb[fock]' * Matrix((Hmap * cb[fock]))
                               end)
        for (i, fock) in enumerate(focks)
            focktrans = (fock,fock)
            verbose == 0 || display(cb[fock].ansatz)
            dicti[focktrans] = vals[i]

            if verbose > 0
                for e in 1:size(cb[fock],2)
                    @printf(" %4i %12.8f\n", e, dicti[focktrans][e,e])
                end
            end
        end
        ops_i["Hcmf"] = dicti
    return ops_i
end
#=}}}=#

function add_cmf_operators!(ops, bases, ints, Da, Db; verbose=0)
    #={{{=#
    n_clusters = length(bases)
    for ci_idx in 1:n_clusters
        _add_cmf_operator_for_cluster!(ops[ci_idx], bases[ci_idx], ints, Da, Db,
                                       verbose=verbose)
        GC.gc(false)
    end
    return 
end
#=}}}=#


"""
	form_schmidt_basis
thresh_orb      :   threshold for determining how many bath orbitals to include
thresh_schmidt  :   threshold for determining how many singular vectors to include for cluster basis

Returns new basis for the cluster
"""
function form_schmidt_basis(ints::InCoreInts, ci::MOCluster, Da, Db; 
        thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,do_embedding=true,
        eig_nr=1, eig_max_cycles=200,
        A::Type=FCIAnsatz)

    println()
    println("------------------------------------------------------------")
    @printf("Form Embedded Schmidt-style basis for Cluster %4i\n",ci.idx)
    D = Da + Db 

    # Form the exchange matrix
    K = zeros(size(ints.h1))
    @tensor begin
	K[q,r]  = ints.h2[p,q,r,s] * D[p,s]
    end

    no = size(ints.h1,1)
    ci_no = length(ci.orb_list)


    na_tot = Int(round(tr(Da)))
    nb_tot = Int(round(tr(Db)))
    println(" Number of electrons in full system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_tot,nb_tot)

    active = ci.orb_list

    backgr = Vector{Int}()
    for i in 1:no
    	if !(i in active)
    	    append!(backgr,i)
        end
    end

    println("active",active)
    println("backgr",backgr)

    K2 = zeros((ci_no,no-ci_no))

    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(backgr)
	    K2[pi,qi] = K[p,q]
	end
    end

    println("The exchange matrix:")
    display(K2)
    F = svd(K2,full=true)

    @printf("\nSing. Val.\n")
    nkeep = 0
    for si in F.S
    	@printf("%16.12f\n",si)
        if si > thresh_orb
            nkeep += 1
        end
    end

    C = zeros(size(ints.h1))
    for (pi,p) in enumerate(active)
    	for (qi,q) in enumerate(active)
	    if pi==qi
	        C[p,qi] = 1
	    end
	end
    end

    #display(C)

    #display(F.Vt)
    #display(F.U)
    for (pi,p) in enumerate(backgr)
    	for (qi,q) in enumerate(backgr)
	    C[p,qi+length(active)] = F.Vt[qi,pi]
	end
    end
    #display(C)

    Cfrag = C[:,1:ci_no]
    Cbath = C[:,ci_no+1:ci_no+nkeep]
    Cenvt = C[:,ci_no+nkeep+1:end]
    
    @printf("Cfrag\n")
    display(Cfrag)
    @printf("\n NElec: %12.8f\n",(tr(Cfrag'*(Da+Db)*Cfrag)))
    @printf("Cbath\n")
    display(Cbath)
    @printf("\n NElec: %12.8f\n",(tr(Cbath'*(Da+Db)*Cbath)))
    @printf("Cenv\n")
    display(Cenvt)
    @printf("\n NElec: %12.8f\n",(tr(Cenvt'*(Da+Db)*Cenvt)))

    K2 = C'* K * C
    Da2 = C'* Da * C
    Db2 = C'* Db * C

    na = tr(Da2[1:ci_no+nkeep,1:ci_no+nkeep])
    nb = tr(Db2[1:ci_no+nkeep,1:ci_no+nkeep])

    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na,nb)

    denvt_a = Cenvt*Cenvt'*Da*Cenvt*Cenvt'
    denvt_b = Cenvt*Cenvt'*Db*Cenvt*Cenvt'
    #println(denvt_a)

    na_env = tr(denvt_a)
    nb_env = tr(denvt_b)

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_env,nb_env)

    na_envt = Int(round(tr(Cenvt'*Da*Cenvt)))
    nb_envt = Int(round(tr(Cenvt'*Db*Cenvt)))


    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_envt,nb_envt)
    #display(Da)

    # rotate integrals to current subspace basis
    denvt_a = C'*denvt_a*C
    denvt_b = C'*denvt_b*C
    ints2 = orbital_rotation(ints, C)

    #avoid very zero numbers in diagonalization
    denvt_a[abs.(denvt_a) .< 1e-15] .= 0
    denvt_b[abs.(denvt_b) .< 1e-15] .= 0
    #display(denvt_a)
    #display(denvt_a)

    # find closest idempotent density for the environment
    if do_embedding
        if size(Cenvt,2)>0

	    #eigenvalue 
	    EIG = eigen(denvt_a)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]
	    #println(n)
	    #display(U)

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_a = U[:,1:na_envt] * U[:,1:na_envt]'

	    EIG = eigen(denvt_b)
	    U = EIG.vectors
	    n = EIG.values

	    U = U[:, sortperm(n,rev=true)]
	    n = n[sortperm(n,rev=true)]

            for i in 1:nkeep
                @assert(n[i]>1e-14)
	    end

            denvt_b = U[:,1:nb_envt] * U[:,1:nb_envt]'

	end
    #form ints in the cluster 
    no_range = collect(1:size(Cfrag,2)+size(Cbath,2))

    #ints_f = subset(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)

    ints_f = subset(ints2,no_range,denvt_a,denvt_b)
    #ints_f = TPSChem.form_1rdm_dressed_ints(ints2,no_range,denvt_a,denvt_b)

    else
        denvt_a *= 0 
        denvt_b *= 0 
        ints_f = form_casci_eff_ints(ints2,collect(1:size(Cfrag,2)+size(Cbath,2)), denvt_a, denvt_b)
    end

    println(" Number of electrons in Environment system:")
    @printf("  α: %12.8f  β:%12.8f \n ",tr(denvt_a),tr(denvt_b))

    na_actv = na_tot - na_envt
    nb_actv = nb_tot - nb_envt
    println(" Number of electrons in Fragment+Bath system:")
    @printf("  α: %12.8f  β:%12.8f \n ",na_actv,nb_actv)

    norb2 = size(ints_f.h1,1)

    ansatz = FCIAnsatz(norb2, na_actv, nb_actv)
    Hmap = LinearMap(ints_f, ansatz)
    v0 = svd(rand(ansatz.dim,eig_nr)).U
    davidson = TPSChem.Davidson(Hmap,v0=v0,max_iter=200, max_ss_vecs=20, nroots=eig_nr, tol=1e-8)
    #TPSChem.solve(davidson)
    @printf(" Now iterate: \n")
    flush(stdout)
    #@time TPSChem.iteration(davidson, Adiag=Adiag, iprint=2)
    @time e,v = BlockDavidson.eigs(davidson);

    solution = Solution(ansatz, e, v)
    ansatz = FCIAnsatz(norb2, na_actv, nb_actv)
    
    #solution = solve(ints_f, ansatz, SolverSettings(maxiter=200, nroots=eig_nr, tol=1e-8))
    #solution = solve(ints_f, ansatz, SolverSettings(maxiter=200, nroots=eig_nr, tol=1e-8))
    
    basis = svd_state(solution, length(active), nkeep, thresh_schmidt)

    return basis
end


"""
    compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                       clusters::Vector{MOCluster}, 
                                       rdm1::RDM1{T},
                                       delta_elec::Vector,
                                       ref_fock::FockConfig; 
                                       verbose=0, 
                                       max_roots=10, 
                                       A::Type=FCIAnsatz) where T

Return a Vector of `ClusterBasis` for each `Cluster`.
For each number of electrons specified by ref_fock +- 1->delta_elec (for each cluster), 
we solve the CASCI problem, collecting `max_roots` of the lowest energy eigenvectors for the half-filled (or of odd number nalpha = nbeta+1) level. Then we apply S^- and S^+ to generate the higher/lower m_s blocks directly. 

# Arguments
#
- `ints`: InCoreInts integrals
- `clusters`: Clusters 
- `verbose`: Print level
- `ref_fock`:  reference space for defining target focksectors with `delta_elec`
- `delta_elec`: number of electrons different from reference (init_fspace) for each cluster
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1`: background density matrix for embedding local hamiltonian 
- `A`: the type of Ansatz object used to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis_spin(   ints::InCoreInts{T}, 
                                            clusters::Vector{MOCluster}, 
                                            rdm1::RDM1{T},
                                            delta_elec::Vector,
                                            ref_fock::FockConfig; 
                                            verbose=0, 
                                            max_roots=10, 
                                            A::Type=FCIAnsatz) where T
    #={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    length(delta_elec) == length(clusters) || error("length(delta_elec) != length(clusters)") 
    for ci in clusters
        verbose == 0 || display(ci)
        

        ints_i = subset(ints, ci, rdm1) 


        # 
        # Verify that density matrix provided is consistent with reference fock sectors
        occs = diag(rdm1.a)
        occs[ci.orb_list] .= 0
        na_embed = sum(occs)
        occs = diag(rdm1.b)
        occs[ci.orb_list] .= 0
        nb_embed = sum(occs)
        verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f\n", na_embed, nb_embed)


        delta_e_i = delta_elec[ci.idx] 

        #
        # Get list of Fock-space sectors for current cluster
        #
        ni = ref_fock[ci.idx][1] + ref_fock[ci.idx][2]  # number of electrons in ci
        sectors = []
        max_e = 2*length(ci)
        min_e = 0
        for nj in ni-delta_e_i:ni+delta_e_i
        
            nj <= max_e || continue
            nj >= min_e || continue

            naj = nj÷2 + nj%2
            nbj = nj÷2
            push!(sectors, (naj, nbj))
        end

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T) 
        for sec in sectors

            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose == 0 || @printf(" Preparing to compute : \n")
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)

            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)

                basis_i[sec] = Solution(ansatz, F.values[1:nr], F.vectors[:,1:nr])

                #display(e)
            else
                #
                # Do sparse build 
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr,package="arpack"))
            end

            #
            # Loop over spin-flips
            # 
            # s2 = s(s+1) 
            

            s2 = compute_s2(basis_i[sec])    

            nr = length(basis_i[sec].energies)
            #for r in 1:nr
            #    S = (-1 + sqrt(1+4*s2[r]))/2
            #    gr = 2*S+1 # Degeneracy
            #end
          
            #
            #   S-
            #
            # find how many applications of S- we need to try
           
            verbose == 0 || println(" Compute higher and lower Ms components")
            n_sm = minimum((sec[1], ansatz.no-sec[2]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for smi in 1:n_sm
                vi, ansatzi = apply_sminus(vi, ansatzi)

                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
                #ei = compute_energy(vi, ansatzi)
            
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end
            #
            #   S+
            #
            # find how many applications of S+ we need to try
            
            n_sp = minimum((sec[2], ansatz.no-sec[1]))
            vi = deepcopy(basis_i[sec].vectors)
            ansatzi = deepcopy(basis_i[sec].ansatz)
            for spi in 1:n_sp
                vi, ansatzi = apply_splus(vi, ansatzi)
                
                verbose == 0 || display(ansatzi) 
                flush(stdout)

                if size(vi,2) == 0
                    # we have killed all the spin states
                    continue
                end

                Hmapi = LinearMap(ints_i, ansatzi)
                ei = diag(vi' * Matrix(Hmapi*vi))
                #ei = compute_energy(vi, ansatzi)
            
                si = Solution(ansatzi, ei, vi)
                seci = (ansatzi.na, ansatzi.nb)
                basis_i[seci] = si
            end

        end
           
        flush(stdout)
        if verbose > 0
            println()
            for (sec, sol) in basis_i    
                println()
                display(sol.ansatz)
                s2 = compute_s2(sol)    
                for i in 1:length(sol.energies)
                    @printf("   State %4i Energy: %12.8f S2: %12.8f\n",i, sol.energies[i], s2[i])
                end
                flush(stdout)
            end
        end

        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing, T::Type=Float64)

Return a Vector of `ClusterBasis` for each `Cluster` 
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{MOCluster}`: Clusters 
- `verbose::Int`: Print level
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `delta_elec`: number of electrons different from reference (init_fspace)
- `max_roots::Int`: Maximum number of vectors for each focksector basis
- `rdm1a`: background density matrix for embedding local hamiltonian (alpha)
- `rdm1b`: background density matrix for embedding local hamiltonian (beta)
- `ansatze`: should be a list of Ansatz objects so that we know how to solve each cluster. Default is FCIAnsatz     
- `T`: Data type of the eigenvectors 
"""
function compute_cluster_eigenbasis(ints::InCoreInts, clusters::Vector{MOCluster}; 
                init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
                rdm1a=nothing, rdm1b=nothing, 
                ansatze=nothing,     
                T::Type=Float64, A::Type=FCIAnsatz)
#={{{=#
    # initialize output
    #
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose == 0 || display(ci)
        
        if (rdm1a !== nothing && init_fspace == nothing)
            error(" Cant embed without init_fspace")
        end

        #
        # Get subset of integrals living on cluster, ci
        if rdm1a === nothing && rdm1b === nothing
            ints_i = subset(ints, ci.orb_list) 
        else
            ints_i = subset(ints, ci.orb_list, rdm1a, rdm1b) 
        end


        if all( (rdm1a,rdm1b,init_fspace) .!= nothing)
            # 
            # Verify that density matrix provided is consistent with reference fock sectors
            occs = diag(rdm1a)
            occs[ci.orb_list] .= 0
            na_embed = sum(occs)
            occs = diag(rdm1b)
            occs[ci.orb_list] .= 0
            nb_embed = sum(occs)
            verbose == 0 || @printf(" Number of embedded electrons a,b: %f %f", na_embed, nb_embed)
        end
            
        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec)
        end
        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci, T=T) 
        for sec in sectors
            
            #
            # prepare for FCI calculation for give sector of Fock space
            ansatz = FCIAnsatz(length(ci), sec[1], sec[2])
            verbose == 0 || display(ansatz)
            verbose == 0 || flush(stdout)
            
            nr = min(max_roots, ansatz.dim)

            if ansatz.dim < 500 || ansatz.dim == nr 
                #
                # Build full Hamiltonian matrix in cluster's Slater Det basis
                Hmat = build_H_matrix(ints_i, ansatz)
                F = eigen(Hmat)

                basis_i[sec] = Solution(ansatz, Vector{T}(F.values[1:nr]), Matrix{T}(F.vectors[:,1:nr]))
                #display(e)
            else
                #
                # Do sparse build 
                #if ansatz.dim > 3000
                #    display(norm(ints_i.h1))
                #    display(norm(ints_i.h2))
                #end
                basis_i[sec] = solve(ints_i, ansatz, SolverSettings(nroots=nr))
            end
            if verbose > 0
                state=1
                for ei in basis_i[sec].energies
                    @printf("   State %4i Energy: %12.8f %12.8f\n",state,ei, ei+ints.h0)
                    state += 1
                end
                flush(stdout)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#


"""
    compute_cluster_est_basis(ints::InCoreInts, clusters::Vector{MOCluster}; 
        init_fspace=nothing, delta_elec=nothing, verbose=0, max_roots=10, 
        rdm1a=nothing, rdm1b=nothing)

Return a Vector of `ClusterBasis` for each `Cluster`  using the Embedded Schmidt Truncation
- `ints::InCoreInts`: In-core integrals
- `clusters::Vector{MOCluster}`: Clusters 
- `Da`: background density matrix for embedding local hamiltonian (alpha)
- `Db`: background density matrix for embedding local hamiltonian (beta)
- `init_fspace`: list of pairs of (nα,nβ) for each cluster for defining reference space
                 for selecting out only certain fock sectors
- `thresh_schmidt`: the threshold for the EST 
- `thresh_orb`: threshold for the orbital
- `thresh_ci`: threshold for the ci problem
"""
function compute_cluster_est_basis(ints::InCoreInts{T}, clusters::Vector{MOCluster},Da,Db; 
                thresh_schmidt=1e-3, thresh_orb=1e-8, thresh_ci=1e-6,
                do_embedding=true,verbose=0,init_fspace=nothing,delta_elec=nothing,
                est_nr=1, est_max_cycles=200, est_thresh=1e-6, 
                A::Type=FCIAnsatz) where T
#={{{=#
    # initialize output
    cluster_bases = Vector{ClusterBasis{A,T}}()

    for ci in clusters
        verbose == 0 || display(ci)

        # Obtain the schmidt basis
        basis = TPSChem.form_schmidt_basis(ints, ci, Da, Db,thresh_schmidt=thresh_schmidt,
					   eig_nr=est_nr, eig_max_cycles=est_max_cycles, thresh_ci=est_thresh)

        delta_e_i = ()
        if all( (delta_elec,init_fspace) .!= nothing)
            delta_e_i = (init_fspace[ci.idx][1], init_fspace[ci.idx][2], delta_elec[ci.idx])
        end

        
        #
        # Get list of Fock-space sectors for current cluster
        #
        sectors = possible_focksectors(ci, delta_elec=delta_e_i)

        #
        # Loop over sectors and do FCI for each
        basis_i = ClusterBasis(ci) 


        #for (key, value) in basis
        #    basis_i[key] = value
	#    println(key)
	#    display(value)
        #end


        for sec in sectors
            if sec in keys(basis) 
                basis_i[sec] = Solution(FCIAnsatz(length(ci), sec[1], sec[2]), zeros(size(basis[sec],2)), basis[sec])
		#display(basis[sec])
		#st = "fock_"*string(ci.idx)*"_"*string(sec)
		#npzwrite(st, Matrix(basis[sec]))
            else
	    	#println(sec)
            end
        end
        push!(cluster_bases,basis_i)
    end
    return cluster_bases
end
#=}}}=#
    
