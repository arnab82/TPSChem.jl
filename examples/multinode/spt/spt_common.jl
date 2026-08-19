# Shared setup for the SPT sharding benchmarks in this directory.
#
# The input JLD2 needs the usual keys (`ints`, `clusters`, `d1`, `init_fspace`,
# `cluster_bases`).  A saved SPT state is used when present, otherwise a
# reference is built from the CMF start and grown once so the benchmark runs on
# a genuinely multi-Fock state rather than a single sector.

using LinearAlgebra, Printf, JLD2, TPSChem, TPSChem.QCBase

"""
    spt_reference_state(data) -> SPTstate

Reuse `data["spt_vec"]` / `data["ref_vec"]` if it is already an `SPTstate`;
otherwise build one from `init_fspace` and grow it by one FOIS expansion so the
benchmark has many Fock sectors to shard.

Environment:
  TPSCHEM_NROOTS        number of roots (default 4)
  TPSCHEM_GROW_THRESH   FOIS threshold used to grow the reference (default 1e-3)
  TPSCHEM_SPT_KEY       key to read a pre-built SPTstate from
"""
function spt_reference_state(data)
    key = get(ENV, "TPSCHEM_SPT_KEY", "spt_vec")
    for k in (key, "ref_vec", "ci_vector")
        if haskey(data, k) && data[k] isa TPSChem.SPTstate
            @printf("Using pre-built SPT state from key '%s'\n", k)
            return data[k]
        end
    end

    nroots = parse(Int, get(ENV, "TPSCHEM_NROOTS", "4"))
    thresh = parse(Float64, get(ENV, "TPSCHEM_GROW_THRESH", "1e-3"))
    nbody  = parse(Int, get(ENV, "TPSCHEM_NBODY", "4"))
    clusters      = data["clusters"]
    cluster_bases = data["cluster_bases"]
    ints          = data["ints"]
    d1            = data["d1"]

    ch = TPSChem.extract_ClusteredTerms(ints, clusters)
    co = TPSChem.compute_cluster_ops(cluster_bases, ints)
    TPSChem.add_cmf_operators!(co, cluster_bases, ints, d1.a, d1.b)

    @printf("Building SPT reference (nroots=%d, grow thresh=%.1e)\n", nroots, thresh)
    v = TPSChem.SPTstate(clusters, FockConfig(data["init_fspace"]), cluster_bases)
    x = TPSChem.build_compressed_1st_order_state(v, co, ch; nbody=nbody, thresh=thresh)
    x = TPSChem.compress(x, thresh=thresh)
    TPSChem.nonorth_add!(v, x)
    v = TPSChem.SPTstate(v, R=nroots)
    TPSChem.randomize!(v)
    TPSChem.orthonormalize!(v)
    @printf("  reference: dim=%d  Fock sectors=%d\n", length(v), length(v.data))
    return v
end

"""
    spt_operators(data) -> (clustered_ham, cluster_ops)
"""
function spt_operators(data)
    ch = TPSChem.extract_ClusteredTerms(data["ints"], data["clusters"])
    co = TPSChem.compute_cluster_ops(data["cluster_bases"], data["ints"])
    TPSChem.add_cmf_operators!(co, data["cluster_bases"], data["ints"],
                               data["d1"].a, data["d1"].b)
    return ch, co
end

state_mb(s) = sum(sum(sizeof(c) for c in t.core) + sum(sizeof(f) for f in t.factors)
                  for (_, cfgs) in s.data for (_, t) in cfgs; init=0) * 1e-6
