using TPSChem
using Test
using Random

Random.seed!(1234567)

@testset "TPSChem" begin
    include("test_Clusters.jl")
    include("test_FCI.jl")
    include("test_TDMs.jl")
    include("test_tpsci.jl")
    include("test_bounds.jl")
    include("test_s2.jl")
    include("test_hosvd.jl")
    include("test_spt.jl")
    include("test_bs.jl")
    include("test_schmidt.jl")
    include("test_openshell.jl")
    include("test_qdpt.jl")
    include("test_variance.jl")
    include("test_tpsci_rdm2.jl")
    include("test_tpsci_rdm_bruteforce.jl")
    include("test_tpsci_rdm_invariants.jl")
    include("test_tpsci_rdm_threaded.jl")
end

