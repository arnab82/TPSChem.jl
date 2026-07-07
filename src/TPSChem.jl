"""
TPSChem: electronic structure in a tensor product state (TPS) basis.

Consolidates the former FermiCG ecosystem (FermiCG, QCBase, InCoreIntegrals,
BlockDavidson, RDM, ActiveSpaceSolvers, ClusterMeanField) into a single package.
Each former package lives on as a submodule of the same name.
"""
module TPSChem

#####################################
# External packages
#
using Compat
using KrylovKit
using LinearAlgebra
using Printf
using IterativeSolvers
using SparseArrays
using LinearOperators
using Krylov
using TimerOutputs
using OrderedCollections
using IterTools
using StaticArrays
using TensorOperations

using ThreadPools
using Distributed
using JLD2
using LinearMaps
using Random

#####################################
# Vendored submodules (former standalone packages), in dependency order
#
include("QCBase/QCBase.jl")
using .QCBase
include("InCoreIntegrals/InCoreIntegrals.jl")
using .InCoreIntegrals
include("BlockDavidson/BlockDavidson.jl")
using .BlockDavidson
include("RDM/RDM.jl")
using .RDM
include("ActiveSpaceSolvers/ActiveSpaceSolvers.jl")
using .ActiveSpaceSolvers
include("ClusterMeanField/ClusterMeanField.jl")

#####################################
# Core (former FermiCG)
#
include("core/Utils.jl")
include("core/hosvd.jl")
include("core/SymDenseMats.jl");

# Local data
include("core/type_ClusterOps.jl")
include("core/type_ClusterBasis.jl")
include("core/type_ClusterSubspace.jl")

#indexing
include("core/type_SparseIndex.jl")
include("core/type_ClusterConfig.jl")
include("core/type_TransferConfig.jl")
include("core/type_FockConfig.jl")
include("core/type_TuckerConfig.jl")
include("core/type_OperatorConfig.jl")
include("core/Indexing.jl")
include("core/build_local_quantities.jl")

include("core/type_AbstractState.jl")
include("core/type_BSstate.jl")
include("core/type_SPTstate.jl")
include("core/type_TPSCIstate.jl")

include("core/type_ClusteredTerm.jl")
include("core/type_ClusteredOperator.jl")

include("core/tucker_inner.jl")
include("core/tucker_build_dense_H_term.jl")
include("core/tucker_contract_dense_H_with_state.jl")
include("core/tucker_form_sigma_block_expand.jl")
include("core/tucker_outer.jl")
include("core/tucker_pt.jl")
include("core/spt.jl")
include("core/spt_helpers.jl")

include("core/tpsci_inner.jl")
include("core/tpsci_matvec_thread.jl")
include("core/tpsci_pt1_wavefunction.jl")
include("core/tpsci_pt2_energy.jl")
include("core/tpsci_outer.jl")
include("core/tpsci_multinode.jl")
include("core/spt_multinode.jl")
include("core/tpsci_sharded_davidson.jl")
include("core/tps_cepa_sharded.jl")
include("core/tpsci_helpers.jl")
include("core/tpsci.jl")

include("core/dense_inner.jl")
include("core/dense_outer.jl")
include("core/spt_variance.jl")
include("core/spt_variance_multinode.jl")

include("core/tpsci_property.jl")
include("core/tpsci_property_multinode.jl")
include("core/absorption_spectrum.jl")

#
#####################################

export RDM
export ClusterMeanField
export InCoreInts
export Molecule
export Atom
export MOCluster
export ClusterBasis
export ClusterSubspace
export ClusteredOperator
export TPSCIstate
export SPTstate
export ClusterConfig
export FockConfig
export TuckerConfig
export n_orb
export add_subspace!
export add_fockconfig!
export expand_each_fock_space!
export compute_cluster_ops_2rdm
export compute_cluster_ops_2rdm_distributed
export add_spinfree_2rdm_ops!
export add_spinfree_2rdm_ops_distributed!
export subspace_product_tucker
export subspace_product_tucker_multinode
export spt_multinode
export compute_spt_sigma_norm_blockwise_distributed
export spt_variance_multinode
export build_compressed_1st_order_state_distributed
export build_sigma_distributed
export compute_expectation_value_distributed
export compute_pt2_energy_distributed
export compute_pt2_energy_blockwise_distributed
export spt_pt2_multinode
export correlation_functions
export compute_1rdm
export compute_1rdm_sf
export compute_1rdm_threaded
export compute_1rdm_sf_threaded
export compute_1rdm_distributed
export compute_1rdm_sf_distributed
export contract_1rdm_property
export compute_1e_property_direct
export compute_1e_property_direct_distributed
export compute_transition_dipoles
export compute_oscillator_strengths
export absorption_spectrum
export print_stick_spectrum
export compute_2rdm
export compute_2rdm_threaded
export compute_2rdm_blas
export compute_2rdm_distributed
export DistributedTPSCIstate
export DistributedClusterOps
export distribute_tpsci_state
export collect_tpsci_state
export copy_sharded_state
export similar_sharded_state
export restrict_to_basis_sharded
export compute_cluster_ops_distributed
export collect_cluster_ops
export cluster_ops_summary
export add_cmf_operators_distributed!
export sharded_state_summary
export destroy!
export ensure_tpsci_multinode_workers!
export open_matvec_distributed
export open_matvec_sharded
export tps_ci_matvec_sharded
export compute_expectation_value_sharded_h0
export compute_pt1_wavefunction_sharded
export tpsci_cepa_solve_sharded
export do_fois_cepa_sharded
export tps_ci_matvec_distributed
export tps_ci_davidson_distributed
export compute_pt1_wavefunction_distributed
export tpsci_ci_multinode
export tps_ci_davidson_sharded
export build_block_h_sharded
export estimate_sharded_H_nnz
export sharded_H_memory_report
export compute_diagonal_sharded
export tps_sharded_cepa_solve
export do_tps_sharded_cepa
end
