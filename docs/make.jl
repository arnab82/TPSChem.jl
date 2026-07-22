using TPSChem
using Documenter

# ---------------------------------------------------------------------------
# Where / what do we deploy?
#
# The docs deploy to whichever GitHub repo the workflow runs in — the `arnab82`
# fork now, and `nmayhall` automatically once the branch is merged there (no
# deploy key needed: the default GITHUB_TOKEN can push gh-pages of the same
# repo). GITHUB_REPOSITORY ("owner/name") and GITHUB_REF_NAME are set by GitHub
# Actions; both fall back sensibly for local builds (where deploy is a no-op).
#
# Each branch gets its own folder on gh-pages:
#   main -> /dev/ , multinode -> /multinode/ , any other branch -> /<branch>/
# ---------------------------------------------------------------------------
const REPO_SLUG   = get(ENV, "GITHUB_REPOSITORY", "nmayhall/TPSChem.jl")
const _slug_parts = split(REPO_SLUG, "/")
const OWNER       = String(_slug_parts[1])
const REPO_NAME   = String(_slug_parts[end])
const BRANCH      = get(ENV, "GITHUB_REF_NAME", "main")
const DEVURL      = BRANCH == "main" ? "dev" : BRANCH

# ---------------------------------------------------------------------------
# The distributed / sharded ("multinode") source only exists on the multinode
# branch, so the corresponding API page and tutorial are only built when the
# files are actually present in this checkout. This keeps a single make.jl
# working for both branches.
# ---------------------------------------------------------------------------
const CORE          = joinpath(@__DIR__, "..", "src", "core")
const HAS_MULTINODE = isfile(joinpath(CORE, "tpsci_multinode.jl"))
const TUTORIAL_SRC  = joinpath(@__DIR__, "..", "examples", "multinode", "TUTORIAL.md")
const HAS_TUTORIAL  = isfile(TUTORIAL_SRC)

# The multinode tutorial is authored under examples/ (single source of truth);
# copy it into the docs tree so Documenter renders it on the site.
const GENERATED = joinpath(@__DIR__, "src", "generated")
mkpath(GENERATED)
if HAS_MULTINODE && HAS_TUTORIAL
    cp(TUTORIAL_SRC, joinpath(GENERATED, "multinode_tutorial.md"); force=true)
end

# ---------------------------------------------------------------------------
# Page tree
# ---------------------------------------------------------------------------
api_pages = Any[
    "Overview"            => "library/TPSChem.md",
    "QCBase"              => "library/QCBase.md",
    "InCoreIntegrals"     => "library/InCoreIntegrals.md",
    "RDM"                 => "library/RDM.md",
    "BlockDavidson"       => "library/BlockDavidson.md",
    "ActiveSpaceSolvers"  => "library/ActiveSpaceSolvers.md",
    "ClusterMeanField"    => "library/CMFs.md",
    "States & Configs"    => "library/States.md",
    "Clustered Operators" => "library/ClusteredTerms.md",
    "TPSCI"               => "library/TPSCI.md",
    "SPT"                 => "library/SPT.md",
    "RDMs & Properties"   => "library/Properties.md",
    "Utilities"           => "library/Utils.md",
]
if HAS_MULTINODE
    push!(api_pages, "Multinode & Sharded" => "library/Multinode.md")
end
push!(api_pages, "Function Index" => "library/function_index.md")

manual = Any["Methods & Workflows" => "methods.md",
             "Examples" => Any["cmf.md", "fci.md"]]
if HAS_MULTINODE && HAS_TUTORIAL
    push!(manual, "Multinode / distributed" => "generated/multinode_tutorial.md")
end

pages = Any[
    "Home"          => "index.md",
    "Installation"  => "installation_instructions.md",
    manual...,
    "API Reference" => api_pages,
]

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
makedocs(;
    warnonly = true,
    modules  = [TPSChem,
                TPSChem.QCBase,
                TPSChem.InCoreIntegrals,
                TPSChem.RDM,
                TPSChem.BlockDavidson,
                TPSChem.ActiveSpaceSolvers,
                TPSChem.ActiveSpaceSolvers.FCI,
                TPSChem.ClusterMeanField],
    authors  = "Nick Mayhall and collaborators",
    repo     = Documenter.Remotes.GitHub(OWNER, REPO_NAME),
    sitename = "TPSChem.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://$(OWNER).github.io/$(REPO_NAME)/$(DEVURL)",
        assets     = String[],
    ),
    pages    = pages,
)

# ---------------------------------------------------------------------------
# Deploy (no-op outside GitHub Actions)
#
# Deploys the branch currently being built to its own folder on gh-pages of
# whichever repo the workflow runs in:
#   push to main      -> gh-pages:/dev/
#   push to multinode -> gh-pages:/multinode/
#   push to <branch>  -> gh-pages:/<branch>/
#   git tag vX.Y.Z    -> gh-pages:/vX.Y.Z/  (+ /stable/)
# ---------------------------------------------------------------------------
_versions = Any["stable" => "v^", "v#.#", "dev" => "dev", "multinode" => "multinode"]
DEVURL in ("dev", "multinode") || push!(_versions, DEVURL => DEVURL)

deploydocs(
    repo         = "github.com/$(REPO_SLUG).git",
    branch       = "gh-pages",
    devbranch    = BRANCH,
    devurl       = DEVURL,
    versions     = _versions,
    target       = "build",
    push_preview = false,
)
