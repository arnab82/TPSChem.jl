using TPSChem
using Documenter

pages = [
    "Home" => "index.md",
    "Installation Instructions" => "installation_instructions.md",
    # "Code Basics" => "basics.md",
    # "Grids" => "grids.md",
    # "Problem" => "problem.md",
    # "GPU" => "gpu.md",
    "Examples" => ["cmf.md","fci.md"],
#    "Library" => [
#        "Contents" => "library/outline.md",
#        "Public" => "library/public.md",
#        "Private" => "library/internals.md",
#        "Function index" => "library/function_index.md",
#        ],
"Functions" => Any[
                   #"TPSChem" => "library/TPSChem.md",
                   "CMF" => "library/CMFs.md",
                   "TPSCI" => "library/TPSCI.md",
                   "SPT" => "library/SPT.md",
                   "Internals" => "library/Internals.md",
                   "ClusteredTerms" => "library/ClusteredTerms.md",
                   "States" => "library/States.md",
                   
                   "ActiveSpaceSolvers" => "library/ActiveSpaceSolvers.md",
                   "Utils" => "library/Utils.md",
                  ],
]

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")


examples = [
    "test_cmf.jl",
    "test_fci.jl",
]

# for example in examples
#   example_filepath = joinpath(EXAMPLES_DIR, example)
#   withenv("GITHUB_REPOSITORY" => "FourierFlows/FourierFlowsDocumentation") do
#     example_filepath = joinpath(EXAMPLES_DIR, example)
#     Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
#     Literate.notebook(example_filepath, OUTPUT_DIR, documenter=true)
#     Literate.script(example_filepath, OUTPUT_DIR, documenter=true)
#   end
# end

makedocs(;
    warnonly=true,
    modules=[TPSChem],
    authors="Nick Mayhall <nmayhall@gmail.com> and contributors",
    repo=Documenter.Remotes.GitHub("nmayhall", "TPSChem.jl"),
    sitename="TPSChem",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall.github.io/TPSChem.jl/stable",
        assets=String[],
    ),
    #html_prettyurls = !("local" in ARGS),
    pages=pages,
)

deploydocs(
    repo="github.com/nmayhall/TPSChem.jl.git",
    branch = "gh-pages",
    devbranch = "main",
    #push_preview = true,
    target= "build",
)
