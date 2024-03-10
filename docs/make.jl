using Documenter
using BrenierTwoFluid

# this is necessary to avoid warnings. See https://documenter.juliadocs.org/dev/man/syntax/
ENV["GKSwstype"] = "100"

# if the docs are generated with github actions, then this changes the path; see: https://github.com/JuliaDocs/Documenter.jl/issues/921 
const buildpath = haskey(ENV, "CI") ? ".." : ""

const html_format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", nothing) == "true",
    repolink = "https://github.com/ToBlick/BrenierTwoFluid.jl",
    canonical = "https://toblick.github.io/BrenierTwoFluid.jl",
    assets = [
        "assets/extra_styles.css",
        ],
    # specifies that we do not display the package name again (it's already in the logo)
    sidebar_sitename = false,
    )

const latex_format = Documenter.LaTeX()

const output_type = isempty(ARGS) ? :html : ARGS[1] == "latex_output" ? :latex : :html

const format = output_type == :latex ? latex_format : html_format

makedocs(;
    modules = [BrenierTwoFluid],
    authors = "Tobias Blickhan",
    repo = "https://github.com/ToBlick/BrenierTwoFluid.jl/blob/{commit}{path}#L{line}",
    sitename = "BrenierTwoFluid.jl",
    format = format,
    pages=[
        "Home" => "index.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo   = "https://github.com/ToBlick/BrenierTwoFluid.jl",
    devurl = "latest",
    devbranch = "main",
)