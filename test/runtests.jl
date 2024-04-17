using SafeTestsets

@safetestset "Costs                                                                           " begin include("costtests.jl") end
@safetestset "Distances                                                                       " begin include("distancetests.jl") end
# @safetestset "Gradients                                                                       " begin include("gradienttests.jl") end
@safetestset "Barycenters                                                                     " begin include("barycentertests.jl") end
