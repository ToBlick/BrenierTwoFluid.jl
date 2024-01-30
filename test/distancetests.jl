# compute W_2 distance between two uniform distributions

c = (x,y) -> 0.5 * sqeuclidean(x,y)
∇c = (x,y) -> x - y
d = 3
N = 20^2
M = 30^2
α = ones(N) / N
β = ones(M) / M

d′ = 2*Int(floor(d/2))
ε = N^(-1/(d′+4))
q = 1.0
Δ = 1.0
s = ε
tol = 1e-8
crit_it = 20
p_ω = 2

offset = 0.5

Random.seed!(123)
X = rand(N,d) .- 0.5

for i in 1:2
        if i == 1 # Scaling test
                truevalue = 1/2 * d * 1/12 * offset^2 #1/2 * d * offset^2
                Y = (rand(M,d) .- 0.5) .* (1 + offset)
        else # Shifting test
                truevalue = 1/2 * d * offset^2
                Y = (rand(M,d) .- 0.5) .+ offset
        end

        CC = CostCollection(X, Y, c)
        V = SinkhornVariable(X,α)
        W = SinkhornVariable(Y,β)

        # no acc, no sym
        params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=false)
        S = SinkhornDivergence(V,W,CC,params)
        initialize_potentials!(V,W,CC)
        valueS = compute!(S)
        @test abs(valueS - truevalue) < (sqrt(N*M))^(-2/(d′+4))

        # acc, no sym
        params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=false,acc=true)
        S = SinkhornDivergence(V,W,CC,params)
        initialize_potentials!(V,W,CC)
        valueS = compute!(S)
        @test abs(valueS - truevalue) < (sqrt(N*M))^(-2/(d′+4))

        # acc, sym
        params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=true,acc=true)
        S = SinkhornDivergence(V,W,CC,params)
        initialize_potentials!(V,W,CC)
        valueS = compute!(S)
        @test abs(valueS - truevalue) < (sqrt(N*M))^(-2/(d′+4))

        # no acc, sym
        params = SinkhornParameters(CC;ε=ε,q=1.0,Δ=1.0,s=s,tol=tol,crit_it=crit_it,p_ω=p_ω,sym=true,acc=false)
        S = SinkhornDivergence(V,W,CC,params)
        initialize_potentials!(V,W,CC)
        valueS = compute!(S)
        @test abs(valueS - truevalue) < (sqrt(N*M))^(-2/(d′+4))
end