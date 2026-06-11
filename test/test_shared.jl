tuckA = TPSChem.Tucker(rand(12,11,1,1,1,10,1,1,3,10,2,1,1,10,1,1,1,1,1,1));
tuckB = TPSChem.Tucker(rand(12,11,1,1,1,10,1,1,3,10,2,1,1,10,1,1,1,1,1,1));

# tuckA = TPSChem.compress(tuckA)
# tuckB = TPSChem.compress(tuckB)
scr = Vector{Vector{Float64}}([Vector{Float64}([]) for i in 1:ndims(tuckA)]);
@timev TPSChem.nonorth_add([tuckA, tuckB]);
@timev TPSChem.nonorth_add([tuckA, tuckB],scr);
@btime TPSChem.nonorth_add([tuckA, tuckB]);
@btime TPSChem.nonorth_add([tuckA, tuckB],scr);