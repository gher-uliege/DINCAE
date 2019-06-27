using DINCAE
using Random

data = collect(1:10)

sdata = DINCAE.RVec(data)

@show collect(sdata)

shuffle!(sdata)

@show collect(sdata)
