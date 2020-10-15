## Display functions

import Base.show

function show(io::IO, wf::WaveFunctions)
    print(io, "WaveFunctions: $(typeof(wf.functionset))")
    print(io, "\n")
	print(io, "\nN: $(length(wf.energy))")
	print(io, "\nenergy: $(wf.energy[1:8]/eV) eV")
	print(io, "\n")
end

function show(io::IO, wf::GroupedWaveFunctions)
	print(io, "GroupedWaveFunctions with $(typeof(wf.wavefunctions.functionset))")
	print(io, "\n\tN: $(length(wf.energy)) ($(length(wf.wavefunctions.energy)))")
	print(io, "\n\teigenvalues: $(wf.energy[1:8]/eV) eV")
	print(io, "\n")
end
