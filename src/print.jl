## Display functions

import Base.show

function show(io::IO, wf::WaveFunctions)
	print(io, "WaveFunctions(")
	print(io, join((@sprintf("%.4f", v/eV) for v in energy(wf)[1:6]), ", "))
	print(io, " ...  eV)\n")
end

function show(io::IO, wf::GroupedWaveFunctions)
	print(io, "GroupedWaveFunctions(")
	print(io, join((@sprintf("%.4f", v/eV) for v in energy(wf)[1:6]), ", "))
	print(io, " ...  eV)\n")
end
