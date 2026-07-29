# STENCIL reproduction workload

This is a deterministic pthread implementation added for the CLIP-3D reproduction. The paper identifies a two-dimensional Jacobi five-point `STENCIL` workload but does not publish its source, grid dimensions, iteration count, boundary conditions, initialization, or compiler flags. This implementation must therefore be reported as a reproduction assumption, not as the authors' exact binary.

Each iteration writes a second grid using the center, north, south, east, and west values with coefficient `0.2`, then swaps the source and destination grids. Interior rows are divided contiguously among threads, and two pthread barriers make every iteration a true Jacobi step. Boundaries remain fixed.

The main thread computes partition zero and creates `threads - 1` pthread workers, so `-t 4` means four total execution threads. Deterministic initialization, a finite-value scan, a fixed-boundary check, and a checksum provide repeatable validation.

Defaults are `-n 2048 -i 500 -t 4`. The default is intentionally long enough for an instruction-limited gem5 region. It is not claimed to be the unpublished paper input size.
