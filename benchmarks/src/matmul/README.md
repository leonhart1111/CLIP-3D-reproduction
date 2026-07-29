# MATMUL reproduction workload

This is a deterministic pthread implementation added for the CLIP-3D reproduction. The paper describes only a dense `MATMUL` workload and does not publish its source, loop ordering, data type, dimensions, iteration count, or compiler flags. This implementation must therefore be reported as a reproduction assumption, not as the authors' exact binary.

The kernel computes double-precision `C = A x B` using an `i-k-j` loop order. Rows of `C` are divided contiguously among threads. The main thread computes partition zero and creates `threads - 1` pthread workers, so `-t 4` means four total execution threads rather than four workers plus a coordinator.

Deterministic initialization and five long-double reference samples provide a correctness check. A checksum makes the result observable. Defaults are `-n 1024 -r 1 -t 4`; all three values can be changed on the command line.

The default size is intentionally long enough for an instruction-limited gem5 region. It is not claimed to be the unpublished paper input size.
