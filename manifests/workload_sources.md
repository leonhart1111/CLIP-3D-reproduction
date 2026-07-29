# Workload source provenance

Downloaded on 2026-07-22.

| Workload material | Upstream | Branch head at download | Archive SHA-256 | Installed path |
|---|---|---|---|---|
| SPLASH-2 FFT and CHOLESKY sources | `https://github.com/gem5/gem5-resources` (`stable`) | `0f5945412244a7a777fe40a5db4efc441a9a9f73` | `ef1a800e23b0070c6a9d3f4033f33a1ed0b7477f04f369a94ce4513546e6bd41` | `benchmarks/src/splash2/` |
| McCalpin STREAM | `https://github.com/jeffhammond/STREAM` (`master`) | `6703f7504a38a8da96b353cadafa64d3c2d7a2d3` | `b16e091f2c4c0f685183b6244ea1fc4992c6666502fd6e7380f337f285065883` | `benchmarks/src/stream/` |
| Original SPLASH-2 CHOLESKY inputs mirror | `https://github.com/staceyson/splash2` (`master`) | `be64f8e4840fc29ff4c076c3f747e27fdb55ac68` | `ad40dda40676a7d8309830dea406a22383eface4cc03a7ec11fa2d1949a48de6` | `benchmarks/inputs/cholesky/` |

Actual archive URLs:

- `https://codeload.github.com/gem5/gem5-resources/tar.gz/refs/heads/stable`
- `https://codeload.github.com/jeffhammond/STREAM/tar.gz/refs/heads/master`
- `https://codeload.github.com/staceyson/splash2/tar.gz/refs/heads/master`

## Reproduction notes

- FFT generates its data internally. The PARSEC-packaged SPLASH-2 run configuration uses `-m16` and a configurable processor count.
- CHOLESKY needs a sparse-matrix input. All standard input files from the SPLASH-2 mirror were unpacked, including `tk14.O` and `tk29.O`.
- The translated paper says `tkl4`, but no such SPLASH-2 input exists. This is almost certainly the visually ambiguous spelling of `tk14` (letter `l` versus digit `1`). Preserve this as an explicit reproduction assumption.
- The paper names dense MATMUL and a 2-D Jacobi five-point STENCIL, but does not cite or identify their exact source implementation, input dimensions, iteration counts, compiler flags, or initialization. Clearly labelled pthread reproduction implementations were therefore added locally under `benchmarks/src/matmul/` and `benchmarks/src/stencil/`. Their defaults and algorithm choices are documented in the respective README files and must be reported as reproduction assumptions rather than the authors' exact workloads.
- STREAM retains the upstream 10-million-element array and standard four kernels, but is built with the upstream-supported `NTIMES=20` instead of the default 10. This supplies enough dynamic instructions for the paper's 100-million warmup plus 500-million measured CPU0 window without changing the memory working set. The paper does not publish its STREAM iteration count, so this is a disclosed harness-completion assumption.
