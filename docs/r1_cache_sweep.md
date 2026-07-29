# R1 cache sweep

The R1 design grid contains 100 independently resumable gem5 runs:

- five workloads: FFT, CHOLESKY, STREAM, MATMUL, STENCIL;
- L1D size in `{16, 32, 64, 128}` kB;
- shared L2 size in `{128, 256, 512, 1024, 2048}` kB.

The grid is machine-readable in `configs/experiments/r1_cache_sweep.json`. The runner is `scripts/run_r1_sweep.py`.

## Generate the paper plan without running

```bash
cd /home/zyjiang/Agenticflow/CLIP
python3 scripts/run_r1_sweep.py --profile paper
```

This creates `runs/architecture_sweep/r1/paper/planned_jobs.json` and a 100-row `summary.csv`. It does not consume gem5 simulation time.

## Execute or resume

```bash
python3 scripts/run_r1_sweep.py \
  --profile paper \
  --execute \
  --jobs 1
```

Successful points are skipped on later invocations. Use `--rerun` only when results must be replaced. Each point owns its output directory and contains:

- `command.json`: exact argv and architectural point;
- `r1_metadata.json`, `config.ini`, `config.json`: materialized gem5 configuration;
- `stdout.log`, `stderr.log`: simulator logs;
- `stats.txt`: measured region only;
- `status.json`: state, runtime, four-core counts, per-core IPC and aggregate IPC.

The sweep root contains `summary.csv` and `summary.json`. A point is successful only if gem5 exits normally and all four cores have positive measured committed-instruction counts.

## Small end-to-end check

```bash
python3 scripts/run_r1_sweep.py \
  --profile smoke \
  --workloads matmul \
  --l1d-sizes 32kB \
  --l2-sizes 512kB \
  --execute
```

The `paper` profile uses the paper's 100-million-instruction CPU0 warmup and 500-million-instruction CPU0 measurement. The `smoke` profile is only a pipeline validation and must never be included in paper tables.

The explicit `paper_all_cores` profile implements the literal alternative in which every core must reach both instruction targets. It waits for four distinct gem5 instruction-stop events and then verifies every measured core has at least 500 million instructions. Never mix `paper` and `paper_all_cores` points in one sweep: the currently running formal sweep uses the homogeneous `paper`/CPU0 policy.

## FFT reproduction note

The unmodified SPLASH-2 FFT `-m16 -p4` run commits about 27.23 million
instructions on CPU0 and about 6.24 million on each worker, so one invocation
cannot cover the paper's 100-million warmup plus 500-million measurement
window.  R1 therefore runs `-m16 -p4 -r100`: the problem size and working set
remain the SPLASH-2 base case, while the same parallel FFT is repeated enough
times to fill the instruction-limited region.  This is a disclosed harness
completion because the paper does not publish its repetition mechanism.

Two source-level compatibility fixes are also applied in
`benchmarks/src/splash2/kernels/fft/src/fft.C`:

- per-thread roots are allocated before `pthread_create`, avoiding a gem5 23.1
  SE/glibc `brk` mapping fault caused by concurrent post-clone allocation;
- the timing arrays declared as `long *` are allocated with `sizeof(long)`
  instead of the upstream `sizeof(int)`.

The FFT arithmetic and measured-region memory accesses are unchanged.  The
native inverse-transform check passes after these fixes.

## CHOLESKY reproduction note

The paper names the SPLASH-2 `tk14.O` input. One unmodified factorization
commits about 28.92 million instructions on CPU0 and 6.88--6.99 million on
each worker, which is also shorter than the paper instruction window. The
previously unused upstream `iters` variable is exposed as `-r`; R1 uses
`-p4 -r100`. Symbolic analysis and matrix blocking run once. Before the first
numeric factorization, all domain and block numeric values are snapshotted;
before each later iteration they are restored, and the task queues and update
counters are rebuilt. Thus every iteration factors the same `tk14` numeric
matrix and retains its sparse access structure.

## STREAM reproduction note

The upstream STREAM defaults to a 10-million-element array and `NTIMES=10`.
On the four-core gem5 configuration used here, that binary exits after about
519.26 million CPU0 instructions in total. It can complete the 100-million
instruction warmup, but supplies only about 419.26 million of the required
500-million-instruction measured region.

The reproduction build therefore keeps `STREAM_ARRAY_SIZE=10000000` and sets
the upstream-supported compile-time parameter `NTIMES=20`. This preserves the
228.9-MiB three-array working set and the standard Copy, Scale, Add, and Triad
kernels; it only repeats those kernels long enough for gem5's instruction-stop
event to close the formal R1 window. The exact build defaults are recorded in
`benchmarks/src/stream/Makefile`, and the installed binary checksum is recorded
in `manifests/workload_binaries.sha256`. This repetition choice is a disclosed
harness-completion assumption because the paper does not publish its STREAM
iteration count.

## gem5 23.1 and glibc memcpy compatibility

This host glibc reads `__x86_shared_non_temporal_threshold` from CPU/cache
information. Under gem5 23.1 SE it remained zero. An instruction trace showed
a valid 7,228-byte `memcpy` (`RDX=0x1c3c`) being sent into glibc's loop that
assumes at least one 16-KiB non-temporal block; that loop then accessed an
unmapped page. Every R1 process therefore receives:

```text
GLIBC_TUNABLES=glibc.cpu.x86_non_temporal_threshold=1073741824
```

This keeps benchmark-sized copies on the ordinary SSE2 path, as on the native
host, and avoids modifying gem5's `brk`/VMA semantics. The setting and exact
workload argv are recorded in each point's `r1_metadata.json`.

## Measured cost before launching the full grid

The validated MATMUL smoke point simulated 2.3 million CPU0 instructions (2.0 million warmup plus 0.3 million measured) in 48.46 host seconds. A linear projection for 600 million CPU0 instructions is about 3.51 hours per MATMUL point, or about 70 hours for its 20 cache points. STREAM's large working set was materially slower in validation. The complete 100-point sweep should therefore be treated as a multi-day shared-node job, even with moderate parallelism; the projection is recorded in `runs/architecture_sweep/r1/smoke/calibration.json`.

The R1 configuration uses 2 GiB of simulated physical memory; all five
workloads have much smaller working sets, while 8 GiB caused avoidable mmap
failures on a memory-committed shared node. A one-instruction
`/usr/bin/time -v` check measured about 107 MiB maximum RSS with the earlier
8-GiB mapping; resident memory then grows only with touched benchmark pages.
Do not increase `--jobs` merely to reduce wall time without checking node
policy, live RSS, CPU load, and available memory. Every completed point is
resumable.
