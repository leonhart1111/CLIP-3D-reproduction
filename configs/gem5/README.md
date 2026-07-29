# CLIP-3D gem5 R1 configuration

`clip_r1.py` models the paper's shared architectural front end: four 2 GHz X86 out-of-order cores, four-wide fetch/decode/rename/dispatch/issue/commit, a 192-entry ROB, private split L1 caches, and one shared L2. Writeback is eight-wide so simultaneous completions from variable-latency units cannot overflow gem5 23.1's short forward time buffer. L1 associativity is fixed at 2 and L2 associativity at 8. The default reference point is 32 kB L1D and 512 kB L2; both sizes are command-line parameters for the paper's 20-point cache sweep.

R1 uses one-cycle L1/L2 tag, data, and response latencies plus a one-cycle L1-to-L2 crossbar. These fixed nominal values are deliberate. The same configuration accepts `--stage R2` and per-cache/per-crossbar latency options; `workflow/r2/build_latency_vector.py` generates those arguments from CACTI and equation (6).

## Correct pthread mapping

The same gem5 `Process` object is assigned to all four one-thread CPU contexts. At initialization, only the first context becomes active. Each Linux `clone()` issued by pthreads then claims one of the three halted contexts. This runs one process with four threads. Creating four Process objects would instead run four unrelated copies and produce invalid statistics.

At the end of every successful measurement, the configuration reads `stats.txt` and requires positive committed-instruction counts for `cpu0` through `cpu3`.

## Short validation run

From the project root:

```bash
tools/src/gem5/build/X86/gem5.opt \
  --outdir=runs/gem5_r1/matmul-smoke3 \
  configs/gem5/clip_r1.py \
  --workload matmul \
  --options='-n 128 -r 20 -t 4' \
  --warmup-insts 2000000 \
  --measure-insts 300000
```

The two-million-instruction smoke warmup is intentional. With a much shorter
window, CPU0 can still be in the dynamic loader or serial array initialization,
so a correct configuration will reject the run because cores 1--3 have not yet
entered the parallel kernel. Paper-scale runs use the 100-million-instruction
warmup.

## Paper-scale reference run

```bash
tools/src/gem5/build/X86/gem5.opt \
  --outdir=runs/gem5_r1/matmul-l1d32-l2-512 \
  configs/gem5/clip_r1.py \
  --workload matmul \
  --l1d-size 32kB \
  --l2-size 512kB \
  --warmup-insts 100000000 \
  --measure-insts 500000000
```

Instruction exits are anchored to CPU0 because gem5 23.1 exposes per-CPU instruction-stop events, not a single cross-core counted event. For these row-balanced workloads, CPU0 is the reproducible phase anchor; the recorded per-core counts must still be inspected. This convention should be reported alongside experimental results.

The five workload defaults are:

| Name | Command arguments | Standard input |
|---|---|---|
| `fft` | `-m16 -p4 -r100` | none |
| `cholesky` | `-p4 -r100` | `benchmarks/inputs/cholesky/tk14.O` |
| `stream` | none; OpenMP environment requests four threads | none |
| `matmul` | `-n 1024 -r 1 -t 4` | none |
| `stencil` | `-n 2048 -i 500 -t 4` | none |

Use `--options='...'`, `--binary`, or `--stdin` to override these defaults. Native timings are not comparable with gem5 results.
