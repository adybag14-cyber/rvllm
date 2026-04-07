Recovered from the original H100 instance on 2026-04-06/07 after tracing the
`N=64` fast path regression to the runtime-loaded CUTLASS shared object.

Source on remote:
`/workspace/runs/4c9154b16-main/kernels/sm_90/libcutlass_kernels.so`

SHA256:
`37451058c78153ac826aa90a639cd6a3f00e2537c271edd5d31a5b033194cd04`

Observed effect on the original H100 with the same binary:
- clean cwd without this `.so`: about `8498 tok/s`
- fast cwd or clean cwd with this `.so`: about `9844-9859 tok/s`
