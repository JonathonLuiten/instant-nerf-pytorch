# instant-nerf-pytorch

This is **WORK IN PROGRESS**, please feel free to contribute via pull request.

We are trying to make NeRF train super fast in pytorch by using pytorch bindings for Instant-NGP.

## Current Progress:
 - Code is implemented and runs, but cannot achieve super good results.
 - Per iteration it is ~3.5x faster than the nerf-pytorch code it is built upon.
 - VERY quickly (1 min) gets up to ~20 PSNR.
 - But doesn't really get above ~25 PSNR even when training for a long time.

## How to get running:
1. Install tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn).
2. Download, install dependencies and run this code.

Code based upon nerf-pytorch (https://github.com/yenchenlin/nerf-pytorch).

Authors:
 - Jonathon Luiten
 - Kangle Deng


## Notes:
Speicify `--backbone ngp` to enable Instant-NGP (already done in `configs/fern_ngp.txt`).
