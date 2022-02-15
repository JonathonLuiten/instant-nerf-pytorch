# instant-nerf-pytorch

This is **WORK IN PROGRESS**, please feel free to contribute via pull request.

We are trying to make NeRF train super fast in pytorch by using pytorch bindings for Instant-NGP.

## Current Progress:
 - Code is implemented and runs, but cannot achieve super good results.
 - Per iteration it is ~3.5x faster than the nerf-pytorch code it is built upon.
 - VERY quickly (1 min) gets up to ~20 PSNR (this is MUCH faster, even in iteration count than the normal NeRF).
 - But doesn't really get above ~25 PSNR even when training for a long time.
 - There is a bug where running the 'fine' network doesn't work, results above are only for coarse network (e.g. N_importance = 0), and speed comparisons also to only course network, this might be the reason the PSNR doesn't get super high.

## How to get running:
1. Install tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn).
2. Download, install dependencies and run this code.

## Links to sources:
 - The code is based upon nerf-pytorch (https://github.com/yenchenlin/nerf-pytorch).
 - We are trying to make this faster, by replacing the torch MLP and Pos_enc with those from tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn).
 - This should then become a pytorch version of the (cuda) code for Instant-NGP (https://github.com/NVlabs/instant-ngp)

## Authors:
 - Jonathon Luiten
 - Kangle Deng


## Notes:
Speicify `--backbone ngp` to enable Instant-NGP (already done in `configs/fern_ngp.txt`).
