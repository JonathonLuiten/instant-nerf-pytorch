# instant-nerf-pytorch

This is **WORK IN PROGRESS**, please feel free to contribute via pull request.

We are trying to make NeRF train super fast in pytorch by using pytorch bindings for Instant-NGP.

## How to get running:
1. Install tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn).
2. Download, install dependencies and run this code.

Code based upon nerf-pytorch (https://github.com/yenchenlin/nerf-pytorch).

Authors:
 - Jonathon Luiten
 - Kangle Deng


## Notes:
Speicify `--backbone ngp` to enable Instant-NGP (already done in `configs/fern_ngp.txt`).