# instant-nerf-pytorch

This is **WORK IN PROGRESS**, please feel free to contribute via pull request.

We are trying to make NeRF train super fast in pytorch by using pytorch bindings for Instant-NGP.

## How to get running:

1. Install both tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn) and nerf-pytorch (https://github.com/yenchenlin/nerf-pytorch).
2. Replace the call to the function 'create_nerf(args)' in nerf-pytorch's run_nerf.py with a call to the 'create_instant_nerf(args)' function in instant_nerf_pytorch.py of this repo.
