import os
import sys
import torch
import torch.nn as nn
import tinycudann as tcnn

from utils import *


def create_instant_ngp(args):
    """Instantiate instant-NGP nerf model.
    """

    # TODO: Config is hardcoded right now, need to be integrated into args
    hash_encoding_config = {
        "otype": "HashGrid",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 19,
        "base_resolution": 16,
        "per_level_scale": 1.3819
    }
    base_network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1
    }
    dir_encoding_config = {
        "otype": "Composite",
        "nested": [
            {
                "n_dims_to_encode": 3,
                "otype": "SphericalHarmonics",
                "degree": 4
            },
            {
                "otype": "Identity"
            }
        ]
    }
    rgb_network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2
    }
    num_feats_from_base_to_rgb = 15

    # Model
    class Instant_NGP(nn.Module):
        def __init__(self, hash_encoding_config, base_network_config, dir_encoding_config, rgb_network_config,
                     num_feats_from_base_to_rgb):
            super(Instant_NGP, self).__init__()
            self.input_ch = 3
            self.input_ch_views = 3
            self.base_model = tcnn.NetworkWithInputEncoding(n_input_dims=3,
                                                            n_output_dims=1 + num_feats_from_base_to_rgb,
                                                            encoding_config=hash_encoding_config,
                                                            network_config=base_network_config)
            self.rgb_model = tcnn.NetworkWithInputEncoding(n_input_dims=3 + num_feats_from_base_to_rgb,
                                                           n_output_dims=3,
                                                           encoding_config=dir_encoding_config,
                                                           network_config=rgb_network_config)

        def forward(self, x):
            # TODO: Hack to fit the pts into [0,1], should adapt to different bounds
            x = (x+1)/2
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
            base_output = self.base_model(input_pts)
            alpha = base_output[:, 0]
            concat_feats = torch.cat([input_views, base_output[:, 1:]], 1)
            rgb = self.rgb_model(concat_feats)
            out = torch.cat([rgb, alpha[:, None]], 1)
            return out

    model = Instant_NGP(hash_encoding_config, base_network_config, dir_encoding_config, rgb_network_config,
                        num_feats_from_base_to_rgb).to(device)
    grad_vars = list(model.parameters())
    model_fine = None
    if args.N_importance > 0:
        model_fine = Instant_NGP(hash_encoding_config, base_network_config, dir_encoding_config, rgb_network_config,
                                 num_feats_from_base_to_rgb).to(device)
        grad_vars += list(model_fine.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)

    def run_network_instant_NGP(inputs, viewdirs, fn, netchunk=1024 * 256):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        concat_input = torch.cat([inputs_flat, input_dirs_flat], -1)
        outputs_flat = batchify(fn, netchunk)(concat_input)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network_instant_NGP(inputs, viewdirs, network_fn,
                                                                netchunk=args.netchunk)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################


    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
