'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)
p.add_argument('--latvec_cnt',type=int, default=2)
p.add_argument('--explicit_latvec',type=float,default=None)

opt = p.parse_args()


class MultiSDFDecoder(torch.nn.Module):
    def __init__(self,latvec_size=1):
        super().__init__()
        # Define the model.
        in_features=3+latvec_size
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=in_features)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=in_features)
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']


sdf_decoder = MultiSDFDecoder()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

epoch=opt.checkpoint_path.split('/')[-1].split('.')[0].split('_')[2]

sdf_meshing.create_multi_mesh(sdf_decoder, shapecnt=opt.latvec_cnt,outfile_prefix=os.path.join(root_path, 'epoch_{}_res_{}'.format(epoch,opt.resolution)), N=opt.resolution)
