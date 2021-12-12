
# Enable import from parent package
import sys
import os
import torch.nn as nn
import torch
import math
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader,Dataset
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', '--e',type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=100,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

# changed this to path to input folder of xyzns
p.add_argument('--point_cloud_path', type=str, default='./data/deepsdf',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--checkpoint_path', '--c',default=None, help='Checkpoint to trained model.')
p.add_argument('--latent_size',default=256,type=int)
opt = p.parse_args()





#=================================================================
# New DataSet class for deepsdf-siren
class PointClouds(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()
        
        self.normals=[]
        self.coords=[]
        self.shapecnt=0

        self.on_surface_points = on_surface_points

        for file in os.listdir(pointcloud_path):
            if not file.endswith('.xyzn'):
                continue
            
            self.shapecnt+=1

            print("Loading point cloud for {}".format(file))
            point_cloud=np.genfromtxt(os.path.join(pointcloud_path,file))

            coords=point_cloud[:,:3]
            normals=point_cloud[:,3:]

            # TODO: might wanna tweak this for ours
            # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
            # sample efficiency)
            coords -= np.mean(coords, axis=0, keepdims=True)
            if keep_aspect_ratio:
                coord_max = np.amax(coords)
                coord_min = np.amin(coords)
            else:
                coord_max = np.amax(coords, axis=0, keepdims=True)
                coord_min = np.amin(coords, axis=0, keepdims=True)

            coords = (coords - coord_min) / (coord_max - coord_min)
            coords -= 0.5
            coords *= 2.

            self.coords.append(coords)
            self.normals.append(normals)
            if self.shapecnt==20:
                return

    def __len__(self):
        return self.shapecnt


    def __getitem__(self, idx):
        # idx should indicate shape index
        coords=self.coords[idx]
        normals=self.normals[idx]

        point_cloud_size = coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = coords[rand_idcs, :]
        on_surface_normals = normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return idx, {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

#=================================================================

latent_size=opt.latent_size
in_features=latent_size+3

# Define the model.
model = modules.DeepSDF(type=opt.model_type, in_features=in_features)
model.cuda()


# =========================================================
# TODO: enable multi gpu support
print("training with {} GPU(s)".format(torch.cuda.device_count()))
#model=nn.DataParallel(model)

#num_samp_per_scene = specs["SamplesPerScene"]
#scene_per_batch = specs["ScenesPerBatch"]

sdf_dataset = PointClouds(opt.point_cloud_path, on_surface_points=opt.batch_size)
num_scenes=len(sdf_dataset)
dataloader = DataLoader(sdf_dataset, shuffle=False, batch_size=num_scenes, pin_memory=True, num_workers=0)

num_scenes=len(sdf_dataset)
code_bound=1.0
code_initstd=1.0

latvecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
torch.nn.init.normal_(
    latvecs.weight.data,
    0.0,
    code_initstd / math.sqrt(latent_size),
)
latvecs.cuda()


# Define the loss
loss_fn = loss_functions.deepsdf
summary_fn = None#utils.write_multi_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train_deepsdf(model=model, latvecs=latvecs,train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
