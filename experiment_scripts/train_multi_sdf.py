
# Enable import from parent package
import sys
import os
import torch.nn as nn
import torch
import numpy as np
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader,Dataset
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
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
p.add_argument('--point_cloud_path', type=str, default='./data/multi',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()

#=================================================================
# New DataSet class for multi shapes
class PointClouds(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()
        
        self.normals=[]
        self.coords=[]
        self.len_samples=[]
        shapecnt=0

        self.on_surface_points = on_surface_points

        for file in os.listdir(pointcloud_path):
            if not file.endswith('.xyzn'):
                continue
            
            shapecnt+=1

            print("Loading point cloud for {}".format(file))
            point_cloud=np.genfromtxt(os.path.join(pointcloud_path,file))
            print("Done loading")

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
            self.len_samples.append(len(coords)//self.on_surface_points)
        

    def __len__(self):
        return sum(self.len_samples)


    def __getitem__(self, idx):
        # idx should indicate shape index
        cnt=0
        for i,sz in enumerate(self.len_samples):
            cnt+=sz
            if idx<cnt:
                idx=i
                break

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

        #===================================================================
        on_surface_coords= np.concatenate(([[i]]*self.on_surface_points,on_surface_coords),axis=1)
        off_surface_coords= np.concatenate(([[i]]*off_surface_samples,off_surface_coords),axis=1)
        #===================================================================

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
      
        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

#=================================================================

sdf_dataset = PointClouds(opt.point_cloud_path, on_surface_points=opt.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=2, pin_memory=True, num_workers=0)

latvec_size=1
in_features=latvec_size+3
# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=in_features)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=in_features)
model.cuda()

# Define the loss
loss_fn = loss_functions.multi_sdf
summary_fn = None#utils.write_multi_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
