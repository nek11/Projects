#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]



class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, intrinsic, extrinsic, mesh_name


class RGBA2SDF_rototranslation(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)


        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        xyz = sdf_samples[:, 0:3]
        sdf_gt = sdf_samples[:, 3]

        # Only keep 3x3 matrix of roto-translation (take off translation)
        extrinsic_np = extrinsic.numpy()
        #extrinsic_np = extrinsic_np[0:3, 0:3]

        # Define intrinsic as np_array
        intrinsic_np = intrinsic[0].numpy()

        # Convert xyz to numpy
        one = [1]
        xyz_nump = xyz.numpy()

        # Add one in each sdf_sample for extrinsic multiplication
        xyz_np = [np.concatenate((point, one), axis = 0) for point in xyz_nump]
        xyz_np = np.array(xyz_np)


        # 1) map from world to camera coordinates
        xyz_camera = np.dot(xyz_np, extrinsic_np.T)
        # 2) map from camera to image coordinates
        # xyz_image = np.dot(xyz_camera, intrinsic_np.T)

        # Reshape sdf array
        sdf_reshaped = sdf_gt.reshape(len(sdf_gt),1)

        # Put values back in sdf_samples
        sdf_samples_camera_np = np.concatenate((xyz_camera, sdf_reshaped), axis = 1)

        sdf_samples_camera = torch.tensor(sdf_samples_camera_np).float()

        return sdf_samples_camera, RGBA, intrinsic, extrinsic, mesh_name

class RGBA2SDF_reconstruct(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)


        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        # Define intrinsic and extrinsic matrices as np arrays
        # intrinsic_np = intrinsic[0].numpy()
        extrinsic_np = extrinsic.numpy()

        # Comment this line to get translation
        extrinsic_np_rot = extrinsic_np[0:3, 0:3]

        # load ground truth mesh
        mesh_filename = os.path.join(self.data_source, mesh_name.replace("samples", "meshes") + ".obj")
        mesh = trimesh.load(mesh_filename)
        vertices = mesh.vertices
        faces = mesh.faces
        faces_ret = torch.tensor(faces).float()

        # Convert vertices to numpy
        one = [1]
        vertices_nump = np.array(vertices)

        # Add one in each sdf_sample for extrinsic multiplication
        vertices_np_t = [np.concatenate((point, one), axis=0) for point in vertices_nump]
        vertices_np_t = np.array(vertices_np)

        # Rotate mesh vertices, 1st one is with translation, 2nd one without
        #rot_vertices = np.dot(vertices_np, extrinsic_np.T)
        rot_vertices = np.dot(vertices_nump, extrinsic_np_rot.T)
        rot_vertices = torch.tensor(rot_vertices).float()

        # Set rotation of extrinsic as identity matrix (we don't want to rotate again)
        extrinsic_np[0:3, 0:3] = np.eye(3)
        extrinsic_torch = torch.from_numpy(extrinsic_np)

        return sdf_samples, RGBA, intrinsic, extrinsic_torch, mesh_name, rot_vertices, faces_ret