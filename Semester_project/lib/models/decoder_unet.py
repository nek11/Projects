#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from lib.utils import *

class DISN(nn.Module):
    def __init__(
        self,
        latent_size,
        perceptual_latent_in_size1=(),
        perceptual_latent_in_size2=(),
        perceptual_latent_in_size3=(),
        perceptual_latent_in_size4=(),
        s_global_size = ()
    ):
        super(DISN, self).__init__()
        perceptual_size1 = np.sum(perceptual_latent_in_size1)
        perceptual_size2 = np.sum(perceptual_latent_in_size2)
        perceptual_size3 = np.sum(perceptual_latent_in_size3)
        perceptual_size4 = np.sum(perceptual_latent_in_size4)

        # GLOBAL feature branch
        global_net = []
        global_net.append(nn.Linear(3, 64))
        global_net.append(nn.BatchNorm1d(num_features=64))
        global_net.append(nn.ReLU())
        global_net.append(nn.Linear(64, 256))
        global_net.append(nn.BatchNorm1d(num_features=256))
        global_net.append(nn.ReLU())
        global_net.append(nn.Linear(256, 256))
        global_net.append(nn.BatchNorm1d(num_features=256))
        global_net.append(nn.ReLU())
        self.global_net = nn.Sequential(*global_net)

        global_concat_net = []
        global_concat_net.append(nn.Linear(256+latent_size, 256))
        global_concat_net.append(nn.BatchNorm1d(num_features=256))
        global_concat_net.append(nn.ReLU())
        global_concat_net.append(nn.Linear(256, 256))
        global_concat_net.append(nn.BatchNorm1d(num_features=256))
        global_concat_net.append(nn.ReLU())
        #global_concat_net.append(nn.Linear(256, 1))
        global_concat_net.append(nn.Linear(256, 64))
        self.global_concat_net = nn.Sequential(*global_concat_net)

        """perceptual_concat_net = []
        perceptual_concat_net.append(nn.Linear(512+perceptual_size, 512))
        perceptual_concat_net.append(nn.BatchNorm1d(num_features=512))
        perceptual_concat_net.append(nn.ReLU())
        perceptual_concat_net.append(nn.Linear(512, 256))
        perceptual_concat_net.append(nn.BatchNorm1d(num_features=256))
        perceptual_concat_net.append(nn.ReLU())
        perceptual_concat_net.append(nn.Linear(256, 1))
        self.perceptual_concat_net = nn.Sequential(*perceptual_concat_net)"""

        perceptual_concat1_net = []
        perceptual_concat1_net.append(nn.Linear(256+perceptual_size4+s_global_size, 128))
        """perceptual_concat1_net.append(nn.BatchNorm1d(num_features=256))
        perceptual_concat1_net.append(nn.ReLU())
        perceptual_concat1_net.append(nn.Linear(256, 128))"""
        perceptual_concat1_net.append(nn.BatchNorm1d(num_features=128))
        perceptual_concat1_net.append(nn.ReLU())
        self.perceptual_concat1_net = nn.Sequential(*perceptual_concat1_net)

        perceptual_concat2_net = []
        perceptual_concat2_net.append(nn.Linear(128+perceptual_size3, 64))
        """perceptual_concat2_net.append(nn.BatchNorm1d(num_features=128))
        perceptual_concat2_net.append(nn.ReLU())
        perceptual_concat2_net.append(nn.Linear(128, 64))"""
        perceptual_concat2_net.append(nn.BatchNorm1d(num_features=64))
        perceptual_concat2_net.append(nn.ReLU())
        self.perceptual_concat2_net = nn.Sequential(*perceptual_concat2_net)

        perceptual_concat3_net = []
        perceptual_concat3_net.append(nn.Linear(64+perceptual_size2, 32))
        perceptual_concat3_net.append(nn.BatchNorm1d(num_features=32))
        perceptual_concat3_net.append(nn.ReLU())
        self.perceptual_concat3_net = nn.Sequential(*perceptual_concat3_net)

        perceptual_concat4_net = []
        perceptual_concat4_net.append(nn.Linear(64+perceptual_size1, 32))
        perceptual_concat4_net.append(nn.BatchNorm1d(num_features=32))
        perceptual_concat4_net.append(nn.ReLU())
        perceptual_concat4_net.append(nn.Linear(32, 1))
        self.perceptual_concat4_net = nn.Sequential(*perceptual_concat4_net)

        self.th = nn.Tanh()



    # input: N x (L+3)
    """def forward(self, xyz, global_feature, perceptual_feature):

        # first lift up xyz to high dimensional representation
        xyz_rep = self.global_net(xyz)

        # process global features
        x_global_features = torch.cat((xyz_rep, global_feature), 1)
        s_global = self.global_concat_net(x_global_features)

        # process local features
        x_local_features = torch.cat((xyz_rep, perceptual_feature), 1)
        s_local = self.perceptual_concat_net(x_local_features)

        # finally compute output
        x = self.th(s_global + s_local)
        return x"""

    def forward(self, xyz, global_feature, p_feature_1, p_feature_2, p_feature_3, p_feature_4):



        # first lift up xyz to high dimensional representation
        xyz_rep = self.global_net(xyz)

        # process global features
        x_global_features = torch.cat((xyz_rep, global_feature), 1)
        s_global = self.global_concat_net(x_global_features)

        # process local feature 1
        #x_local_feature_1 = torch.cat((xyz_rep, p_feature_4), 1)
        x_local_feature_1 = torch.cat((xyz_rep, p_feature_4, s_global), 1)
        s_local_1 = self.perceptual_concat1_net(x_local_feature_1)

        #process local feature 2
        x_local_feature_2 = torch.cat((s_local_1, p_feature_3), 1)
        s_local_2 = self.perceptual_concat2_net(x_local_feature_2)

        # process local feature 3
        x_local_feature_3 = torch.cat((s_local_2, p_feature_2), 1)
        s_local_3 = self.perceptual_concat3_net(x_local_feature_3)

        # process local feature 4
        x_local_feature_4 = torch.cat((s_local_3, p_feature_1), 1)
        s_local_4 = self.perceptual_concat4_net(x_local_feature_4)

        # compute output
        #x = self.th(s_local_4 + s_global)
        x = self.th(s_local_4)

        return x



class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = False,
        fourier_degree = 1
    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        else:
            dims = [latent_size + 3] + dims + [1]

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, latent, xyz):

        if self.positional_encoding:
            xyz = fourier_transform(xyz, self.fourier_degree)
        input = torch.cat([latent, xyz.cuda()], dim=1)

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
