import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import copy


class SpatialTransformer(nn.Module):
    def __init__(self, input_nc, dataset, loadSize, tOut, ngf, norm_layer=nn.BatchNorm2d):
        # ngf number of generator filters in first conv layer
        """
        :param input_nc: Input channels for each frames times frame number
        :param output_nc: Output channels for predict next frame
        :param ngf:number of generator filters in first conv layer
        :param n_downsampling: 3
        :param n_blocks:
        :param norm_layer:
        :param padding_type:
        """

        super(SpatialTransformer, self).__init__()
        activation = nn.LeakyReLU(True)
        self.tOut = tOut
        # Downsample input data to get features
        encoder = []
        # Conv0
        if loadSize == 1024:
            encoder += [nn.Conv2d(input_nc, ngf, kernel_size=[3, 3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]
            encoder += [nn.Conv2d(ngf, ngf, kernel_size=[3, 3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]
        # Conv1
        else:
            encoder += [nn.Conv2d(input_nc, ngf, kernel_size=[3, 3], stride=1), norm_layer(ngf), activation]
            encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]
        # Conv2
        encoder += [nn.Conv2d(ngf, ngf * 2, kernel_size=[3, 3], stride=1), norm_layer(ngf * 2), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]
        # Conv3
        encoder += [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=[3, 3], stride=1), norm_layer(ngf * 4), activation]
        encoder += [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=[3, 3], stride=1), norm_layer(ngf * 4), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]
        # Conv4
        encoder += [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=[3, 3], stride=1), norm_layer(ngf * 8), activation]
        encoder += [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=[3, 3], stride=1), norm_layer(ngf * 8), activation]
        encoder += [nn.MaxPool2d(kernel_size=[3, 3], stride=2)]

        # Conv5
        encoder += [nn.Conv2d(ngf * 8, 500, kernel_size=[3, 3], stride=1), norm_layer(500), activation]
        encoder += [nn.Conv2d(500, 500, kernel_size=[3, 3], stride=1), norm_layer(500), activation]
        if dataset == 'cityscapes':
            encoder += [nn.AvgPool2d(kernel_size=[8, 16], stride=[8, 16])]
        else:
            encoder += [nn.AvgPool2d(kernel_size=[8, 26], stride=[8, 26])]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc_0 = nn.Sequential(
            nn.Linear(100, 32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc_0[2].weight.data.zero_()
        self.fc_loc_0[2].bias.data.copy_(torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float))

        self.fc_loc_1 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_2 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_3 = copy.deepcopy(self.fc_loc_0)
        self.fc_loc_4 = copy.deepcopy(self.fc_loc_0)
        self.encoder = nn.Sequential(*encoder)

    def theta2affine(self, theta, loadSize):

        # print("theta =", theta.size())
        bs = theta.size()[0]
        size = torch.empty(bs, 2, 3)
        # print(size.type())
        affine_matrix = torch.zeros_like(size).cuda()
        # print(affine_matrix.type())
        sx, sy = theta[:, 0], theta[:, 1]
        rotation, shear = theta[:, 2], theta[:, 3]
        tx, ty = theta[:, 4], theta[:, 5]
        sx = sx.clamp(min=0.6, max=1.4)
        sy = sy.clamp(min=0.6, max=1.4)
        # if loadSize == 1024:
        #    tx*=2
        #    ty*=2
        # sx sy rotation shear tx ty
        affine_matrix[:, 0, 0] = sx * torch.cos(rotation)
        affine_matrix[:, 0, 1] = - sy * torch.sin(rotation + shear)
        affine_matrix[:, 0, 2] = tx
        affine_matrix[:, 1, 0] = sx * torch.sin(rotation)
        affine_matrix[:, 1, 1] = sy * torch.cos(rotation + shear)
        affine_matrix[:, 1, 2] = ty
        affine_matrix = affine_matrix.view(-1, 2, 3)
        return affine_matrix

    def clip_mask(self, mask):
        one_ = torch.ones_like(mask)
        zero_ = torch.zeros_like(mask)
        return torch.where(mask > 0.5, one_, zero_)

    def warp(self, affine_matrix, x, mask):
        grid = F.affine_grid(affine_matrix, x.size())
        t_x = F.grid_sample(x, grid)
        mask_grid = F.affine_grid(affine_matrix, mask.size())
        t_mask = F.grid_sample(mask, mask_grid)
        t_mask = self.clip_mask(t_mask)
        return t_x, t_mask

    # Spatial transformer network forward function
    def stn(self, x, mask, feature, loadSize):
        params = self.encoder(feature)
        # print("params = ", params.size())
        params = params.view(-1, 500)
        # print("params = ", params.size())
        # STN for object timestamp t + 1
        trans_x = []
        trans_mask = []
        affine_matrix = []
        theta_0 = self.fc_loc_0(params[:, :100])
        theta_0 = theta_0.view([-1, 6])
        affine_0 = self.theta2affine(theta_0, loadSize)
        x_0, mask_0 = self.warp(affine_0, x, mask)
        trans_x.append(x_0)
        trans_mask.append(mask_0)
        affine_matrix.append(theta_0)

        theta_1 = self.fc_loc_1(params[:, 100:200])
        theta_1 = theta_1.view([-1, 6])
        affine_1 = self.theta2affine(theta_1, loadSize)
        x_1, mask_1 = self.warp(affine_1, x, mask)
        trans_x.append(x_1)
        trans_mask.append(mask_1)
        affine_matrix.append(theta_1)

        theta_2 = self.fc_loc_2(params[:, 200:300])
        theta_2 = theta_2.view([-1, 6])
        affine_2 = self.theta2affine(theta_2, loadSize)
        x_2, mask_2 = self.warp(affine_2, x, mask)
        trans_x.append(x_2)
        trans_mask.append(mask_2)
        affine_matrix.append(theta_2)

        theta_3 = self.fc_loc_3(params[:, 300:400])
        theta_3 = theta_3.view([-1, 6])
        affine_3 = self.theta2affine(theta_3, loadSize)
        x_3, mask_3 = self.warp(affine_3, x, mask)
        trans_x.append(x_3)
        trans_mask.append(mask_3)
        affine_matrix.append(theta_3)

        theta_4 = self.fc_loc_4(params[:, 400:])
        theta_4 = theta_4.view([-1, 6])
        affine_4 = self.theta2affine(theta_4, loadSize)
        x_4, mask_4 = self.warp(affine_4, x, mask)
        trans_x.append(x_4)
        trans_mask.append(mask_4)
        affine_matrix.append(theta_4)

        return trans_x, trans_mask, affine_matrix

    def forward(self, loadSize, combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped,
                target_back_reshaped, last_object=None, last_mask=None):
        input = torch.cat(
            [combine_reshaped, semantic_reshaped, flow_reshaped, conf_reshaped, mask_reshaped, target_back_reshaped],
            dim=1)
        if last_object is not None:
            last_object = last_object
            last_mask = last_mask
        else:
            last_object = combine_reshaped[:, -3:, :, :] * mask_reshaped[:, -1:, :, :].repeat(1, 3, 1, 1)
            last_mask = mask_reshaped[:, -1:, ...]
        # print("input size = ", input.size())
        warped_x, warped_mask, affine_matrix = self.stn(last_object, last_mask, input, loadSize)
        pred_complete = []
        for i in range(self.tOut):
            cnt_pred_mask = warped_mask[i].repeat(1, 3, 1, 1)
            # if loadSize == 1024:
            #    cnt_pred_complete = target_back_map[:, i, ...] * (1.0 - cnt_pred_mask) + warped_x[i] * cnt_pred_mask
            # else:
            cnt_pred_complete = target_back_reshaped[:, i * 3:(i + 1) * 3, ...] * (1.0 - cnt_pred_mask) + warped_x[
                i] * cnt_pred_mask
            pred_complete.append(cnt_pred_complete)
        return warped_x, warped_mask, affine_matrix, pred_complete
