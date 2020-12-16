import functools
import os
import time
from typing import Optional, Callable

import torch
import pytorch_lightning as pl
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.Generator import Generator
from models.SequenceDiscriminator import SequenceDiscriminator
import torch.nn as nn
import data.dataset as data

from options.base_options import BaseOptions
from options.test_options import TestOptions
from options.train_options import TrainOptions


def create_model(opt):
    modelG = Generator(opt)
    modelD = None
    if opt.isTrain:
        modelD = SequenceDiscriminator(opt)

        # from .pwcnet import PWCNet
        # flowNet = PWCNet()

        # if opt.isTrain and len(opt.gpu_ids):
        #     # flowNet.initialize(opt)
        #     if opt.n_gpus_gen == len(opt.gpu_ids):
        #         modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        #         modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
        #         # flowNet = nn.DataParallel(flowNet, device_ids=opt.gpu_ids)
        #     else:
        #         if opt.batchSize == 1:
        #             gpu_split_id = opt.n_gpus_gen + 1
        #             modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1])
        #         else:
        #             gpu_split_id = opt.n_gpus_gen
        #             modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id])
        #         modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids[gpu_split_id:] + [opt.gpu_ids[0]])
        #         # flowNet = nn.DataParallel(flowNet, device_ids=[opt.gpu_ids[0]] + opt.gpu_ids[gpu_split_id:])
        return [modelG, modelD]
    else:
        # flowNet.initialize(opt)
        return [modelG]


class NightCity(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.modelG, self.modelD = create_model(opt)
        self.opt = opt
        self.tIn = opt.tIn
        self.tOut = opt.tOut

    def forward(self):
        pass

    # added optimizer_idx needed for multiple optimizers TODO: understand it
    def training_step(self, batch, batch_idx, optimizer_idx):
        # TODO: tensorboard
        # total_steps += opt.batchSize
        # # print("idx = ", idx)
        # save_fake = total_steps % opt.display_freq == 0
        # if total_steps % opt.print_freq == 0:
        #     iter_start_time = time.time()

        # TODO: implement input processing in the data loader
        # _, n_frames_total, height, width = batch['Image'].size()  # n_frames_total = n_frames_load * n_loadings + tG - 1
        # input_image = Variable(batch['Image'][:, :tIn * 3, ...]).view(-1, tIn, 3, height, width)
        # input_semantic = Variable(batch['Semantic'][:, :tIn * semantic_nc, ...]).view(-1, tIn, semantic_nc, height,
        #                                                                              width)
        # input_combine = Variable(batch['Combine'][:, :tIn * image_nc, ...]).view(-1, tIn, image_nc, height, width)
        # input_flow, input_conf = compute_flow(input_image, tIn, flowNet)
        # target_back_map = Variable(batch['Back'][:, tIn * image_nc:(tIn + tOut) * image_nc, ...]).view(-1, tOut,
        #                                                                                               image_nc,
        #                                                                                               height, width)
        # input_mask = Variable(batch['Mask'][:, :tIn * 1, ...]).view(-1, tIn, 1, height, width)
        # last_object = Variable(batch['LastObject']).view(-1, 3, height, width)
        #
        # ### Label for loss here
        # label_combine = Variable(batch['Combine'][:, tIn * image_nc:(tIn + tOut) * image_nc, ...]).view(-1, tOut,
        #                                                                                                image_nc,
        #                                                                                                height,
        #                                                                                                width)
        # label_mask = Variable(batch['Mask'][:, (tIn) * 1:(tIn + tOut) * 1, ...]).view(-1, tOut, 1, height, width)

        # TODO: check how to avoid this part using pytorch lightning
        # input_semantic = input_semantic.float().cuda()
        # input_combine = input_combine.float().cuda()
        # target_back_map = target_back_map.float().cuda()
        # input_mask = input_mask.float().cuda()
        # last_object = last_object.float().cuda()
        # label_combine = label_combine.float().cuda()
        # label_mask = label_mask.float().cuda()

        tIn = self.tIn
        tOut = self.tOut

        input_combine, input_semantic, input_flow, input_conf, target_back_map, \
        input_mask, last_object, label_combine, label_mask = batch

        warped_object, warped_mask, affine_matrix, pred_complete = self.modelG(input_combine, input_semantic,
                                                                               input_flow,
                                                                               input_conf, target_back_map, input_mask,
                                                                               last_object)
        losses = self.modelD(0, [warped_object, warped_mask, affine_matrix, pred_complete, label_combine, label_mask])

        real_sequence, fake_sequence = self.modelD.module.gen_seq(input_mask, warped_mask, label_mask, tIn, tOut)
        losses_T = self.modelD(1, [real_sequence, fake_sequence])

        losses = [torch.mean(x) if x is not None else 0 for x in losses]
        losses_T = [torch.mean(x) if x is not None else 0 for x in losses_T]
        loss_dict = dict(zip(self.modelD.module.loss_names, losses))
        loss_dict_T = dict(zip(self.modelD.module.loss_names_T, losses_T))

        # collect losses
        loss_G, loss_D, loss_D_T = self.modelD.module.get_losses(loss_dict, loss_dict_T)

        # TODO: tensorboard
        # ### display output images
        # if save_fake:
        #     # print("pred_semantic", pred_semantic[0].size())
        #     visual_list = []
        #     visual_list.append(('last_object', util.tensor2im(last_object[0, ...])))
        #     for k in range(tOut):
        #         visual_list.append(('pred_combine_%02d' % k, util.tensor2im(pred_complete[k][0, ...])))
        #         visual_list.append(('label_combine_%02d' % k, util.tensor2im(label_combine[0, k, ...])))
        #         visual_list.append(('label_mask_%02d' % k, util.tensor2mask(label_mask[0, k, ...])))
        #         visual_list.append(('target_back_%02d' % k, util.tensor2im(target_back_map[0, k, ...])))
        #         visual_list.append(('pred_mask_%02d' % k, util.tensor2mask(warped_mask[k][0, ...])))
        #     for k in range(tIn):
        #         visual_list.append(('input_combine_%02d' % k, util.tensor2im(input_combine[0, k, ...])))
        #         visual_list.append(('input_mask_%02d' % k, util.tensor2mask(input_mask[0, k, ...])))
        #         visual_list.append(('input_semantic_%02d' % k, \
        #                             util.tensor2label(input_semantic[0, k, ...], opt.semantic_nc)))
        #     visuals = OrderedDict(visual_list)
        #     visualizer.display_current_results(visuals, epoch, total_steps)

        # TODO: think if we want to check memory usage, and implement better
        # if opt.debug:
        #     call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        # TODO: tensorboard
        # ### print out errors
        # errors = {k: v.data.item() for k, v in loss_dict.items()}
        # loss_cnt = 0
        # for k, v in sorted(errors.items()):
        #     all_loss[idx, loss_cnt] = v
        #     loss_cnt += 1
        # if total_steps % opt.print_freq == 0:
        #     t = (time.time() - iter_start_time) / opt.print_freq
        #     # errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        #     errors = {k: v.data.item() for k, v in loss_dict.items()}
        #     visualizer.print_current_errors(epoch, epoch_iter, errors, t, all_loss)
        #     visualizer.plot_current_errors(errors, total_steps)

        # TODO: set saving of models
        # ### save latest model
        # if total_steps % opt.save_latest_freq == 0:
        #     visualizer.vis_print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        #     modelG.module.save('latest')
        #     modelD.module.save('latest')
        #     np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    def configure_optimizers(self):
        optimizer_G = self.modelG.optimizer_G
        optimizer_D_T = self.modelD.optimizer_D_T
        scheduler_G = LambdaLR(optimizer_G, lr_lambda=self.modelG.update_learning_rate)
        scheduler_D_T = LambdaLR(optimizer_D_T, lr_lambda=self.modelD.update_learning_rate)
        return [optimizer_G, optimizer_D_T], [scheduler_G, scheduler_D_T]


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


def cut_channel(x):
    return x[:6, ::]


if __name__ == '__main__':
    # ------------some meta data to change imported from our mocogan setup------------
    n_channels = 6
    video_batch = 6
    mean_tuple = tuple([0.5] * 6)
    std_tuple = tuple([0.5] * 6)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        cut_channel,
        transforms.Normalize(mean_tuple, std_tuple),
    ])
    video_transforms = functools.partial(video_transform, image_transform=image_transforms)
    data_folder = './data_example/doom/'
    # ------------------end of meta data----------------------

    # old config tool
    opt = TrainOptions().parse()
    # creating the part of the model
    model = NightCity(opt)
    # datasets Loader set up TODO: arg parser for folder of data
    dataset = data.VideoFolderDataset(data_folder, cache=os.path.join(data_folder, 'local.db'))
    video_dataset = data.VideoDataset(dataset, 16, 2, video_transforms)
    video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    dataloader = video_loader  # TODO: add a dataloader
    # pytorch lightning trainer for training the model
    trainer = pl.Trainer(min_epochs=10)
    trainer.fit(model, dataloader)
