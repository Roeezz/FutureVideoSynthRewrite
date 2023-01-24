import functools
import os

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pytorch_lightning import loggers as pl_loggers
import fore.data.dataset as data
from fore.lightningModels.Generator import Generator
from fore.lightningModels.SequenceDiscriminator import SequenceDiscriminator
from fore.options.train_options import TrainOptions


class NightCity(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.modelG = Generator(opt)
        self.modelD = SequenceDiscriminator(opt)
        self.opt = opt
        self.tIn = opt.tIn
        self.tOut = opt.tOut
        self.modelG_out = None
        self.loss_G = None
        self.loss_D_T = None

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
        tensorboard = self.logger.experiment

        tIn = self.tIn
        tOut = self.tOut
        optimizer_G, optimizer_D_T = self.optimizers()
        input_combine, input_semantic, input_flow, input_mask, label_combine, label_mask = batch.values()

        tensorboard.add_video('input_combine', input_combine.data.permute(0, 2, 1, 3, 4), self.current_epoch, 2)
        modelG_out = self.modelG(input_combine, input_semantic, input_flow, input_mask)

        warped_object, warped_mask, affine_matrix, pred_complete = modelG_out
        losses = self.modelD(0,
                             [warped_object, warped_mask, affine_matrix, pred_complete, label_combine, label_mask])

        real_sequence, fake_sequence = self.modelD.gen_seq(input_mask, warped_mask, label_mask, tIn, tOut)
        losses_T = self.modelD(1, [real_sequence, fake_sequence])

        losses = [torch.mean(x) if x is not None else 0 for x in losses]
        losses_T = [torch.mean(x) if x is not None else 0 for x in losses_T]
        # TODO: find how they used it
        # loss_dict = dict(zip(self.modelD.model.loss_names, losses))
        loss_dict = dict(zip(['Image', 'Scale', 'Rotation', 'Shear',
                              'Translation', 'smooth'], losses))
        # loss_dict_T = dict(zip(self.modelD.model.loss_names_T, losses_T))
        loss_dict_T = dict(zip(['G_T_GAN', 'D_T_real', 'D_T_fake'], losses_T))

        # collect losses
        self.loss_G, self.loss_D_T = self.modelD.get_losses(loss_dict, loss_dict_T)

        self.manual_backward(self.loss_G, optimizer_G)
        optimizer_G.step()

        self.manual_backward(self.loss_D_T, optimizer_D_T)
        optimizer_D_T.step()
        print(self.loss_G.item(), ' loss_G')
        print(self.loss_D_T.item(), ' loss_D_T')
        self.log('loss_G', self.loss_G, on_epoch=True)
        self.log('loss_D_T', self.loss_D_T, on_epoch=True)
        vid_log = torch.stack(warped_object).permute(1, 0, 2, 3, 4).data
        tensorboard.add_video('warped_object', vid_log, self.current_epoch, 2)

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
        scheduler_G = LambdaLR(optimizer_G,
                               lr_lambda=lambda epoch: (1 - (epoch - self.opt.niter) / self.opt.niter_decay))
        scheduler_D_T = LambdaLR(optimizer_D_T,
                                 lr_lambda=lambda epoch: (1 - (epoch - self.opt.niter) / self.opt.niter_decay))
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
    video_dataset = data.VideoDataset(dataset, 16, t_in=opt.tIn, t_out=opt.tOut, every_nth=1)
    video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    dataloader = video_loader  # TODO: add a dataloader
    # pytorch lightning trainer for training the model
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    trainer = pl.Trainer(gpus=1, logger=tb_logger, automatic_optimization=False)
    trainer.fit(model, dataloader)
