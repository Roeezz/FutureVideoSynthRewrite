import argparse
import os
import torch

from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):                
        self.parser.add_argument('--ImagesRoot', type=str, default='/disk1/shared/dataset/cityscapes/cityscapes/leftImg8bit_sequence_256p/')
        self.parser.add_argument('--BackRoot', type=str, default='')
        self.parser.add_argument('--InstanceGTRoot', type=str, default='')
        self.parser.add_argument('--Instance', type=str, default='')
        self.parser.add_argument('--SemanticGTRoot', type=str, default='')
        #self.parser.add_argument('--ImagesRoot', type=str, default='/data/shared/dataset/cityscapes/leftImg8bit_sequence_512p/')
        # 256p root /disk1/shared/dataset/cityscapes/cityscapes/leftImg8bit_sequence_256p/
        # 512p: /disk1/shared/dataset/cityscapes/cityscapes/leftImg8bit_sequence_512p
        # 1024p: /disk1/shared/dataset/cityscapes/cityscapes/leftImg8bit_sequence
        self.parser.add_argument('--SemanticRoot', type=str, default='/disk1/shared/dataset/cityscapes/cityscapes_2/leftImg8bit_sequence_semantic/')
        #self.parser.add_argument('--SemanticRoot', type=str, default='/data/yue/cityscapes/leftImg8bit_sequence_semantic/')
        # 512p:
        #self.parser.add_argument('--InstanceRoot', type=str, default='/disk1/shared/dataset/cityscapes/cityscapes/InstanceMap_512p/')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--dataset', type=str, default='cityscapes', help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--image_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--back_image_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--semantic_nc', type=int, default=19, help='# of input semantic channels')
        self.parser.add_argument('--back_semantic_nc', type=int, default=11, help='# of input semantic channels')
        self.parser.add_argument('--instance_nc', type=int, default=1, help='# of input instance channels')
        self.parser.add_argument('--flow_nc', type=int, default=2, help='# of output image channels')

        # network arch
        self.parser.add_argument('--netG', type=str, default='composite', help='selects model to use for netG')        
        self.parser.add_argument('--ngf', type=int, default=128, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--n_blocks', type=int, default=9, help='number of resnet blocks in generator')
        self.parser.add_argument('--n_downsample_G', type=int, default=3, help='number of downsampling layers in netG')        

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--n_gpus_gen', type=int, default=-1, help='how many gpus are used for generator (the rest are used for discriminator). -1 means use all gpus')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='temporal', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='vid2vid', help='chooses which model to use. vid2vid, test')        
        self.parser.add_argument('--nThreads', default=10, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')        
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
                        
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='scaleWidth', help='scaling and cropping of images at load time [resize_and_crop|crop|scaledCrop|scaleWidth|scaleWidth_and_crop|scaleWidth_and_scaledCrop|scaleHeight|scaleHeight_and_crop] etc')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')                    
    
        # more features as input        
        self.parser.add_argument('--use_instance', action='store_true', help='if specified, add instance map as feature for class A')        
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, encode label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='number of encoded features')        
        self.parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--netE', type=str, default='simple', help='which model to use for encoder') 
        self.parser.add_argument('--n_downsample_E', type=int, default=3, help='number of downsampling layers in netE')

        # for cascaded resnet        
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of resnet blocks in outmost multiscale resnet')        
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of cascaded layers')        

        # temporal
        self.parser.add_argument('--tIn', type=int, default=4, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        self.parser.add_argument('--tOut', type=int, default=5, help='number of output frames for prediction')
        self.parser.add_argument('--n_scales_spatial', type=int, default=1, help='number of spatial scales in the coarse-to-fine generator')        
        self.parser.add_argument('--fg_labels', type=str, default='11,12,13', help='label indices for foreground objects')

        
        # miscellaneous                
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')                
        self.parser.add_argument('--debug', action='store_true', help='if specified, use small dataset for debug')
        self.parser.add_argument('--only_dynamic', action='store_true', help='whether only train with moving object')
        self.parser.add_argument('--only_large', action='store_true', help='whether only train with object area larger than 0.01 of the image')
        self.initialized = True

    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        
        self.opt.fg_labels = self.parse_str(self.opt.fg_labels)
        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)
        if self.opt.n_gpus_gen == -1:
            self.opt.n_gpus_gen = len(self.opt.gpu_ids)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
