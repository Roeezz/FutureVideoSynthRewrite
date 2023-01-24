import functools
import torch.nn as nn
from fore.models import SpatialTransformer as STN
from . import Discriminator as DC

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, dataset, loasSize, tOut, ngf, gpu_ids=[]):
    # Default generator mode, composite
    norm_layer = get_norm_layer(norm_type='instance')
    netG = STN(input_nc, dataset, loasSize, tOut, ngf, norm_layer)

    # print_network(netG)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type='instance')
    netD = DC.MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)
    # print_network(netD)
    # if len(gpu_ids) > 0:
    #     netD.cuda(gpu_ids[0])
        # netD.cuda(gpu_ids[1])
    netD.apply(weights_init)
    return netD


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)