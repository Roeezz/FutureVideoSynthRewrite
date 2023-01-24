import os
import sys
import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.input = None

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    # used in test time, no backprop
    def test_step(self, *args, **kwargs):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def resolve_version(self):
        import torch._utils
        try:
            torch._utils._rebuild_tensor_v2
        except AttributeError:
            def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
                tensor.requires_grad = requires_grad
                tensor._backward_hooks = backward_hooks
                return tensor

            torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    def grid_sample(self, input1, input2):
        return torch.nn.functional.grid_sample(input1, input2, mode='bilinear', padding_mode='border')

    def resample(self, image, flow):
        b, c, h, w = image.size()
        if not hasattr(self, 'grid') or self.grid.size() != flow.size():
            self.grid = self.get_grid(b, h, w, dtype=flow.dtype)
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
        final_grid = (self.grid + flow).permute(0, 2, 3, 1)
        output = self.grid_sample(image, final_grid)
        return output

    def get_grid(self, batchsize, rows, cols, dtype=torch.float32):
        hor = torch.linspace(-1.0, 1.0, cols)
        hor.requires_grad = False
        hor = hor.view(1, 1, 1, cols)
        hor = hor.expand(batchsize, 1, rows, cols)
        ver = torch.linspace(-1.0, 1.0, rows)
        ver.requires_grad = False
        ver = ver.view(1, 1, rows, 1)
        ver = ver.expand(batchsize, 1, rows, cols)

        t_grid = torch.cat([hor, ver], 1)
        t_grid.requires_grad = False

        if dtype == torch.float16: t_grid = t_grid.half()
        return t_grid
