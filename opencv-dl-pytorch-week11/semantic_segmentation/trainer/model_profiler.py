"""Implementation of the PyTorch Models Profiler"""
import torch
import torch.nn as nn


class ModelProfiler(nn.Module):
    """ Profile PyTorch models.

    Compute FLOPs (FLoating OPerations) and number of trainable parameters of model.

    Arguments:
        model (nn.Module): model which will be profiled.

    Example:
        model = torchvision.models.resnet50()
        profiler = ModelProfiler(model)
        var = torch.zeros(1, 3, 224, 224)
        profiler(var)
        print("FLOPs: {0:.5}; #Params: {1:.5}".format(profiler.get_flops('G'), profiler.get_params('M')))

    Warning:
        Model profiler doesn't work with models, wrapped by torch.nn.DataParallel.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.flops = 0
        self.units = {'K': 10.**3, 'M': 10.**6, 'G': 10.**9}
        self.hooks = None
        self._remove_hooks()

    def get_flops(self, units='G'):
        """ Get number of floating operations per inference.

        Arguments:
            units (string): units of the flops value ('K': Kilo (10^3), 'M': Mega (10^6), 'G': Giga (10^9)).

        Returns:
            Floating operations per inference at the choised units.
        """
        assert units in self.units
        return self.flops / self.units[units]

    def get_params(self, units='G'):
        """ Get number of trainable parameters of the model.

        Arguments:
            units (string): units of the flops value ('K': Kilo (10^3), 'M': Mega (10^6), 'G': Giga (10^9)).

        Returns:
            Number of trainable parameters of the model at the choised units.
        """
        assert units in self.units
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if units is not None:
            params = params / self.units[units]
        return params

    def forward(self, *args, **kwargs):
        self.flops = 0
        self._init_hooks()
        output = self.model(*args, **kwargs)
        self._remove_hooks()
        return output

    def _remove_hooks(self):
        if self.hooks is not None:
            for hook in self.hooks:
                hook.remove()
        self.hooks = None

    def _init_hooks(self):
        self.hooks = []

        def hook_compute_flop(module, _, output):
            self.flops += module.weight.size()[1:].numel() * output.size()[1:].numel()

        def add_hooks(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(hook_compute_flop))

        self.model.apply(add_hooks)


def profile_model(model, input_size, cuda):
    """ Compute FLOPS and #Params of the CNN.

    Arguments:
        model (nn.Module): model which should be profiled.
        input_size (tuple): size of the input variable.
        cuda (bool): if True then variable will be upload to the GPU.

    Returns:
        dict:
            dict["flops"] (float): number of GFLOPs.
            dict["params"] (int): number of million parameters.
    """
    profiler = ModelProfiler(model)
    var = torch.zeros(input_size)
    if cuda:
        var = var.cuda()
    profiler(var)
    return {"flops": profiler.get_flops('G'), "params": profiler.get_params('M')}
