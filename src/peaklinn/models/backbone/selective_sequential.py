import torch.nn as nn


class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self.to_select = to_select

    def forward(self, x, add_to_layer_out: dict = None):
        res = []
        for name, module in self._modules.items():
            x = module(x)
            if add_to_layer_out and name in add_to_layer_out:
                x += add_to_layer_out[name]
            if name in self.to_select:
                res.append(x)
        return res
