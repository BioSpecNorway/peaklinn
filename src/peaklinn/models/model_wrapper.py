import numpy as np
import torch

from peaklinn.models.model import MieDescatteringAutoencoder


class ModelWrapper:

    def __init__(self, model: MieDescatteringAutoencoder, device=None):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device)
        self.model = model.eval()

    def transform(self, wns, samples, batch_size=None):
        start_shape = np.shape(samples)
        n_wns = np.shape(wns)[-1]
        samples: torch.Tensor = torch.atleast_2d(torch.tensor(samples)).reshape(-1, 1, n_wns)
        wns: torch.Tensor = torch.atleast_2d(torch.tensor(wns)).reshape(-1, 1, n_wns)
        wns = torch.broadcast_to(wns, samples.shape)

        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device

        wns = wns.type(dtype).to(device)
        samples = samples.type(dtype).to(device)
        with torch.no_grad():
            if batch_size is None:
                result = self.model.smart_forward(samples, wns)
            else:
                results = []
                bs = batch_size
                for i in range(0, len(samples), batch_size):
                    res = self.model.smart_forward(
                        samples[i: i + bs], wns[i: i + bs])
                    results.append(res)
                result = torch.concat(results)

        return result.cpu().numpy().reshape(start_shape)

