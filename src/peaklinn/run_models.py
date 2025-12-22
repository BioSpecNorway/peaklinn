import numpy as np
import torch
from scipy.io import loadmat
from scipy.interpolate import interp1d

from peaklinn.configs.model_config import load_peaklinn_config
from peaklinn.models.model import MieDescatteringAutoencoder
from peaklinn.models.model_wrapper import ModelWrapper
from peaklinn.paths import WEIGHTS_DIR, DATA_DIR


def load_model(weights_name):
    config = load_peaklinn_config()
    model = MieDescatteringAutoencoder(config)
    state_dict = torch.load(WEIGHTS_DIR / weights_name, weights_only=True)
    model.load_state_dict(state_dict)
    wrapper = ModelWrapper(model)
    return wrapper


def load_peaklinn():
    return load_model('weights_export_peaklinn_de7b8c_epoch=9789.pt')


def load_chemlinn():
    return load_model('weights_export_chemlinn_780e3d_epoch=624.pt')


def main():
    peaklinn = load_peaklinn()
    chemlinn = load_chemlinn()

    data = loadmat(DATA_DIR / 'Fig4A_Hyperspec_7_4.mat')
    spectra = data['rawSpec'].squeeze()
    wns = data['wn'].squeeze()

    new_wns = np.linspace(1000, 3500, 1408)
    spectra = interp1d(wns, spectra)(new_wns)
    wns = new_wns

    peak_res = peaklinn.transform(wns, spectra, batch_size=1024)
    chem_res = chemlinn.transform(wns, spectra, batch_size=1024)

    def plot_results(i):
        import matplotlib.pyplot as plt
        plt.plot(wns, spectra[i], color='red', label='raw')
        plt.plot(wns, peak_res[i], color='blue', label='peaklinn')
        plt.plot(wns, chem_res[i], color='green', label='chemlinn')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()

    plot_results(7452)


if __name__ == '__main__':
    main()
