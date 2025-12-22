from dataclasses import dataclass
import yaml

from peaklinn.paths import CONFIGS_DIR


@dataclass(frozen=True)
class MieDSAEConfig:
    float_precision: bool
    n_blocks: int
    start_n_filters: int
    kernel_size: int
    downsampling: str
    n_end_convs: int
    batch_norm: bool
    cbam: bool
    cbam_kernel_size: int
    with_wavelengths: bool
    skip_connections: bool
    zero_mean_qext: bool
    one_scale_qext: bool


def load_peaklinn_config() -> MieDSAEConfig:
    cfg = yaml.safe_load(open(CONFIGS_DIR / 'paper_peaklinn_chemlinn_model_config.yml'))
    return MieDSAEConfig(**cfg)
