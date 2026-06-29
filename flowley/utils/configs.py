import sys
import yaml
import argparse
from copy import deepcopy
from typing import List
from omegaconf import DictConfig, OmegaConf


def read_config(filepath: str) -> DictConfig:
    with open(filepath, 'r') as f:
        config = DictConfig(yaml.safe_load(f))
    return config


def update_config(config: DictConfig, args: argparse.Namespace) -> DictConfig:
    for key, value in vars(args).items():
        if value is not None:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    return config


def parse_args(config: argparse.Namespace, description: str = None):
    parser = argparse.ArgumentParser(description=description)
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, DictConfig):
            for sub_key, sub_value in value.items():
                parser.add_argument(f'--{key}.{sub_key}', type=type(sub_value))
        else:
            parser.add_argument(f'--{key}', type=type(value))
    return parser


def parse_config_with_modified_args():
    if len(sys.argv) < 2:
        print("Usage: python ... <config_file> [--param1 value1] [--param2 value2] ...")
        sys.exit(1)

    config_file = sys.argv[1]
    config = read_config(config_file)

    # parse the remaining arguments and update the config
    if len(sys.argv) > 2:
        parser = parse_args(config)
        args = parser.parse_args(sys.argv[2:])
        config = update_config(config, args)

    return config


def write_config(config: DictConfig, dirpath: str):
    cfg = OmegaConf.to_container(config, resolve=True)
    with open(f"{dirpath}/config.yaml", "w") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def override_nested(
    full_cfg: DictConfig,
    subset_cfg: DictConfig,
    exclude_keys: List[str],
) -> DictConfig:
    """Return a new DictConfig based on full_cfg,
    but with nested values from subset_cfg overriding only when they differ.
    Top-level keys in subset_cfg are ignored.
    """
    merged = deepcopy(full_cfg)

    def _recurse(src: DictConfig, dst: DictConfig, nested: bool = False):
        for key, val in src.items():
            if isinstance(val, DictConfig) and key not in exclude_keys \
                    and key in dst and isinstance(dst[key], DictConfig):
                # descend into sub-DictConfigs → now “nested=True”
                _recurse(val, dst[key], nested=True)
            else:
                # only override if we’re already nested and the key exists
                if nested and key not in exclude_keys and \
                        key in dst and val is not None and dst[key] != val:
                    dst[key] = val

    _recurse(subset_cfg, merged)
    return merged
