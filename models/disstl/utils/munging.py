import os
import pathlib


def flatten_dict(arb_dict, parent_key="", sep=".", sort_keys=True, ignore_keys=()):
    # for discussion on making this more generic (i.e. for any collections.MutableMapping)
    # see: https://stackoverflow.com/a/6027615
    # TODO: test out _flatten_dict() at
    #  https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
    items = []
    if sort_keys is True:
        arb_dict = dict(sorted(arb_dict.items()))
    for key, val in arb_dict.items():
        if key not in ignore_keys:
            if type(key) is not str:
                key = str(key)
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(val, dict):
                items.extend(flatten_dict(val, new_key, sep=sep, ignore_keys=ignore_keys).items())
            else:
                items.append((new_key, val))
    return dict(items)


def unflatten_dict(flat_dict, sep="."):
    unflattened_dict = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = unflattened_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return unflattened_dict


def get_disstl_root():
    DISSTL_ROOT = os.getenv("DISSTL_ROOT")
    if DISSTL_ROOT is None:
        DISSTL_ROOT = pathlib.Path(__file__).parent.parent.parent

    return DISSTL_ROOT


def hydra_get_class(target):
    return target
