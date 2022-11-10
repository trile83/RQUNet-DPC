import os
import glob
import json
from collections import OrderedDict
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm


def load_annotations(path: str) -> Dict:
    with open(path) as f:
        annotations = OrderedDict(json.load(f))
    return annotations


def load_vrts(path: str, annotations: List[str]) -> List[str]:
    vrts = []
    for site in annotations:
        vrts.append(glob.glob(os.path.join(path, site, "*.vrt")))
    return vrts


@torch.no_grad()
def embed(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    z_ts, c_ts, labels, files, coords, locations = [], [], [], [], [], []
    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch in pbar:

        x, y = batch["x"].to(device), batch["y"].to(device)

        with autocast():
            z_t, c_t = model.embed(x)

        z_ts.append(z_t)
        c_ts.append(c_t)
        labels.append(y)
        files.extend(batch["files"])
        locations.extend(batch["location"])
        coords.extend(batch["coords"])

    z_ts = torch.cat(z_ts, dim=0).cpu().numpy()
    c_ts = torch.cat(c_ts, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    output = dict(zt=z_ts, ct=c_ts, labels=labels)
    metadata = dict(files=files, locations=locations, coords=coords)

    return output, metadata
