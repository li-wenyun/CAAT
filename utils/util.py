import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import os



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)


google_drive_paths = {
    "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",

    "mapper/pretrained/afro.pt": "https://drive.google.com/uc?id=1i5vAqo4z0I-Yon3FNft_YZOq7ClWayQJ",
    "mapper/pretrained/angry.pt": "https://drive.google.com/uc?id=1g82HEH0jFDrcbCtn3M22gesWKfzWV_ma",
    "mapper/pretrained/beyonce.pt": "https://drive.google.com/uc?id=1KJTc-h02LXs4zqCyo7pzCp0iWeO6T9fz",
    "mapper/pretrained/bobcut.pt": "https://drive.google.com/uc?id=1IvyqjZzKS-vNdq_OhwapAcwrxgLAY8UF",
    "mapper/pretrained/bowlcut.pt": "https://drive.google.com/uc?id=1xwdxI2YCewSt05dEHgkpmmzoauPjEnnZ",
    "mapper/pretrained/curly_hair.pt": "https://drive.google.com/uc?id=1xZ7fFB12Ci6rUbUfaHPpo44xUFzpWQ6M",
    "mapper/pretrained/depp.pt": "https://drive.google.com/uc?id=1FPiJkvFPG_y-bFanxLLP91wUKuy-l3IV",
    "mapper/pretrained/hilary_clinton.pt": "https://drive.google.com/uc?id=1X7U2zj2lt0KFifIsTfOOzVZXqYyCWVll",
    "mapper/pretrained/mohawk.pt": "https://drive.google.com/uc?id=1oMMPc8iQZ7dhyWavZ7VNWLwzf9aX4C09",
    "mapper/pretrained/purple_hair.pt": "https://drive.google.com/uc?id=14H0CGXWxePrrKIYmZnDD2Ccs65EEww75",
    "mapper/pretrained/surprised.pt": "https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI",
    "mapper/pretrained/taylor_swift.pt": "https://drive.google.com/uc?id=10jHuHsKKJxuf3N0vgQbX_SMEQgFHDrZa",
    "mapper/pretrained/trump.pt": "https://drive.google.com/uc?id=14v8D0uzy4tOyfBU3ca9T0AzTt3v-dNyh",
    "mapper/pretrained/zuckerberg.pt": "https://drive.google.com/uc?id=1NjDcMUL8G-pO3i_9N6EPpQNXeMc3Ar1r",

    "example_celebs.pt": "https://drive.google.com/uc?id=1VL3lP4avRhz75LxSza6jgDe-pHd2veQG"
}

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )