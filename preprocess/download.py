import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch as th
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "base": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/base.pt",
    "upsample": "https://openaipublic.blob.core.windows.net/diffusion/dec-2021/upsample.pt",
    }




def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    """

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(local_path):
        return local_path
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


if __name__ == '__main__':
    fetch_file_cached(
        MODEL_PATHS['base'], progress=True, cache_dir='./ckpt', chunk_size=4096
    )

    fetch_file_cached(
        MODEL_PATHS['upsample'], progress=True, cache_dir='./ckpt', chunk_size=4096
    )    
