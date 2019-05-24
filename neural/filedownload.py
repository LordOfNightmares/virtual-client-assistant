import math
import requests
from tqdm import tqdm


def url_download(url, filename):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    total_size_int = math.ceil(total_size // 1024)
    wrote = 0
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=total_size_int, unit="kb"):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
    f.close()


