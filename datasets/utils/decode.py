import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
try: 
    from turbojpeg import TurboJPEG
    _jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so.0")
    _use_turbo = True
except Exception as e:
    print(f"TurboJPEG not found or failed to load, falling back to OpenCV/PIL: {e}")
    _use_turbo = False
    import cv2
_image_ext = ('.jpg', '.jpeg', '.png')

_IMG_THREADS = int(os.environ.get('DECODE_THREADS', '8'))

_DECODE_DIVISOR = int(os.environ.get('DECODE_DIVISOR', '4'))
# TurboJPEG scaling_factor tuple; must be one of (1,1),(1,2),(1,4),(1,8).
_TJ_SCALE = (1, _DECODE_DIVISOR) if _DECODE_DIVISOR in (1, 2, 4, 8) else (1, 4)



def _decode_metadata(path):
    metadata = np.load(path, allow_pickle = True)
    
    def find_image_path(data):
        if isinstance(data, np.ndarray):
            # -- Data is loaded via np.load => Handle np.ndarray cases
            if data.ndim == 0:
                return find_image_path(data.item())
            if data.dtype == object:
                for item in data:
                    res = find_image_path(item)
                    if res is not None: return res
            return None

        if isinstance(data, str):
            if data.lower().endswith(_image_ext):
                return data
            return None

        if isinstance(data, dict):
            for value in data.values():
                result = find_image_path(value)
                if result is not None:
                    return result
            return None

        if isinstance(data, (list, tuple)):
            for item in data:
                result = find_image_path(item)
                if result is not None:
                    return result
            return None

        return None
            
    return find_image_path(metadata)
    
def _decode_image(path):
    if path.lower().endswith(_image_ext[:-1]):
        if _use_turbo:
            # TurboJPEG with DCT-domain downscaling — avoids decoding the full
            # 1920×1080 frame when only a 224×224 crop is needed downstream.
            with open(path, "rb") as f:
                img_bytes = f.read()
            return _jpeg_loader.decode(img_bytes, pixel_format=0,
                                       scaling_factor=_TJ_SCALE)
        else:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if _DECODE_DIVISOR > 1:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (w // _DECODE_DIVISOR, h // _DECODE_DIVISOR),
                                     interpolation=cv2.INTER_AREA)
                return img
            else:
                with Image.open(path) as pil:
                    if _DECODE_DIVISOR > 1:
                        w, h = pil.size
                        pil = pil.resize((w // _DECODE_DIVISOR, h // _DECODE_DIVISOR),
                                         Image.BILINEAR)
                    return np.array(pil.convert('RGB'))
    else: 
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))
    
def _decode(path):
    try:
        if path.lower().endswith(('.npz', '.npy')):
            path = os.path.join(
                os.path.dirname(os.path.dirname(path)),
                _decode_metadata(path)
            )

        if path.lower().endswith(_image_ext):
            return _decode_image(path)
    
    except Exception as e:
        print(f"Error processing {path=}: {e}")
        return None
 
 
def decode_batch(paths):
    """
    Decode a list of image paths in parallel using a thread pool.
    Threads overlap I/O wait (file reads) so per-sample latency drops
    roughly proportionally to the number of threads on I/O-bound storage
    (e.g. Kaggle NFS mounts).  CPU-bound JPEG decode also benefits because
    TurboJPEG / OpenCV release the GIL during decompression.
    """
    if not paths:
        return None

    n = len(paths)
    results = [None] * n

    if n == 1 or _IMG_THREADS <= 1:
        for i, p in enumerate(paths):
            results[i] = _decode_image(p)
    else:
        with ThreadPoolExecutor(max_workers=min(_IMG_THREADS, n)) as ex:
            futures = {ex.submit(_decode_image, p): i for i, p in enumerate(paths)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None

    frames = [f for f in results if f is not None]
    if not frames:
        return None
    return np.stack(frames, axis=0)


def start_decode_batch(paths):
    """Submit image decodes to a thread pool immediately; return (executor, futures).

    Call :func:`collect_decode_batch` later to gather results.  The gap between
    the two calls is "free" overlap time — use it for SQL meta queries.
    Returns ``(None, {})`` when *paths* is empty.
    """
    if not paths or _IMG_THREADS <= 1:
        return None, {}
    n = len(paths)
    ex = ThreadPoolExecutor(max_workers=min(_IMG_THREADS, n))
    futures = {ex.submit(_decode_image, p): i for i, p in enumerate(paths)}
    return ex, futures


def collect_decode_batch(executor, futures, n_paths):
    """Collect results from :func:`start_decode_batch`.  Returns ndarray or None.

    Falls back to synchronous sequential decode when *executor* is ``None``
    (i.e. when *paths* was empty or threading is disabled).
    """
    if executor is None:
        return None
    results = [None] * n_paths
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            results[idx] = fut.result()
        except Exception:
            results[idx] = None
    executor.shutdown(wait=False)
    frames = [f for f in results if f is not None]
    if not frames:
        return None
    return np.stack(frames, axis=0)