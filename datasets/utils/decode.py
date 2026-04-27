import os
import numpy as np
from PIL import Image
try: 
    from turbojpeg import TurboJPEG
    _jpeg_loader = TurboJPEG("/usr/lib/libturbojpeg.so.0")
    _use_turbo = True
except Exception as e:
    print(f"TurboJPEG not found or failed to load, falling back to OpenCV/PIL: {e}")
    _use_turbo = False
    import cv2
_image_ext = ('.jpg', '.jpeg', '.png')



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
            # Method 1: TurboJPEG (Fastest)
            with open(path, "rb") as f:
                img_bytes = f.read()
            # pixel_format=0 typically refers to TJPF_RGB in most turbojpeg wrappers
            return _jpeg_loader.decode(img_bytes, pixel_format=0)
        else:
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                with Image.open(path) as img:
                    return np.array(img.convert('RGB'))
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
    Processes a list of image paths.
    Note: We decode sequentially here because the DataLoader 
    already parallelizes across multiple CPU cores.
    """
    frames = []
    for p in paths:
        f = _decode_image(p)
        if f is not None:
            frames.append(f)
    
    if not frames:
        return None
    
    return np.stack(frames, axis=0)