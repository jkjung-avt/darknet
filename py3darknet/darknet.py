"""darknet.py

A python wrapper of libdarknet.so.

Reference: https://github.com/pjreddie/darknet/blob/master/python/darknet.py
"""


from ctypes import *


class _BOX(Structure):
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('w', c_float),
        ('h', c_float)
    ]


class _DETECTION(Structure):
    _fields_ = [
        ('bbox', _BOX),
        ('classes', c_int),
        ('prob', POINTER(c_float)),
        ('mask', POINTER(c_float)),
        ('objectness', c_float),
        ('sort_class', c_int),
        ('uc', POINTER(c_float)),
        ('points', c_int),
        ('embeddings', POINTER(c_float)),
        ('embedding_size', c_int),
        ('sim', c_float),
        ('track_id', c_int)
    ]


class _IMAGE(Structure):
    _fields_ = [
        ('w', c_int),
        ('h', c_int),
        ('c', c_int),
        ('data', POINTER(c_float))
    ]


class _METADATA(Structure):
    _fields_ = [
        ('classes', c_int),
        ('names', POINTER(c_char_p))
    ]


lib = CDLL('./libdarknet.so', RTLD_GLOBAL)

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

_set_gpu = lib.cuda_set_device
_set_gpu.argtypes = [c_int]

_load_net = lib.load_network_custom
_load_net.argtypes = [c_char_p, c_char_p, c_int, c_int]
_load_net.restype = c_void_p

_free_net = lib.free_network
_free_net.argtypes = [c_void_p]

#_load_meta = lib.get_metadata
#_load_meta.argtypes = [c_char_p]
#_load_meta.restype = _METADATA

_load_image = lib.load_image_color
_load_image.argtypes = [c_char_p, c_int, c_int]
_load_image.restype = _IMAGE

_free_image = lib.free_image
_free_image.argtypes = [_IMAGE]

_predict_image = lib.network_predict_image
_predict_image.argtypes = [c_void_p, _IMAGE]
_predict_image.restype = POINTER(c_float)

_get_network_boxes = lib.get_network_boxes
_get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
_get_network_boxes.restype = POINTER(_DETECTION)

_free_detections = lib.free_detections
_free_detections.argtypes = [POINTER(_DETECTION), c_int]

_do_nms_obj = lib.do_nms_obj
_do_nms_obj.argtypes = [POINTER(_DETECTION), c_int, c_int, c_float]

#_free_ptrs = lib.free_ptrs
#_free_ptrs.argtypes = [POINTER(c_void_p), c_int]


def _detect(net, classes, image, thresh=.3, hier_thresh=.5, nms=.5):
    """Detect an image (path name).

    # Returns
        Multiple detection results in the format of:
            ((x, y, w, h of bbox), confidenc_score, class_id)
    """
    im = _load_image(image, 0, 0)  # image has not been resized here
    num = c_int(0)
    pnum = pointer(num)
    _predict_image(net, im)
    dets = _get_network_boxes(
        net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if (nms):
        _do_nms_obj(dets, num, classes, nms);

    res = []
    for j in range(num):
        for i in range(classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append(((b.x, b.y, b.w, b.h), dets[j].prob[i], i))
    res = sorted(res, key=lambda x: -x[1])
    _free_image(im)
    _free_detections(dets, num)
    return res

#
# The public class starts here.
#

def classes_in_cfg(cfg_path):
    """Extract 'classes' from the cfg file."""
    with open(cfg_path, 'r') as f:
        cfg_lines = f.readlines()
    classes_lines = [l for l in cfg_lines if l.startswith('classes')]
    if not classes_lines:
        raise ValueError('no classes in the cfg file')
    return int(classes_lines[-1].strip().split('=')[-1].strip())


class Detector(object):

    def __init__(self, cfg_path, weights_path, gpu_id=-1):
        """Load the cfg/weights files of the darknet detector model."""
        _set_gpu(gpu_id)
        clear = 1  # clear the network
        batch = 1  # batch size
        self.net = _load_net(
            cfg_path.encode('utf-8'),
            weights_path.encode('utf-8'),
            clear,
            batch)
        self.classes = classes_in_cfg(cfg_path)
        self.gpu_id = gpu_id

    def __del__(self):
        #_free_net(self.net)
        pass

    def detect(self, img_path, thresh=.3, nms=.5):
        """Detect an image using net (the detector)."""
        res = _detect(
            self.net,
            self.classes,
            img_path.encode('utf-8'),
            thresh=thresh,
            nms=nms)
        return res


if __name__ == '__main__':
    detector = Detector('cfg/yolov4.cfg', 'yolov4.weights', gpu_id=0)
    res = detector.detect('data/dog.jpg')
    print(res)
