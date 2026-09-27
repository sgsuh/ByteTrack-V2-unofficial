"""Thin inference wrapper around the vendored YOLOX detector."""
import numpy as np
import torch

from .yolox.data.data_augment import preproc
from .yolox.exp.yolox_base import Exp
from .yolox.utils import fuse_model, postprocess

RGB_MEANS = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class YOLOXDetector:
    """Single-image YOLOX inference returning boxes in original image pixels.

    Args:
        ckpt: checkpoint path containing a "model" state dict.
        depth, width: YOLOX scaling factors (YOLOX-X: 1.33, 1.25).
        num_classes: number of detector classes.
        test_size: network input size (h, w).
        conf_thresh: minimum obj_conf * cls_conf kept before NMS.
        nms_thresh: NMS IoU threshold.
        fp16: run the network in half precision.
        fuse: fuse conv + bn layers.
    """

    def __init__(self, ckpt, depth=1.33, width=1.25, num_classes=1, test_size=(800, 1440),
                 conf_thresh=0.01, nms_thresh=0.7, fp16=True, fuse=True, device="cuda"):
        exp = Exp()
        exp.depth, exp.width, exp.num_classes = depth, width, num_classes
        model = exp.get_model()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model = model.to(device).eval()
        if fuse:
            model = fuse_model(model)
        if fp16:
            model = model.half()

        self.model = model
        self.device = device
        self.fp16 = fp16
        self.num_classes = num_classes
        self.test_size = tuple(test_size)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    @torch.no_grad()
    def __call__(self, img_bgr):
        """Detect objects in a BGR image.

        Returns:
            (N, 5) float array of [x1, y1, x2, y2, score], score = obj_conf * cls_conf.
        """
        img, ratio = preproc(img_bgr, self.test_size, RGB_MEANS, RGB_STD)
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        x = x.half() if self.fp16 else x.float()
        outputs = postprocess(self.model(x), self.num_classes, self.conf_thresh, self.nms_thresh)[0]
        if outputs is None:
            return np.zeros((0, 5), dtype=np.float32)
        outputs = outputs.float().cpu().numpy()
        boxes = outputs[:, :4] / ratio
        scores = outputs[:, 4] * outputs[:, 5]
        return np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
