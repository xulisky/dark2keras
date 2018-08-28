# dark2keras

Implementation by Xu Li

<font size=9><b>YOLO v1:</b></font> [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)

<font size=9><b>YOLO v2:</b></font> [YOLO9000: Better, Faster, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)

<font size=9><b>YOLO v3:</b></font> [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

See the [project webpage](https://pjreddie.com/darknet/yolo/) for more details.

## Why convert YOLO to keras #

Darknet is an deeplearning tools written in C and CUDA. It is fast and easy, but it is so hard to custom a layer or loss function in it.

## Requirements
- keras
- tensorflow (backend)
- cv2 (read img, draw bbox)
- matpltlib (show img, you can switch to cv2)

## Usage
`detection_example.py` is a simple example. just swith the cfg, weights file and set suitable head, you can get different model.

## Note
This tool only support YOLO v2, YOLO v3.


## Reference
[keras-yolo](https://github.com/BrainsGarden/keras-yolo)
darkflow
