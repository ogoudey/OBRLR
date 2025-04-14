## Vision 2 -- Using YOLO

1. Annotate data
(2. `pip install labelme2yolo`)
3. Get YOLOformat annotations with `labelme2yolo --json_dir <path-to-annotations>`
4. clone YOLOv5 modules `git clone https://github.com/ultralytics/yolov5`
5. `cd yolov5 && pip install -r requirements.txt`
6. Now transfer the YOLO model to the custom model (200 epochs seems to work):
```
python3 train --img 400 --batch 16 --epochs 200 --data <.yaml generated from 3.> --weights yolov5s.pt --cache
```
Edit the .yaml if it points to the wrong absolute directory paths.

That's it!
``` (python)
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) # path is to the model originally in runs/expX/weights/
img_path = 'YOLODataset/images/val/sideview3.png'
results = model(img_path)
results.show()
```
