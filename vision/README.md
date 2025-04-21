# Training Vision
The vision module should be trained before the RL algorithm gathers data.

## Gather data
18 images gathered in `data/Robosuite2/Images`

## Make annotations
`labelme` makes annotations for the images. (`pip install labelme')

YOLO takes a certain style of annotations:
`labelme2yolo --json_dir <path-to-annotations>` (after `pip install labelme2yolo`)

## Retrain model
Clone YOLOv5 modules with `git clone https://github.com/ultralytics/yolov5`
Go into the cloned directory and `pip install -r requirements.txt`
To start (re)training:
```
python3 train.py --img 400 --batch 16 --epochs 200 --data <.yaml generated inside YOLODataset> --weights yolov5s.pt --cache
```

## Extract trained model
From `yolov5/` do `mv runs/train/exp/weights/best.pt ../../<model_name.pt>`

You can now delete the cloned folder `yolov5/`.

## Test
To run a sanity check:
```
>>> import torch
>>> model = torch.hub.load('ultralytics/yolov5', 'custom', path='<model_name.pt>', force_reload=True) # path is to the model originally in runs/expX/weights/
>>> img_path = 'data/Robosuite2/YOLODataset/images/val/sideview6.png'
>>> results = model(img_path)
>>> results.show()
```


