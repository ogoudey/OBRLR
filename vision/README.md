# File Structure

## ./
scripts for CNN training. Actually its a transfer/finetuning from a pretrained CNN.

## data/
dataset PennFutanPed that was used in the tutorial for finetuning
dataset Robosuite1 that is created with `sideview_image_generator.py`

### Robosuite1
Annotated with `labelme` (`pip install labelme` -- had to reinstall opencv)
Moved `annotation2mask.py` into the annotated directory and ran it, isolating the masks.
`custom_main.py` now runs the whole thing.
