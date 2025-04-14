# 
Install dependencies
1. Create python virtual environment
2. Download the python wheel [here](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.2.0/kortex_api-2.2.0.post31-py3-none-any.whl)
3. Run `pip install <path_to_that_download>`
4. Download other requirements such as `pip install torch torchvision pycocotools` (more?)

5. Run `python3 main.py` for training in robosuite. Parameters are currently set absurdly low.
Does not succeed on HRIlab desktop due to a different robosuite being installed.
