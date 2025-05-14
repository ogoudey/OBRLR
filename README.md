# Component-based Soft Actor Critic (or, Objective-based Reinforcement Learning for Robotics OBRLR)

## Usage
To train in simulation, run:
```
python3 main.py --params parameters/<name of yaml>
```
This will output figures, trained networks, logs, and a replay buffer, in places specified in the parameters file.

After training (unless the --skip_test option is given), you will be asked if you want to visualize the resulting policy. There, hit ^C to kill the current segment of the test (or pass `cut_component=True` to `sac.train()` and (if implemented) the episode will cut after a certain reward is acquired). Run `$ ./batch_train <params1> <params2> <params3> <params4>...` to perform a batch train.

To transfer onto the robot, run:
```
python3 main_transfer.py
```
which is currently bound to one of the successful objectives.

## Installation
1. Clone this repo.
2. Create a python virtual environment with `python3 -m venv <name of venv>`
3. Source the environment with `source <name of venv>/bin/activate`
4. Install dependencies with `pip install torch torchviz h5py pandas tqdm matplotlib pyyaml robosuite

### Kortex API
If you want to test this on the real robot, you have to first download the API that interfaces with the Kinova.
1. Download the `kortex_api` python wheel from [https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.2.0/kortex_api-2.2.0.post31-py3-none-any.whl](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.2.0/kortex_api-2.2.0.post31-py3-none-any.whl)
2. Run `pip install <path_to_that_download>`



## Taking photos
### In simulation
Run `python3 interface.py` and teleop around (`q`, `w`, `e`, `r`). To take photos hit `p`, then type the number of photos you want to take while the arm takes a random walk.

### On the Kinova
Run `python3 real/teleop`.
