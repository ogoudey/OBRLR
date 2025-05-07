# Usage

## Soft Actor Critic on Lift
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


## Sequential training
```
$ ./batch_train <params1> <params2> <params3> <params4>...
```
will train models in a sequence. Useful if training over a long period of time and training times are short-ish.

## Model + ReplayBuffer saving
All replay buffers will be saved after data collection to replays/<rb_save_name>, which is specified in parameter files; and all networks (policy, Q1, Q2) are saved in a folder configurations/<configuration_save_name>, which is also in the parameters.

## Install dependencies using `pip`
1. Create python virtual environment
2. 2. Download other requirements by e.g. `pip install -r requirements.txt` or use `pip install -r requirements_py_3_9_2.txt` if you are using python 3.9.2
  
### Kortex API
1.Download the `kortex_api` python wheel [here](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.2.0/kortex_api-2.2.0.post31-py3-none-any.whl)
2. Run `pip install <path_to_that_download>`

## Taking photos
### In simulation
Run `python3 interface.py` and teleop around (`q`, `w`, `e`, `r`). To take photos hit `p`, then type the number of photos you want to take while the arm takes a random walk.

### On the Kinova
Updated photography script in progress...
