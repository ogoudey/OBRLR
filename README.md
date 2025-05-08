# Component-based Soft Actor Critic

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



# DDPG 
Running this section will require Stable Baselines 3. To install run ```pip install stable_baselines3```.

To train DDPG in simulation, run
```
cd ddpg
python ddpg_main.py --config config.yaml
```

Metrics during training will output to the terminal and a CSV file in /logdir
The training results will be automatically plotted to the /ddpg folder
The plot for training loss can be created by running
```
python plot_training_loss.py tensorboard/DDPG_1/<yourTensorboardLog> --tag train/actor_loss
```


# PPO 

Running this section will require Stable Baselines 3. To install run ```pip install stable_baselines3```.

To train PPO in simulation, run:

```
python3 ppo/ppo_main.py --mode train
```

Metrics during training will output to the terminal, or onto Tensorboard by running

```
tensorboard --logdir ./tensorboard_logs --port 6006
```

which can then be viewed by entering the following URL into a browser: http://localhost:6006.

The trained policy can be tested on the lift task by running:

```
python3 ppo/ppo_main.py --mode test
```

Note: when testing, if the Robosuite GUI does not appear, ensure that ```self.has_renderer = True``` is set in ```ppo/sb_interface.py```.

