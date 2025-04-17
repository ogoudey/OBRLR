# Guide


## Configurations
Run `python3 main.py` for default configuration with parameters set to those in `params.yaml`. The training loop will continue for all `num_iterations` then ask if you want to do more iterations (give a number of iterations) or `n` (the 

To use a specific parameters yaml, use `--params <path_of_yaml>`. Right now there are three parameters files with minor differences. `parameters/data_collection` emphasizes policy-based data-collection, the kind you let run for while. `parameters/train_nn` emphasizes updating the neural networks.

To load a replay buffer use `--rb <path_of_replay_buffer>`, something like `--rb replays/1raise_1e-1else.tmp`.

To skip training, load a model with `--model <path_to_model>`.


## Model + ReplayBuffer saving
All replay buffers will be saved after data collection (teleop or policy-based) to replays/<rb_save_name>, which is specified in parameter files.

Models (policy networks) are saved after training run (after `n` stops further iterating), they are saved to 'saved_model_name` in the parameters yaml.

## Collecting data from teleop
In the data collection phase of training, there are ``data collection'' functions to call. `collect_data_from_policy` or `collect_teleop_data`.

Using teleop, hit <key>, <enter> like regular `input()`. Controls are:
    q: 
    w:      up, down, right, or left of eef
    e:
    r:      idk
    t:      idk
    y:      idk
    u:      gripper
    -1 through 1: sets the speed which all the other controlls use

E.g. A sequence of inputs (0.2, u, u, u) closes the gripper with speed 0.2 three times.

### Seeing detections
To inspect detections (a part of the state representation), change
```
self.state = self.sim_vision.detect(obs["sideview_image"], obs["sideview_depth"], self.env.sim, no_cap=True)
```
to
```
self.state = self.sim_vision.detect(obs["sideview_image"], obs["sideview_depth"], self.env.sim, no_cap=False)
```
This will help with teleop, and with assessing the accuracy of YOLO.

### Inspecting ReplayBuffers
```
>>> import soft_actor_critic as sac
>>> rb = sac.load_replay_buffer('replays/1raise_1e-1else.tmp')
Replay buffer of 141  episodes loaded!
>>> rb
<soft_actor_critic.ReplayBuffer object at 0x7f4397cef770>
```
ReplayBuffers have Episode => Step => SARS
Since a typical episode length is 50, a ReplayBuffer of size 100 has more like 5000 Steps for batch sampling.

## Install dependencies using `pip`
1. Create python virtual environment
2. Download the `kortex_api` python wheel [here](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.2.0/kortex_api-2.2.0.post31-py3-none-any.whl)
3. Run `pip install <path_to_that_download>`
4. Download other requirements by e.g. `pip install requirements.txt` or individually 
