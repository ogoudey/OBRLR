# Guide

## Configurations
Run `python3 main.py` for default configuration with parameters set to those in `params.yaml`. The training loop will proceed then ask if you want to do more iterations (give a number of iterations) or `n`.

To use a specific parameters yaml, use `--params <path_of_yaml>`.

To load a replay buffer use `--rb <path_of_replay_buffer>`, something like `--rb replays/1raise_1e-1else.tmp`.

To load a Q-network, use `--q <path_of_qnetwork`.

To load a Policy Network, use `--pi <path_of_policy>`

## Model + ReplayBuffer saving
All replay buffers will be saved after data collection (teleop or policy-based) to replays/<rb_save_name>, which is specified in parameter files.

Models (policy networks and qnetworks) are saved after training runs (after `n` stops further iterating). They are saved to their respective ``save name''s in the parameters yaml.

In summary, python script args are the load paths and parameter fields are the save paths.

## Collecting data from teleop
Data can be collected from teleoperation with the `human` parameter.

Using teleop, hit <key> then <enter>, like regular `input()`. Controls are:
    q: 
    w:      up, down, right, or left of eef
    e:
    r:      idk
    t:      idk
    y:      idk
    u:      gripper
    -1 through 1: sets the speed which all the other controls use

E.g. A sequence of inputs (0.2, u, u, u) closes the gripper with speed 0.2 three times.

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
4. Download other requirements by e.g. `pip install -r requirements.txt` or use `pip install -r requirements_py_3_9_2.txt` if you are using python 3.9.2
