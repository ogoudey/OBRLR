## `main.py` usage
To train or test a SAC implementation in a simulated robotic arm (Kinova) environment, run `python3 main.py` + a parameters file.

To specify the parameters file, use `--params <path_of_yaml>`, typically `parameters/x` (without `.yaml`). In the parameters file, put `train2` for the newest training script, put `testing=True` to do visual tests at the end of training, and do `teleop` for starting a run with a teleoped "demonstration" (not currently recommended). Put `HER=True` in the parameters to use Hindsight Experience Replay.

To load a replay buffer use `--rb <path_of_replay_buffer>`, something like `--rb replays/teleop.tmp`, but this is not recommended when following SAC conventions.

As of 4/30/25, `python3 main.py --params parameters/home` is the most likely to converge among the usages run with `main.py`.

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
