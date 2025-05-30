# Objective-based Reinforcement Learning for Robotics
The components of the objective to raise the cube 5cm:

![image](https://github.com/user-attachments/assets/5744903c-daf2-4ab3-b73c-4dd53b483d7d)

Sim2Real Transfer:

![image](https://github.com/user-attachments/assets/cf9f72f8-3367-439e-8f6f-5b29fa430cf0)

Here's a [video](https://www.youtube.com/shorts/jzrwVw6-p9k).

This project is an extension of [ogoudey/final_project](https://github.com/ogoudey/final_project), a group project done in Spring 2025.

## Usage
**Training parameters** are kept in `/parameters`. They specify learning components of an objective, and how each component is setup and trained. They are also used for testing, with the `--test` or `--cyclic_test` flags. 

Every run must use the `--params <name of parameters>` argument. To train in simulation, run:
```
python3 main.py --params <name_of_parameters_yaml>
```
This will output figures, trained networks, logs, and a replay buffer, to locations specified in the parameters file.

After training (unless the `--skip_test` option is given), you will be asked if you want to test the resulting policies. Hit ^C to kill the current segment of the test. After each test, you will be asked if you want to commit the policy. After all policies are tested, you will be asked if you want to push the committed policies. Thus, there are several convergent policies already in `policies/pushed`.

Run a pushed policy with:
```
python3 main.py --params standard_dense --test
```

To train a batch of training, run `$ ./batch_train <params1> <params2> <params3> <params4>...`, but this is not too useful because all convergent results are learned in under 5 minutes with a normal computer.



### Sim2Real Transfer
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
Run `python3 real/teleop`.
