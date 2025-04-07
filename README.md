
## Simulated Robot
### Dumb RL
Tabular Monte Carlo: the state/observation space (`obs_vector`) is a truncated x,y,z position (to the tenths digit). Actions are a choice of -0.1, 0, or 0.1 (velocities) for each of the 6 relevant joints (a gripper joint is omitted). Rewards are given by how close the gripper (end effector) is to the cube in a graded manner.

Un-tuned, 10 episodes, gamma=0.99:
```
python3 dumb_rl.py
```
![Preliminary results](dumb_rl.png)
The yellow line is what we care about (closest distance to the cube), but the visualization of the simulation suggests that distance becomes less important as we move closer, since the eef is fairly huge (can it even get closer than grade 7?).

## Real Robot (WIP)
1. Download the python wheel [here](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/3.3.0/kortex_api-3.3.0.2-py3-none-any.whl)
2. `pip install <path_to_that_download>`
(The above is covered by the Kortex [repo](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/tree/master/api_python/examples)
3. [Examples](`git clone https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L`) 
4. `sudo ifconfig eth0 192.168.1.2 netmask 255.255.255.0 up`
5. `ifconfig`
6. `ping 192.168.1.2`

7. ... up next: check the connection 'modes', because connection seems good.

### Requirements (for vision)
`pip install torch, torchvision, pycocotools`
