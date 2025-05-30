### Provides RL algorithms for learning components of the objective ###

# Provides helpers to carry out objective "check"



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.optim as optim

from tqdm import tqdm
import tempfile

import os
import pathlib
import random
import time
import numpy as np
import pickle
import copy
import logging
logging.disable(logging.WARNING)
torch.autograd.set_detect_anomaly(True)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity=500):
        self.episodes = []
        self.capacity = capacity
        self.protected_episodes = 0

    def append(self, episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.forget(len(self.episodes) - self.capacity)
    
    def left_append(self, episode):
        self.episodes.insert(0, episode)
        self.protected_episodes += 1
        
        
    def sample_batch(self, batch_size):
        # Flatten all episodes into a list of steps
        all_steps = []
        for episode in self.episodes:
            all_steps.extend(episode.steps)
        # Randomly sample transitions from all_steps
        batch_steps = random.sample(all_steps, batch_size)
        
        # Build the batch dictionary
        batch = {
            "states": torch.stack([torch.tensor(step.state, dtype=torch.float32) for step in batch_steps]),
            "actions": torch.stack([torch.tensor(step.action, dtype=torch.float32) for step in batch_steps]),
            "rewards": torch.stack([torch.tensor(step.reward, dtype=torch.float32) for step in batch_steps]).unsqueeze(1),
            "next_states": torch.stack([torch.tensor(step.next_state, dtype=torch.float32) for step in batch_steps]),
            "done": torch.stack([torch.tensor(step.done, dtype=torch.float32) for step in batch_steps]).float().unsqueeze(1),
        }
        assert not torch.isnan(batch["states"]).any()
        assert not torch.isnan(batch["actions"]).any()
        assert not torch.isnan(batch["rewards"]).any()
        assert not torch.isnan(batch["next_states"]).any()
        return batch

    def forget(self, forget_size):
        for i in range(0, forget_size):
            self.episodes.remove(self.episodes[self.protected_episodes]) # Avoids the "pilot" episode. Avoids the HER resampled pilot.
    
    def clean(self):
        to_del = []
        for episode in self.episodes:
            for step in episode.steps:
                print(step.state)
                if np.any((step.state < -2) | (step.state > 2)) or np.any((step.action < -2) | (step.action > 2)) or np.any((step.reward < -2) | (step.reward > 1)):
                    to_del.append(episode)
        for episode in to_del:
            self.episodes.remove(episode)
        print("Removed", len(to_del), "episodes from ReplayBuffer")
            
    def save(self, name):
        with open("replays/" + name + ".tmp", "wb") as f:
            pickle.dump(self, f)
        print("Replay buffer saved to replays/" + name + ".tmp")
                 
class Episode:
    def __init__(self):
        self.steps = []
    
    def append(self, state, action, reward, next_state, done):
        #print(state, action, reward, next_state)
        step = Step(state, action, reward, next_state, done)
        self.steps.append(step)
    
class Step:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state.detach().numpy()
        self.action = action.detach().numpy()
        self.reward = reward
        self.next_state = next_state.detach().numpy()
        self.done = done
        

class QNetwork(nn.Module):
    def __init__(self, params):
        
        state_action_dim = params["state_dim"] + params["action_dim"] # 13 state dimension, 4 action dimension
        q_value_dim = 1
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_action_dim, params['hidden_layers']['l1'])
        self.fc2 = nn.Linear(params['hidden_layers']['l1'], params['hidden_layers']['l2'])
        
        self.q_value_layer = nn.Linear(params['hidden_layers']['l2'], q_value_dim)
        
    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        
        x = F.relu(self.fc2(x))
        
        q_value = self.q_value_layer(x)
        
        return q_value
        
    def qvalue(self, state_action):
        q_value = self.forward(state_action)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, params):
        state_dim = params["state_dim"] # 3 eef location, 3 cube location, 3 delta locations, 3 cube goal location, gripper (is the max)
        action_dim = params["action_dim"] # 3 dimensions, gripper (is the max)
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, params['hidden_layers']['l1'])
        self.fc2 = nn.Linear(params['hidden_layers']['l1'], params['hidden_layers']['l2'])
        
        self.mean_layer = nn.Linear(params['hidden_layers']['l2'], action_dim)
        
        self.log_std_layer = nn.Linear(params['hidden_layers']['l2'], action_dim)
    
    def forward(self, state):

        x = F.relu(self.fc1(state))

        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-5, max=2)
        std = torch.exp(log_std)
       
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        #print("Means:", mean, "\nSTD:", std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        #print("Mean:", mean.detach().numpy(), "STD:", std.detach().numpy(), "Action:", action.detach().numpy())
        return action, log_prob
        
    def deterministic_action(self, state):
        mean, std = self.forward(state)

        action = torch.tanh(mean)
        
        return action

def form_state(sim, params, logger=None):
        # Order matters for internal sim state
        state = np.array([])
        if "eef_pos" in params:
            state = np.concatenate((state, sim.get_eef_pos()))
        # REAL GET EEF_POS
        
        if "cube_pos" in params:
            state = np.concatenate((state, sim.get_cube_pos()))
                
        if "eef_cube_displacement" in params:
            state = np.concatenate((state, sim.eef_cube_displacement()))
        
        if "cube_cube_displacement" in params:
            state = np.concatenate((state, sim.cube_cube_displacement()))
        
        if "current_grasp" in params:
            state = np.concatenate((state, sim.get_current_grasp()))
    
        if "cube_goal_pos" in params:    
            state = np.concatenate((state, sim.get_initial_cube_goal()))
        if logger:
            logger.info(f"{params}:\n{state}")
        # tensor for networks
        return torch.tensor(state, dtype=torch.float32)

def form_action(action, params, logger=None):
    action = action.detach().numpy()
    # Must fit into the robosuite 'port'
    simized_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if "eef_desired_move" in params:
        simized_action[0:3] = action[0:3]
    if "gripper_move" in params:
        simized_action[6] = action[3]
    if "all_7_joints" in params:
        simized_action = action

    if logger:
        logger.info(f"{params}:\n {simized_action}") 
    return simized_action

def form_reward(sim, params):
    reward = 0.0
    if "k_cube_eef_distance" in params.keys():
        reward += sim.k_cube_eef_distance(params["k_cube_eef_distance"])
    if "k_cube_cube_distance" in params.keys():
        reward += sim.k_cube_cube_distance(params["k_cube_cube_distance"])
    if "eef_cube_distance" in params.keys():
        reward += sim.eef_cube_distance(params["eef_cube_distance"])
    if "cube_cube_distance" in params.keys():
        reward += sim.cube_cube_distance(params["cube_cube_distance"])
    return reward



#       Control Scripts       #



trained_policy = None
trained_critic1 = None
trained_critic2 = None

# train method based on OpenAI's spinningup
def train(params, composition, parameter_file_name=""):
    import interface
    sim = interface.Sim(params['show']) # will tell the sim what initial internal state to hold on to
    sim.compose(composition)
    
    logger = setup_logger(None)
    
    ### plotting
    q_losses = []
    pi_losses = []
    max_ep_rewards = []
    q1s = []
    q2s = []
    stds = []
    action_magnitudes = []
    q1s_mean = []
    q1s_max = []
    q1s_min = []
    q1s_std = []
    episode_cum_rewards = []
    alphas = []
    mean_reward_per_grad = []
    ###
    
    network_parameters = params["networks"]

    print("Generating new configuration...")
    policy = PolicyNetwork(network_parameters)
    critic1 = QNetwork(network_parameters)
    critic2 = QNetwork(network_parameters)
    
    if "replay_buffer" in params.keys():
        print("Loading replay buffer", params["replay_buffer"])
        rb = load_replay_buffer(params["replay_buffer"])
        if "teleop" in params["replay_buffer"]: # A way to emphasize the demonstration (will be emhpasiized if 'teleop' is in the rb name)
            he = HER_resample(sim, rb.episodes[0], logger, mode="final")
            rb.left_append(he)
    else:
        print("Starting new replay buffer...")
        rb = ReplayBuffer()       
         
    ### If including a teleop episode (DEPRECATED) ###
    if "teleop" in params.keys():
        for tele_episode in range(0, params["teleop"]): # only works with teleop: 1
            e = collect_teleop_data(sim, rb, "teleop")
            if "bonus" in params.keys():
                give_bonus(e, params["bonus"])
            rb.left_append(e)
            he = HER_resample(sim, e, logger, mode="final")
            rb.left_append(he)
    ###
    
    if "all_7_joints" in params["pi"]["outputs"]:
        # Set mode of robosuite control to special case
        print("Not implemented yet!")
    else:
        pass
    
    if "HER" in params.keys():
        if params["HER"]:
            HER = True
    if "A-tuning" in params.keys():
        log_alpha = torch.tensor(0.0, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=params["A-tuning"]["lr"])
        alpha = log_alpha.exp()
        target_entropy = -params["networks"]["action_dim"]  # or a tuned value
        #target_entropy = -1
    else:
        alpha = params['algorithm']['alpha']
    alphas.append(alpha.item())
    policy_optimizer = optim.Adam(policy.parameters(), params['networks']['lr'])
    q1_optimizer = optim.Adam(critic1.parameters(), params['networks']['lr'])
    q2_optimizer = optim.Adam(critic2.parameters(), params['networks']['lr'])
    
    # Make 'target' networks 
    qnetwork1 = copy.deepcopy(critic1)
    qnetwork2 = copy.deepcopy(critic2)
    
    for p in qnetwork1.parameters():
        p.requires_grad = False
    for p in qnetwork2.parameters():
        p.requires_grad = False  
    
    
    total_steps, len_episode = params['algorithm']['num_iterations'], params['algorithm']['len_episode']
    gamma = params['algorithm']['gamma']
    
    gradient_after = params['algorithm']['gradient_after']
    gradient_every = params['algorithm']['gradient_every']
    save_every = params['algorithm']['save_every']
    full_sim_reset_every = 50000
    HER = False
    
    
    pi = params['pi']
    
    
    steps_taken = 0
    state = form_state(sim, pi["inputs"], logger)
    e = Episode()
    max_ep_reward = -99
    cum_reward = 0
    try: 
        for step in tqdm(range(1, total_steps), position=0):
            
            action, std = policy.sample(state)
            
            #logger.info(f"____Taking action {step}___")
            standardized_action = form_action(action, pi["outputs"], logger)
            sim.act(standardized_action)
            #logger.info(f"Standard deviation {std}")
            #action_magnitudes.append(np.linalg.norm(action))
            #logger.info(f"Action {action}; Magnitude {np.linalg.norm(action)}")
            reward = form_reward(sim, params["reward"])
            cum_reward += reward
            max_ep_reward = max(max_ep_reward, reward)
            next_state = form_state(sim, pi["inputs"], logger)
            done = sim.done
            e.append(state, action, reward, next_state, done)
            state = next_state

            

            if step % len_episode == 0 or done == 1:
                if done == 1 and "bonus" in params.keys():
                    give_bonus(e, params["bonus"], 30) # 30 is 'average' of demonstration
                    rb.left_append(e)
                else:
                    rb.append(e)
                max_ep_rewards.append(max_ep_reward)
                episode_cum_rewards.append(cum_reward)
                max_ep_reward = -99
                cum_reward = 0

                if HER:
                    he = HER_resample(sim, e, logger)
                    rb.append(he)
                e = Episode()
                sim.close()
                sim = interface.Sim(params['show']) # will tell the sim what initial internal state to hold on to
                sim.compose(composition)
                state = form_state(sim, pi["inputs"])

                
            if step > gradient_after and step % gradient_every == 0:
                for grad in range(0, gradient_every):
                    batch = rb.sample_batch(params['algorithm']['batch_size'])
                    states = batch['states']
                    actions = batch['actions']
                    #print(f"Action range: {actions.min().item()} to {actions.max().item()}")
                    rewards = batch['rewards']
                    next_states = batch['next_states']
                    done = batch['done'] # Recommended to cast this to float
                    #logger.info(f"_____Q Network Update {step}_____")
                    state_actions = torch.cat((states, actions), dim=-1)
                    q1_current = critic1(state_actions)
                    q2_current = critic2(state_actions)
                    mean_reward_per_grad.append(rewards.mean().item())
                    
                    plottable = q1_current.detach().cpu().numpy()  # shape (batch_size, 1)
                    if step % 100 == 0:
                        q1s_mean.append(plottable.mean())
                        q1s_max.append(plottable.max())
                        q1s_min.append(plottable.min())
                        q1s_std.append(plottable.std())
                        q1s.append(q1_current.mean().item()) # for plotting
                        q2s.append(q2_current.mean().item())

                    #logger.info(f"Q1 {q1_current.detach().numpy().mean()}; Q2 {q2_current.detach().numpy().mean()}")   
                    #logger.info(f"Mean reward {rewards.detach().numpy().mean()}; 1 reward? {np.any(rewards.numpy() == 1)}")
                    
                    
                    with torch.no_grad():
                        next_actions, next_log_probs = policy.sample(next_states)
                        next_state_actions = torch.cat((next_states, next_actions), dim=-1)
                        q1_next = qnetwork1(next_state_actions)
                        q2_next = qnetwork2(next_state_actions)
                        min_q_next = torch.min(q1_next, q2_next)
                        target_q = rewards + gamma * (1 - done) * (min_q_next - (alpha * next_log_probs))
                        #print("rewards shape:", rewards.shape)
                        #print("done shape:", done.shape)
                        #print("min_q_next shape:", min_q_next.shape)
                        #print("log_probs shape:", next_log_probs.shape)
                        # clamp the change around the reward scale
                        #target_q = torch.clamp(raw_target, -1.0, 1.0)   
                    
                    #logger.info(f"Q1 next {q1_next.mean().item()}; Q2 next {q2_next.mean().item()}")
                    q1_loss = F.mse_loss(q1_current, target_q)
                    q2_loss = F.mse_loss(q2_current, target_q)
                    q_loss = q1_loss + q2_loss # Idea from Spinningup
                    #logger.info(f"Q1 Loss {q1_loss.mean().item()}; Q2 Loss {q2_loss.detach().numpy().mean()}") 

                    # Here spinning up does loss_q = lossq1 + lossq2
                    
                    # Critic Backpropogation #

                    q1_optimizer.zero_grad() 
                    q2_optimizer.zero_grad()
                    q_loss.backward()
                    """  
                    q1_loss.backward()  
                    q2_loss.backward()
                    """
                    #torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
                    #torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
                    q1_optimizer.step()
                    q2_optimizer.step()
                    
                        
                    mix = 0.995 # Polyak average
                    for param, target_param in zip(critic1.parameters(), qnetwork1.parameters()):
                        target_param.data.mul_(mix)
                        target_param.data.add_((1-mix) * param.data)
                    
                    for param, target_param in zip(critic2.parameters(), qnetwork2.parameters()):
                        target_param.data.mul_(mix)
                        target_param.data.add_((1 - mix) * param.data)
                    #logger.info(f"_____Actor Update {step}_____")
                    # Actor update #
                    new_actions, log_probs = policy.sample(states)
                    stds.append(log_probs.mean().item())
                    #logger.info(f"Action mean {new_actions.detach().numpy().mean()}; Log prob mean {log_probs.detach().numpy().mean()};")
                    new_state_actions = torch.cat((states, new_actions), dim=-1)
                    q1_val_new = critic1(new_state_actions)
                    q2_val_new = critic2(new_state_actions)
                    q_val_new = torch.min(q1_val_new, q2_val_new)
                    #logger.info(f"Q1 {q1_val_new.detach().numpy().mean()}; Q2 {q2_val_new.detach().numpy().mean()};")
                    policy_loss = (alpha * log_probs - q_val_new).mean()
                    #logger.info(f"Policy Loss {policy_loss.detach().numpy().mean()};")
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                    policy_optimizer.step()
                    
                    ### Entropy coefficient update
                    
                    if "A-tuning" in params.keys():
                        alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
                        logger.info(f"Alpha Loss {alpha_loss.detach().numpy().mean()}; Alpha {alpha.item()};")
                        #print(alpha.item())

                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()

                        # Update alpha after gradient step
                        alpha = log_alpha.exp()
                        alphas.append(alpha.item())
                        #
                    
                    ### plotting ###

                    if step % 100 == 0:
                        q_losses.append(q_loss.detach().numpy())
                        pi_losses.append(policy_loss.detach().numpy())
                    
                
                
            if step > gradient_after and step % save_every == 0:
                # Saving models after training #     
                global trained_policy
                trained_policy = policy
                safe_save_model(trained_policy, parameter_file_name, params['algorithm']["networks_save_name"], "pi", save_state_dict=True)
                global trained_critic1
                trained_critic1 = critic1
                safe_save_model(trained_critic1, parameter_file_name, params['algorithm']["networks_save_name"], "Q1", save_state_dict=True)
                global trained_critic2
                trained_critic2 = critic2
                safe_save_model(trained_critic2, parameter_file_name, params['algorithm']["networks_save_name"], "Q2", save_state_dict=True)
            steps_taken += 1
            #print("Replay buffer was saved to", params["rb_save_name"])
            
        sim.close()
        time.sleep(2)
        del sim # end training loop
    except KeyboardInterrupt:
       print("Exiting")
    common_path = "figures/"+parameter_file_name + "/"+params['algorithm']["networks_save_name"]

    os.makedirs(common_path, exist_ok=True)
    plt.figure()
    
    plt.plot(range(0, len(q1s)), q1s, label="Q1")
    plt.plot(range(0, len(q2s)), q2s, label="Q2")
    plt.title("Q values")
    plt.savefig(common_path+"/qs.png")                
    plt.figure() 
    plt.plot(range(0, len(q_losses)), q_losses)
    plt.title("Q-losses")
    plt.savefig(common_path+"/qlosses.png")                
    plt.figure() 
    plt.plot(range(0, len(pi_losses)), pi_losses)
    plt.title("Policy-losses")
    plt.savefig(common_path+"/policy_losses.png")
    plt.figure()         
    plt.plot(range(0, len(max_ep_rewards)), max_ep_rewards)
    plt.title("Max rewards / episode")
    plt.savefig(common_path+"/max_ep_rewards.png")
    plt.figure()         
    plt.plot(range(0, len(stds)), stds)
    plt.title("Policy entropy")
    plt.savefig(common_path+"/stds.png")
    plt.figure()         
    plt.plot(range(0, len(q1s_mean)), q1s_mean)
    plt.title("Q1 mean")
    plt.savefig(common_path+"/q1_mean.png")
    q1s_mean = []
    plt.figure()         
    plt.plot(range(0, len(q1s_max)), q1s_max)
    plt.title("Q1 max")
    plt.savefig(common_path+"/q1s_max.png")
    plt.figure()         
    plt.plot(range(0, len(q1s_min)), q1s_min)
    plt.title("Q1 min")
    plt.savefig(common_path+"/q1s_min.png")
    plt.figure()         
    plt.plot(range(0, len(q1s_std)), q1s_std)
    plt.title("Q1 standard deviation")
    plt.savefig(common_path+"/q1s_std.png")
    plt.figure()         
    plt.plot(range(0, len(episode_cum_rewards)), episode_cum_rewards)
    plt.title("Cumulative reward / episode")
    plt.savefig(common_path+"/episode_cum_rewards.png")
    plt.figure()         
    plt.plot(range(0, len(alphas)), alphas)
    plt.title("Alpha")
    plt.savefig(common_path+"/alpha.png")
    plt.figure()         
    plt.plot(range(0, len(mean_reward_per_grad)), mean_reward_per_grad)
    plt.title("Mean reward per gradient update")
    plt.savefig(common_path+"/mean_reward_per_grad.png")
    
    safe_save_model(policy, parameter_file_name, params['algorithm']["networks_save_name"], "pi", save_state_dict=True)
    safe_save_model(critic1, parameter_file_name, params['algorithm']["networks_save_name"], "Q1", save_state_dict=True)
    safe_save_model(critic2, parameter_file_name, params['algorithm']["networks_save_name"], "Q2", save_state_dict=True)

    return policy
    
def HER_resample(sim, e, logger, mode="random_future"):
    #logger.info(f"__HER Sampling__")
    he = Episode()
    for t in range(0, len(e.steps) -1):
        if mode == "random_future":
            future = random.randint(t+2, t + min(8, len(e.steps) - t)) -1
        elif mode == "final":
            future = len(e.steps) - 1
        
        final = e.steps[future]
        state = final.state
        achieved_cube_pos = state[3:6] # needs permissability
        new_goal = achieved_cube_pos
        HER_reward = sim.calculate_reward(e.steps[t].state[0:3], e.steps[t].state[3:6], new_goal, e.steps[t].action) # needs permissability
        
        
        H_state = copy.deepcopy(e.steps[t].state)
        H_state[10:13] = achieved_cube_pos  # needs permissability
        H_state = torch.tensor(H_state, dtype=torch.float32)
        H_next_state = copy.deepcopy(e.steps[t].next_state)
        H_next_state[10:13] = achieved_cube_pos # needs permissability
        he.append(H_state, e.steps[t].action, torch.tensor(HER_reward, dtype=torch.float32), torch.tensor(H_next_state, dtype=torch.float32), 0) # done = 0
        
        #logger.info(f"Current {t}, {e.steps[t].state[3:6]} with goal {e.steps[t].state[10:13]}")
        #logger.info(f"Future from step {future} acheived {achieved_cube_pos};")
        #logger.info(f"Assigned {achieved_cube_pos} to current goal {H_state[10:13]};")
        #logger.info(f"EEF at {H_state[0:3]}; Cube position {H_state[3:6]}; Reward {HER_reward}")
    return he

def give_bonus(episode, bonus, steps_from_done=None):
    if not steps_from_done:
        steps_from_done = len(episode.steps)
    for step in range(len(episode.steps) - 1, len(episode.steps) - 1 - steps_from_done): # Excluding the last step
        episode.steps[step].reward += bonus

def collect_data_from_policy(sim, policy, rb, num_action_episodes, len_episode, rb_save_name):
    for action_episode in tqdm(range(0, num_action_episodes), position=1, leave=False):
        e = Episode()
        state = sim.observe()
        for step in range(0, len_episode):
            action = policy.sample(state)[0].detach().numpy()
            sim.act(action)
            reward = sim.reward()
            next_state = sim.observe()
            e.append(state, action, reward, next_state)
            state = next_state
            if reward.item() == sim.reward_for_raise:
                pass
                #sim.reset()
                #rb.append(e)
                #rb.save(rb_save_name)
                print("********** Automatically arrived at reward***********")
                print(reward.item())
                #input("Proceed?")
                #break
            else:
                pass
                
            if reward.item() <= -1:
                print("********** Arrived at very negative reward***********")
                print(reward.item())
                #input("Proceed?")
            
        sim.reset(has_renderer=True)
        
        # Append each episode #
        rb.append(e)
    # Save replay buffer after this iteration's data collection is complete    
    rb.save(rb_save_name)   

def collect_teleop_data(sim, rb, rb_save_name):
    try:
        speed = 0.1
        sim.env = None
        sim.reset(has_renderer=True, use_sim_camera=True)
        e = Episode()
        state = sim.observe()
        while True:
            print("EEF", sim.eef_pos, "Cube", sim.cube_pos)
            action = np.array([0.0,0.0,0.0,0.0])
            trigger = input("Button: ")
            if trigger == "q":
                action[0] = speed
            elif trigger == "w":
                action[1] = speed
            elif trigger == "e":
                action[2] = speed
            elif trigger == "r":
                action[3] = speed
            elif trigger == "p":
                sim.take_photo(int(random.random()*100))
            else:
                print("Assigning speed!")
                try:
                    speed = float(trigger)
                except ValueError:
                    print("OOPS!")
                    continue
            sim.act(action, w_video=True)
            reward = sim.reward()   
            next_state = sim.observe()
            done = sim.done
            e.append(state, action, reward, next_state, 0)
            if done:
                break
                
            print("State:", state, "\nAction:", action, "\nReward:", reward, "\nState':", next_state, "\nDone?", done)         
            state = next_state
            
            
        
    except KeyboardInterrupt:
        pass
    sim.env = None
    sim.reset()
    rb.append(e)
    rb.save(rb_save_name)
    return e

def standardize_keyboard(key, speed=0.3):
    action = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    
    if key == "q":
        action[0] = speed
    elif key == "w":
        action[1] = speed
    elif key == "e":
        action[2] = speed
    elif key == "r":
        action[6] = speed
    else:
        try:
            speed = float(key)
            print("Assigning speed!")
        except ValueError:
            print("OOPS!")
    return action, speed

def teleop(composition):
    import interface
    sim = interface.Sim(show=True)
    sim.compose(composition)
    speed = 0.3
    while True:
        print("Cube position:", sim.get_cube_pos())
        print("Cube goal position:", sim.get_initial_cube_goal())
        print("Cube displacement:", sim.cube_cube_displacement())
        print("Reward:", sim.cube_cube_distance(1.0))
        print("Gripper position:", sim.get_current_grasp())
        action, speed = standardize_keyboard(input("Button: "), speed )
        sim.act(action)
        
def test(params, composition, policy, inter_test_memory = None, router=None, cut_component=False, epilogue=None):
    # hard coded - not generalized
    if router:
        really_do = True
        from real import test_cartesian
        real_robot = test_cartesian.Real(router, composition)
    else:
        really_do = False
    import time
    import interface
    pi = params["pi"]
    sim = interface.Sim(show=True) # will tell the sim what initial internal state to hold on to
    sim.compose(composition, inter_test_memory)
    num_steps = 1000
    
    print("Episodes of len", num_steps)
    input("<press any key>")
    done = 0
    state = form_state(sim, pi["inputs"])
    try:
        for step in range(0, num_steps):

            action = policy.deterministic_action(state)
            #action = policy.sample(state)[0].detach().numpy()
            sim.act(form_action(action, pi["outputs"]))
            if really_do:
                real_robot.really_do(action)
                
                #if input("Proceed(y/n)?:") == "n":
                #    return
            done = sim.done
            print(form_reward(sim, params["reward"]))
            time.sleep(0.1)
            # One last done condition 
            done = form_reward(sim, params["reward"]) > -.0134 and cut_component
            if done or step >= 1000:
                time.sleep(2)
                break
               
            
            
            state = form_state(sim, pi["inputs"])
    except KeyboardInterrupt:
        print("Exiting...")
    if epilogue:
        sim.epilogue(epilogue)
    inter_test_memory = sim.get_inter_test_memory()
    sim.close()

    print("Done testing visually.")
    return inter_test_memory


def load_replay_buffer(rb_name):
    file_path = "replays/" + rb_name + ".tmp"
    with open(file_path, "rb") as f:
        replay_buffer = pickle.load(f)
    print("Replay buffer named", rb_name, "with", len(replay_buffer.episodes), "episodes has been loaded.")
    return replay_buffer

def load_pushed_policy(learned_component, params):
    
    filename = "policies/pushed/" + learned_component+".pt"

    policy = PolicyNetwork(params["networks"])

    policy.load_state_dict(torch.load(filename))

    return policy

def load_saved_model(parameters, params, model_type):
    parameters = parameters.split('/')[1]
    component_name = params["algorithm"]["networks_save_name"]
    filename = "sac_models/" + parameters + "/" + component_name + "/"+model_type+".pt"
    print("Generating new configuration...")
    if model_type == "pi":
        model = PolicyNetwork(params["networks"])
    elif model_type == "Q1":
        model = QNetwork(params["networks"])
    elif model_type == "Q2":
        model = QNetwork(params["networks"])
    model.load_state_dict(torch.load(filename))

    return model
    
def load_saved_qnetwork(params, model_type):
    trained_qnetwork = QNetwork(params['networks'])
    filename = "sac_models/" + parameters + "/" + component_name + "/pi.pt"
    trained_qnetwork.load_state_dict(torch.load(qnetwork_path))
    return trained_qnetwork

def redundancy_check(learning_component):
    for policy in os.listdir("policies/committed"):
        if learning_component + ".pt" == policy:
            print("Policy found for component", learning_component,"in committed policies...")
            return True
    for policy in os.listdir("policies/pushed"):
        if learning_component + ".pt" == policy:
            print("Policy found for component", learning_component,"in pushed policies...")
            return True
    return False

def commit(learned_component, policy):
    if input("Commit?(y/n): ") == "y":
        pass
    else:
        return False
    
    data_to_save = policy.state_dict()
    filename = "policies/committed/" + learned_component + ".pt"
    # Get the target directory from filename
    target_dir = os.path.dirname(os.path.abspath(filename))
    
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Use a temporary file in the same directory for atomic write.
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp_file:
        temp_filename = tmp_file.name
        torch.save(data_to_save, tmp_file)
    
    # Atomically replace the target file with the temporary file.
    os.replace(temp_filename, filename)
    print(f"Model saved to {filename}")
    #print(f"Model successfully saved to {filename}")  
    return True
    
def push():
    print(os.listdir("policies/committed"))
    if input("Push?(y/n): ") == "y":
        pass
    else:
        return False
    src = pathlib.Path('policies/committed')
    dst = pathlib.Path('policies/pushed')

    # Make sure destination exists
    dst.mkdir(parents=True, exist_ok=True)

    for file in src.glob('*.pt'):
        new_path = dst / file.name
        file.rename(new_path)   # or: file.replace(new_path)
        print(f"Moved {file} → {new_path}")
    return True

def safe_save_model(model, parameters, component_name, model_type, save_state_dict=True):
    # Choose the data to save
    data_to_save = model.state_dict() if save_state_dict else model
    filename = "sac_models/" + parameters + "/" + component_name + "/" + model_type + ".pt"
    # Get the target directory from filename
    target_dir = os.path.dirname(os.path.abspath(filename))
    
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Use a temporary file in the same directory for atomic write.
    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False) as tmp_file:
        temp_filename = tmp_file.name
        torch.save(data_to_save, tmp_file)
    
    # Atomically replace the target file with the temporary file.
    os.replace(temp_filename, filename)
    #print(f"Model successfully saved to {filename}")  
   
def setup_logger(params):
    
    experiment_name = "WIP"
    logger = logging.getLogger(experiment_name)  
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    log_path = "logs/" + experiment_name + ".logs"
    
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

### Main here is to teleop state and rewards ###
if __name__ == "__main__":
    composition = "reset_eef"
    teleop(composition)
        
    
