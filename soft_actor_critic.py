import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.optim as optim

import torchviz
from tqdm import tqdm
import tempfile

import os
import random
import time
import numpy as np
import pickle
import copy

torch.autograd.set_detect_anomaly(True)

class ReplayBuffer:
    def __init__(self):
        self.episodes = []

    def append(self, episode):
        self.episodes.append(episode)
        
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
            "rewards": torch.stack([torch.tensor(step.reward, dtype=torch.float32) for step in batch_steps]),
            "next_states": torch.stack([torch.tensor(step.next_state, dtype=torch.float32) for step in batch_steps])
        }
        assert not torch.isnan(batch["states"]).any()
        assert not torch.isnan(batch["actions"]).any()
        assert not torch.isnan(batch["rewards"]).any()
        assert not torch.isnan(batch["next_states"]).any()
        return batch

    def forget(self, forget_size):
        for i in range(0, forget_size):
            self.episodes.remove(self.episodes[0])
    
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
            
def load_replay_buffer(filepath):
    with open(filepath, "rb") as f:
        replay_buffer = pickle.load(f)
    print("Replay buffer of", len(replay_buffer.episodes), " episodes loaded!")
    return replay_buffer
            
        
        
class Episode:
    def __init__(self):
        self.steps = []
    
    def append(self, state, action, reward, next_state):
        step = Step(state, action, reward, next_state)
        self.steps.append(step)
    
class Step:
    def __init__(self, state, action, reward, next_state):
        self.state = state.detach().numpy()
        self.action = action
        self.reward = reward.detach().numpy()
        self.next_state = next_state.detach().numpy()

class QNetwork(nn.Module):
    def __init__(self, state_action_dim=16, q_value_dim=1, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.q_value_layer = nn.Linear(hidden_dim, q_value_dim)
        
    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        
        x = F.relu(self.fc2(x))
        
        q_value = self.q_value_layer(x)
        
        return q_value
        
    def qvalue(self, state_action):
        q_value = self.forward(state_action)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=9, action_dim=7, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
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
        dist = torch.distributions.TransformedDistribution(normal, torch.distributions.TanhTransform(cache_size=1))
        action = dist.rsample()
        #print("Mean:", mean, "Std:", std, "Action:", action)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob




#       Control Scripts       #



trained_policy = None
trained_qnetwork = None

def train(sim, params, args):
    _policy_losses_over_time = []
    _q_losses_over_time = []
    left_margin = 0
    
    if not args.pi:
        policy = PolicyNetwork()
    else:
        policy = load_saved_policy(args.pi)
    if not args.q1:
        critic1 = QNetwork()
    else:
        critic1 = load_saved_qnetwork(args.q1)
    if not args.q2:
        critic2 = QNetwork()
    else:
        critic2 = load_saved_qnetwork(args.q2)
    
    if not args.rb:
        rb = ReplayBuffer()
    else:
        rb = load_replay_buffer(args.rb)
    
    policy_optimizer = optim.Adam(policy.parameters(), params['policy_lr'])
    q1_optimizer = optim.Adam(critic1.parameters(), params['q1_lr'])
    q2_optimizer = optim.Adam(critic2.parameters(), params['q2_lr'])   
    
    for name, p in critic1.named_parameters():
        print("critic1.",name," requires_grad=",p.requires_grad)

    for name, p in critic2.named_parameters():
        print("critic2.",name," requires_grad=",p.requires_grad)
    
    # Make 'target' networks 
    qnetwork1 = copy.deepcopy(critic1)
    qnetwork2 = copy.deepcopy(critic2)
    
    for p in qnetwork1.parameters():
        p.requires_grad = False
    for p in qnetwork2.parameters():
        p.requires_grad = False  
    
    num_iterations, num_action_episodes, len_episode = params['num_iterations'], params['num_action_episodes'], params['len_episode']
    gamma, alpha = params['gamma'], params['alpha']
    
    avg_succ_rts = []
    while True:
    
        for iteration in tqdm(range(0, num_iterations), position=0):


            if bool(params['human']):
                collect_teleop_data(sim, rb, params['rb_save_name'])
            else:
                collect_data_from_policy(sim, policy, rb, num_action_episodes, len_episode, params['rb_save_name'])
            gradient_steps = params['num_gradient_steps']

            for gradient_step in tqdm(range(0, gradient_steps), position=1, leave=False):
                print("Replay buffer size:", len(rb.episodes), "episodes.")
                batch = rb.sample_batch(params['batch_size'])
                states = batch['states']
                actions = batch['actions']
                rewards = batch['rewards']
                next_states = batch['next_states']
                
                if (rewards > .99).any():
                    print(f"⚡️ Positive reward in this batch at gradient step {gradient_step}")
                    input("Proceed?")
                
                # Critic update #
                state_actions = torch.cat((states, actions), dim=-1)
                q1_current = critic1(state_actions)
                q2_current = critic2(state_actions)


                
                           
                with torch.no_grad():
                    next_actions, next_log_probs = policy.sample(next_states)
                    next_state_actions = torch.cat((next_states, next_actions), dim=-1)
                    q1_next = qnetwork1(next_state_actions)
                    q2_next = qnetwork2(next_state_actions)
                    min_q_next = torch.min(q1_next, q2_next)
                    raw_target      = rewards + gamma * (min_q_next - (alpha * next_log_probs))
                        # clamp the change around your reward scale
                    target_q = torch.clamp(raw_target, -1.0, 1.0)

                    
                print("\nQ-next:", min_q_next.mean(), "+/-", min_q_next.std().item())
                print("Log_prob:", next_log_probs.mean(), "+/-", next_log_probs.std().item())
                print("Rewards:", rewards.mean(), "+/-", rewards.std().item())
                q1_loss = F.mse_loss(q1_current, target_q)
                q2_loss = F.mse_loss(q2_current, target_q)
                assert not torch.isnan(q1_loss).any()
                assert not torch.isnan(q2_loss).any()

                # Critic Backpropogation #

                q1_optimizer.zero_grad() 
                q2_optimizer.zero_grad()    
                q1_loss.backward()  
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
                q1_optimizer.step()
                q2_optimizer.step()
                
                mix = 0.005 # Polyak average
                for param, target_param in zip(critic1.parameters(), qnetwork1.parameters()):
                    target_param.data.mul_(1 - mix)
                    target_param.data.add_(mix * param.data)
                
                for param, target_param in zip(critic2.parameters(), qnetwork2.parameters()):
                    target_param.data.mul_(1 - mix)
                    target_param.data.add_(mix * param.data)
                
                 # Actor update #
                new_actions, log_probs = policy.sample(states)  
                new_state_actions = torch.cat((states, new_actions), dim=-1)
                q1_val_new = critic1(new_state_actions)
                q2_val_new = critic2(new_state_actions)
                q_val_new = torch.min(q1_val_new, q2_val_new)
                policy_loss = (alpha * log_probs - q_val_new).mean()
                
                print("log_probs.requires_grad?", log_probs.requires_grad, "grad_fn:", log_probs.grad_fn)
                print("q_new_min.requires_grad?", q_val_new.requires_grad, "grad_fn:", q_val_new.grad_fn)
                
                for name, p in policy.named_parameters():
                    print(f"  after backward {name}.grad norm:", None if p.grad is None else p.grad.norm().item())
                        
                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                policy_optimizer.step()
                
                
                

                
                
                
                
            #
            #_q_losses_over_time.append(q_loss.detach().numpy())
            #_policy_losses_over_time.append(policy_loss.detach().numpy())
            # Per iteration plotting
            #
            #avg_succ_rts.append(test(sim, num_episodes=10, render=False))
        """
        plt.plot(range(0, num_iterations + left_margin), _policy_losses_over_time, label="Policy")
        plt.plot(range(0, num_iterations + left_margin), _q_losses_over_time, label="Q-Network")
        plt.legend()
        plt.title("lr policy " + str(params['policy_lr']) + ", lr q " + str(params['q_lr']) + " alpha " + str(params['alpha']))
        plt.show()
        """
        
        global trained_policy
        trained_policy = policy
        safe_save_model(trained_policy, params["policy_save_name"] +".pt", save_state_dict=True)
        global trained_critic1
        trained_critic1 = critic1
        safe_save_model(trained_critic1, params["critic1_save_name"] +".pt", save_state_dict=True)
        global trained_critic2
        trained_critic2 = critic2
        safe_save_model(trained_critic2, params["critic2_save_name"] +".pt", save_state_dict=True)
        
        inp = input("#/n: ")
        while not inp == "n":
            try:
                num_iterations = int(inp)
                left_margin += num_iterations
                break
            except ValueError:
                inp = input("#/n: ")
        if inp == "n":
            break
    #print(num_iterations + left_margin, avg_succ_rts)
    #plt.plot(range(0, num_iterations + left_margin), avg_succ_rts)
    #plt.show()      
   
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
            if reward.item() == 1.0:
                sim.reset()
                rb.append(e)
                rb.save(rb_save_name)
                print("********** Automatically arrived at reward***********")
                print(reward.item())
                #input("Proceed?")
                break
            if reward.item() <= -1:
                print("********** Arrived at very negative reward***********")
                print(reward.item())
                input("Proceed?")
        sim.reset(has_renderer=False)
        rb.append(e)
    save_name = rb_save_name
    rb.save(save_name)   

def collect_teleop_data(sim, rb, rb_save_name):
    try:
        speed = 0.1
        
        sim.reset(has_renderer=True)
        e = Episode()
        state = sim.observe()
        while True:
            action = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            trigger = input("Button: ")
            if trigger == "q":
                action[0] = speed
            elif trigger == "w":
                action[1] = speed
            elif trigger == "e":
                action[2] = speed
            elif trigger == "r":
                action[3] = speed
            elif trigger == "t":
                action[4] = speed
            elif trigger == "y":
                action[5] = speed
            elif trigger == "u":
                action[6] = speed
            elif trigger == "p":
                sim.take_photo(int(random.random()*10))
            else:
                
                print("Assigning speed!")
                try:
                    speed = float(trigger)
                except ValueError:
                    print("OOPS!")
                    continue

            sim.act(action)
            reward = sim.reward()   
            next_state = sim.observe()
            e.append(state, action, reward, next_state)
            print("State:", state, "\nAction:", action, "\nReward:", reward, "\nState':", next_state, "\n")         
            state = sim.observe()
            if reward.item() == 1.0: # To reduce corrupt data, end the episode here
                sim.reset()
                rb.append(e)
                rb.save(rb_save_name)
                break
        
    except KeyboardInterrupt:
        sim.reset()
        rb.append(e)
        rb.save(rb_save_name)

def test(sim, num_episodes=10, render=True):
    import time
    sim.has_renderer = True
    global trained_policy
    num_steps = 50
    successes = 0
    trials = 0
    for episode in range(0, num_episodes):
        sim.reset(render)
        state = sim.observe()
        for step in range(0, num_steps):
            print("State", state)
            action = trained_policy.sample(state)[0].detach().numpy()
            print("Action:", action)        
            sim.act(action) 
            state = sim.observe()
            reward = sim.reward()
            print("\nReward", reward,"\n")
            if reward.item() == 1.0:
                time.sleep(2)
                successes += 1
                break
        trials += 1
        print("Success rate:", successes/trials)
    return successes/trials  
       
def test_single_SAR(sim):
    policy = PolicyNetwork()

    state = sim.observe()
    
    # create tensor out of human-readable "state"
    state = torch.tensor(state)
    
    action, log_prob = policy.sample(state)

def load_saved_policy(policy_path):
    trained_policy = PolicyNetwork()
    trained_policy.load_state_dict(torch.load(policy_path))

    return trained_policy
    
def load_saved_qnetwork(qnetwork_path):
    trained_qnetwork = QNetwork()
    trained_qnetwork.load_state_dict(torch.load(qnetwork_path))
    return trained_qnetwork

def safe_save_model(model, filename, save_state_dict=True):
    """
    Safely save a PyTorch model or its state_dict to a file using an atomic write.
    
    Parameters:
        model (torch.nn.Module): The model to save.
        filename (str): The target filename where the model will be saved.
        save_state_dict (bool): If True, only the model's state_dict will be saved.
                                Otherwise, the entire model is saved.
    """
    # Choose the data to save
    data_to_save = model.state_dict() if save_state_dict else model

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
    print(f"Model successfully saved to {filename}")    
