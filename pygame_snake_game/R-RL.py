import csv
import os
import random
import sys

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from snake_game import SnakeGame
from collections import deque

import cv2

# Game Constants
WIDTH, HEIGHT = 800, 600
BLOCK_SIZE = 20
GRID_WIDTH = WIDTH // BLOCK_SIZE
GRID_HEIGHT = HEIGHT // BLOCK_SIZE

# Neural Network Parameters
STATE_SIZE = 14
output_size = 4
BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
# beta_start = 0.4
# beta_frames = 100000
# beta = min(1.0, beta_start + self.steps_done * (1.0 - beta_start) / beta_frames)

game = SnakeGame(width=WIDTH, height=HEIGHT)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# screen = pygame.Surface((WIDTH, HEIGHT))  # Off-screen rendering
clock = pygame.time.Clock()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# csv save function
def save_weights_to_csv(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for key, weight in state_dict.items():
            writer.writerow([key])
            flat_weights = weight.flatten().tolist()
            writer.writerow(flat_weights)
            writer.writerow([])
    print(f"Weights saved to {path}")


# TODO: step 2 - Noisy Layer
class NoisyLinear(nn.Module):
    """
    Replaced all the dense layer with the noisy layer
    nn.Linear(input_size, 128) is changed to NoisyLinear(input_size, 128),
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mean parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Sigma (noise scale) parameters
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()

        # Register buffers for sampled noise (non-trainable)
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)

    def forward(self, input):
        # print("Input type to NoisyLinear:", type(input))
        self.sample_noise()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(input, weight, bias)

    # had to modify this for to go to the GPU
    def sample_noise(self):
        device = self.weight_mu.device  # get current device of layer
        epsilon_in = self._scale_noise(self.in_features, device)
        epsilon_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def _scale_noise(self, size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()


"""This is the Dueling DQN"""


# class DQN(nn.Module):
#     """Dueling Deep Q-Network with state representation"""
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         # TODO: Step 1 - Add the Duelling DQN
#         """ Disabled the Dueling DQN to test out the Categorical DQN """
#         self.fc = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 128),
#             nn.ReLU(),
#             # Removed this part because you’re to split the output of a common network into value + advantage.
#             # if you added it you'd be redundantly processing the raw input separately for each stream,
#             # which isn’t what Dueling DQN intends.
#             # nn.Linear(128, output_size)
#         )
#
#         # TODO: Step 1 - Add the Duelling DQN
#         """ Disabled the Dueling DQN to test out the Categorical DQN """
#         # Step 1 - added the Duelling DQN where I split the output features from the FC to a value and advantage
#         # streams
#
#         # Value function
#         self.value_stream = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 1),
#         )
#
#         # Advantage Function
#         self.advantage_stream = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 1),
#         )
#
#     def forward(self, x):
#         """Also part of Dueling"""
#         x = self.fc(x)
#         value = self.value_stream(x)
#         advantage = self.advantage_stream(x)
#
#         qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return qvals


# class DQN(nn.Module):
#     """Distributional/Categorical Deep Q-Network with state representation"""
#     # This is overcomplicating things
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#
#         # Distributional RL parameters
#         self.V_MIN = -10
#         self.V_MAX = 10
#         self.N_ATOMS = 51
#         self.output_size = output_size
#         self.DELTA_Z = (self.V_MAX - self.V_MIN) / (self.N_ATOMS - 1)
#
#         self.fc = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 128),
#             nn.ReLU()
#         )
#
#         self.output = NoisyLinear(128, output_size * self.N_ATOMS)
#
#     def forward(self, x):
#         # print("Input type to NoisyLinear:", type(x))
#         x = self.fc(x)
#         logits = self.output(x)
#         logits = logits.view(-1, self.output_size, self.N_ATOMS)
#         probs = F.softmax(logits, dim=2)  # Distribution over atoms
#         return probs

class CNN_DQN(nn.Module):
    def __init__(self, num_actions):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # input: 4 x 84 x 84
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.feature_size = self._get_conv_output((4, 84, 84))  # pass input shape
        self.fc = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions)
        )

    def forward(self, x):
        print(f"Input to CNN: {x.shape}")  # Add this line
        x = self.conv(x)
        print(f"After CNN: {x.shape}")  # Add this line
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))


# TODO: Implement a PrioritizedReplayBuffer

class PrioritizedReplayBuffer:
    """
    I found a better PrioritizedReplayBuffer class that is more efficient but will try it later
    link: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03.per.ipynb
    """

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_prio  # new experience gets max priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6


"""This is the Dueling DQN"""


# class DQN(nn.Module):
#     """Dueling Deep Q-Network with state representation"""
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         # TODO: Step 1 - Add the Duelling DQN
#         """ Disabled the Dueling DQN to test out the Categorical DQN """
#         self.fc = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 128),
#             nn.ReLU(),
#             # Removed this part because you’re to split the output of a common network into value + advantage.
#             # if you added it you'd be redundantly processing the raw input separately for each stream,
#             # which isn’t what Dueling DQN intends.
#             # nn.Linear(128, output_size)
#         )
#
#         # TODO: Step 1 - Add the Duelling DQN
#         """ Disabled the Dueling DQN to test out the Categorical DQN """
#         # Step 1 - added the Duelling DQN where I split the output features from the FC to a value and advantage
#         # streams
#
#         # Value function
#         self.value_stream = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 1),
#         )
#
#         # Advantage Function
#         self.advantage_stream = nn.Sequential(
#             NoisyLinear(input_size, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 1),
#         )
#
#     def forward(self, x):
#         """Also part of Dueling"""
#         x = self.fc(x)
#         value = self.value_stream(x)
#         advantage = self.advantage_stream(x)
#
#         qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return qvals


# DQN AGENT
from collections import deque
class DQNAgent:
    def __init__(self):
        # Policy Network
        self.frame_stack = deque(maxlen=4)
        self.policy_net = CNN_DQN(output_size).to(device)
        self.target_net = CNN_DQN(output_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # for categorical DQN
        # self.z = torch.linspace(self.policy_net.V_MIN, self.policy_net.V_MAX, self.policy_net.N_ATOMS).to(device)

        # multistep return
        # Buffer to temporarily store N-step transitions
        self.N_STEP = 3  # You can tune this
        self.n_step_buffer = deque(maxlen=self.N_STEP)

    def stack_frames(self, new_frame):
        # Ensure frame is [1, 84, 84]
        if new_frame.ndim == 2:
            new_frame = new_frame.unsqueeze(0)

        new_frame = new_frame.to(device)

        if len(self.frame_stack) < 4:
            for _ in range(4):
                self.frame_stack.append(new_frame)
        else:
            self.frame_stack.append(new_frame)

        # Stack frames along dim=0 (channels)
        stacked = torch.cat(list(self.frame_stack), dim=0)
        # print(f"stacked shape{stacked.shape}")# [4, 84, 84]
        return stacked


    # def get_state(self, game_state):
    #     head = game_state['snake_head']
    #     food = game_state['food']
    #     body = game_state['snake_body']
    #     direction = game_state['direction']  # Assume direction is a tuple like (dx, dy)
    #
    #     # Normalized head position
    #     grid_x = head[0] / WIDTH
    #     grid_y = head[1] / HEIGHT
    #
    #     # Relative food direction (normalized)
    #     dx = (food[0] - head[0]) / WIDTH
    #     dy = (food[1] - head[1]) / HEIGHT
    #
    #     # Snake direction (one-hot encoded)
    #     dir_up = 1 if direction == (0, -BLOCK_SIZE) else 0
    #     dir_right = 1 if direction == (BLOCK_SIZE, 0) else 0
    #     dir_down = 1 if direction == (0, BLOCK_SIZE) else 0
    #     dir_left = 1 if direction == (-BLOCK_SIZE, 0) else 0
    #
    #     # Danger detection (binary)
    #     def is_danger(pos):
    #         x, y = pos
    #         return (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or (x, y) in body)
    #
    #     left = (head[0] - BLOCK_SIZE, head[1])
    #     right = (head[0] + BLOCK_SIZE, head[1])
    #     up = (head[0], head[1] - BLOCK_SIZE)
    #     down = (head[0], head[1] + BLOCK_SIZE)
    #
    #     danger_left = int(is_danger(left))
    #     danger_right = int(is_danger(right))
    #     danger_up = int(is_danger(up))
    #     danger_down = int(is_danger(down))
    #
    #     # Snake length normalized
    #     max_possible_length = (GRID_WIDTH * GRID_HEIGHT) - 1
    #     snake_length = len(body) / max_possible_length
    #
    #     # Build enriched state vector
    #     state = [
    #         grid_x, grid_y,  # normalized head position
    #         dx, dy,  # relative food position
    #         danger_left,
    #         danger_right,
    #         danger_up,
    #         danger_down,
    #         dir_up,
    #         dir_right,
    #         dir_down,
    #         dir_left,
    #         snake_length,
    #         1.0  # normalized snake length
    #     ]
    #
    #     return torch.FloatTensor(state)
    #
    #     # I modified this to accumulate the multistep return experience
    def get_state(self):
        surface = pygame.display.get_surface()
        raw_frame = pygame.surfarray.array3d(surface)  # shape: (W, H, 3)

        # Convert to grayscale and resize
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)  # shape: (H, W)
        resized = cv2.resize(gray, (84, 84))  # shape: (84, 84)
        normalized = resized / 255.0  # normalize
        tensor = torch.tensor(normalized, dtype=torch.float32)  # shape: (84, 84)
        return tensor


    def remember(self, state, action, reward, next_state, done):
        # Convert to CPU tensors for storage
        state = state.cpu().clone().detach() if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
        next_state = next_state.cpu().clone().detach() if isinstance(next_state, torch.Tensor) else torch.FloatTensor(
            next_state)

        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step_buffer.maxlen:
            return

        # Build N-step experience
        reward_sum = 0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            reward_sum += (GAMMA ** idx) * r
            if d:
                break

        state_n, action_n, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

        experience = (state_n, action_n, reward_sum, next_state_n, done_n)
        self.memory.add(experience)

    # def act(self, state):
    #     """This is for the Dueling DQN"""
    #     if random.random() < self.epsilon:
    #         return random.randint(0, 3)
    #
    #     with torch.no_grad():
    #         q_values = self.policy_net(state)
    #         return q_values.argmax().item()


    def act(self, state):
        """This is for the Dueling DQN"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            if isinstance(state, tuple):
                state = state[0]
            state = state.unsqueeze(0) if state.dim() == 1 else state
            q_values = self.policy_net(state)
            action = q_values.argmax(1).item()
        return action

    # def act(self, state):
    #     """This is for the Categorical DQN"""
    #     if random.random() < self.epsilon:
    #         return random.randint(0, 3)
    #
    #     with torch.no_grad():
    #         if isinstance(state, tuple):
    #             state = state[0]
    #         state = state.unsqueeze(0) if state.dim() == 1 else state  # Ensure proper shape, # Shape: (1, num_actions, N_ATOMS)
    #         probs = self.policy_net(state)
    #         z = torch.linspace(self.policy_net.V_MIN, self.policy_net.V_MAX, self.policy_net.N_ATOMS).to(device)
    #         q = (probs * z).sum(dim=2)  # Shape: (1, num_actions)
    #         action = q.argmax(1).item()
    #     return action

    def learn(self):
        def project_distribution(next_probs, rewards, dones, gamma, z, V_min, V_max):
            batch_size = rewards.shape[0]
            N_ATOMS = z.size(0)
            delta_z = (V_max - V_min) / (N_ATOMS - 1)

            projected = torch.zeros((batch_size, N_ATOMS), device=rewards.device)

            Tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * z.unsqueeze(0)
            Tz = Tz.clamp(min=V_min, max=V_max)

            b = (Tz - V_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l[(u > 0) * (l == u)] -= 1
            u[(l < N_ATOMS - 1) * (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * N_ATOMS, batch_size).long().unsqueeze(1).to(rewards.device)

            projected.view(-1).index_add_(
                0, (l + offset).view(-1),
                (next_probs * (u.float() - b)).view(-1)
            )

            projected.view(-1).index_add_(
                0, (u + offset).view(-1),
                (next_probs * (b - l.float())).view(-1)
            )

            return projected

        """For the Dueling DQN"""
        # 1. Check if enough samples are available in replay buffer
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        # 2. Anneal beta (importance-sampling correction) from 0.4 to 1.0 over time
        beta_start = 0.4
        beta_frames = 100000
        beta = min(1.0, beta_start + self.steps_done * (1.0 - beta_start) / beta_frames)

        # 3. Sample batch from prioritized replay buffer
        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta)

        # 4. unpacks them into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to GPU
        states = torch.stack(states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # 5. Compute  Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        current_q = current_q.squeeze()

        # Compute Target Q values
        with torch.no_grad():
            # next_q = self.target_net(next_states).max(1)[0]
            # Double DQN
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * (GAMMA ** self.N_STEP) * next_q

        # Compute loss and TD errors - Temporal Difference Error
        td_errors = target_q - current_q
        loss = (td_errors.pow(2) * weights).mean()  # Weight loss by importance-sampling weights

        # Optimize the model
        self.optimizer.zero_grad()
        # Back propagation
        loss.backward()
        self.optimizer.step()

        # Update priorities in replay buffer based on TD errors
        td_errors_np = td_errors.detach().cpu().numpy()  # pass the td_error from the GPU to the CPU to, so you can pass
        # it in numpy
        self.memory.update_priorities(indices, np.abs(td_errors_np))

        # Decay epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # Update target network periodically
        # TODO: might want to consider changing to 1000
        if self.steps_done % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

        if done:
            self.n_step_buffer.clear()

    # def learn(self):
    #     def project_distribution(next_probs, rewards, dones, gamma, z, V_min, V_max):
    #         batch_size = rewards.shape[0]
    #         N_ATOMS = z.size(0)
    #         delta_z = (V_max - V_min) / (N_ATOMS - 1)
    #
    #         projected = torch.zeros((batch_size, N_ATOMS), device=rewards.device)
    #
    #         Tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * z.unsqueeze(0)
    #         Tz = Tz.clamp(min=V_min, max=V_max)
    #
    #         b = (Tz - V_min) / delta_z
    #         l = b.floor().long()
    #         u = b.ceil().long()
    #
    #         l[(u > 0) * (l == u)] -= 1
    #         u[(l < N_ATOMS - 1) * (l == u)] += 1
    #
    #         offset = torch.linspace(0, (batch_size - 1) * N_ATOMS, batch_size).long().unsqueeze(1).to(rewards.device)
    #
    #         projected.view(-1).index_add_(
    #             0, (l + offset).view(-1),
    #             (next_probs * (u.float() - b)).view(-1)
    #         )
    #
    #         projected.view(-1).index_add_(
    #             0, (u + offset).view(-1),
    #             (next_probs * (b - l.float())).view(-1)
    #         )
    #
    #         return projected
    #
    #     """For the Categorical DQN"""
    #     if len(self.memory.buffer) < BATCH_SIZE:
    #         return
    #
    #     beta = min(1.0, 0.4 + self.steps_done * (1.0 - 0.4) / 100000)
    #     batch, indices, weights = self.memory.sample(BATCH_SIZE, beta)
    #
    #     # Validate batch
    #     if not batch:
    #         return
    #
    #     # Unpack and convert to tensors
    #     states = []
    #     actions = []
    #     rewards = []
    #     next_states = []
    #     dones = []
    #
    #     # Process each experience in the batch
    #     for experience in batch:
    #         s, a, r, ns, d = experience
    #         states.append(s if isinstance(s, torch.Tensor) else torch.FloatTensor(s))
    #         actions.append(a)
    #         rewards.append(r)
    #         next_states.append(ns if isinstance(ns, torch.Tensor) else torch.FloatTensor(ns))
    #         dones.append(d)
    #
    #     try:
    #         # Stack tensors and move to device
    #         states = torch.stack(states).to(device)
    #         next_states = torch.stack(next_states).to(device)
    #         actions = torch.LongTensor(actions).to(device)
    #         rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    #         dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
    #         weights = torch.FloatTensor(weights).to(device)
    #
    #     except RuntimeError as e:
    #         print(f"Error creating tensors: {e}")
    #         return
    #
    #     # Step 1: Get distributional predictions
    #     dist = self.policy_net(states)
    #     dist = dist[range(len(batch)), actions]  # Use len(batch) in case of partial batch
    #
    #     # Step 2: Get next-state distribution and project it
    #     with torch.no_grad():
    #         next_probs = self.policy_net(next_states)
    #         next_q = (next_probs * self.z).sum(dim=2)
    #         next_actions = next_q.argmax(dim=1)
    #         next_dist = self.target_net(next_states)
    #         next_dist = next_dist[range(len(batch)), next_actions]
    #
    #         target_dist = project_distribution(
    #             next_dist, rewards, dones,
    #             gamma=(GAMMA ** self.N_STEP),
    #             z=self.z.to(device),
    #             V_min=self.policy_net.V_MIN,
    #             V_max=self.policy_net.V_MAX
    #         )
    #
    #     # Step 3: Compute loss
    #     log_probs = torch.log(dist.clamp(min=1e-6))
    #     loss_per_sample = -(target_dist * log_probs).sum(dim=1)  # Per-sample loss
    #     loss = (loss_per_sample * weights).mean()  # Weighted mean loss
    #
    #     # Step 4: Optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     # Step 5: Update priorities
    #     errors = loss_per_sample.detach().abs().cpu().numpy()
    #     self.memory.update_priorities(indices, errors)
    #
    #     self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    #     if self.steps_done % 1000 == 0:
    #         self.target_net.load_state_dict(self.policy_net.state_dict())
    #
    #     self.steps_done += 1

# Training Setup moving to "GPU"
agent = DQNAgent()
agent.policy_net.to(device)
agent.target_net.to(device)
scores = []
mean_scores = []
best_mean_score = float('-inf')

# TODO: Change the file name
# Model Loading
WEIGHT_PATH = 'RAINBOW WEIGHTS/CNN_weights_For_RW.pth'
RETRAIN = True
if os.path.exists(WEIGHT_PATH):
    # soon In pytouch, this code below would not be able to run without this {weights_only = True}, check for the
    # updates overtime.
    agent.policy_net.load_state_dict(torch.load(WEIGHT_PATH, map_location=torch.device('cpu')))
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.epsilon = 0.2 if RETRAIN else EPSILON_END

    print("Loaded saved weights")

# Training Loop
for episode in range(5000):
    frame = agent.get_state()
    current_state = agent.stack_frames(frame).unsqueeze(0).to(device)
    print("Current state shape:", current_state.shape)
    current_score = game.get_state()["score"]
    high_score = game.get_state()["highscore"]
    state = game.reset()
    total_reward = 0
    # state.shape = [BATCH_SIZE, 4, 84, 84]
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
                # print("Saved weights in .pth")
                pygame.quit()
                sys.exit()

        action = agent.act(current_state)
        next_state, reward, done = game.step(action)
        # reward = reward.to(device)
        next_state_processed = agent.get_state().to(device)  # get the processed state and move to GPU
        next_state_processed = agent.stack_frames(next_state_processed).unsqueeze(0).to(device)        # Store experience with negative reward for collisions
        agent.remember(current_state, action, reward, next_state_processed, done)
        agent.learn()

        current_state = next_state_processed
        total_reward += reward

        # Rendering
        game.render(screen, clock.get_fps())
        pygame.display.flip()
        clock.tick(120)  # Reduce speed for better observation

    # Episode statistics
    scores.append(total_reward)
    mean_score = np.mean(scores[-100:])
    mean_scores.append(mean_score)

    # Save best model
    if mean_score > best_mean_score:
        best_mean_score = mean_score
        torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)
        print("Saved new weights")

    # Save weights to CSV every 500 episodes
    # if episode % 500 == 0 and episode != 0:
    #     save_weights_to_csv(agent.policy_net.state_dict(), "csv files/DQN-Weights.csv")

    print(f"Ep {episode:04d} | Score: {total_reward:3.0f} | ε: {agent.epsilon:.3f} | Mean: {mean_score:.1f} | "
          f"highscore : {high_score} | current_score: {current_score}")

# Final Save at the 5000 episode
# torch.save(agent.policy_net.state_dict(), WEIGHT_PATH)

# Note Csv files of the weights are large
# Save to CSV
# save_weights_to_csv(agent.policy_net.state_dict(), "csv files/DQN-Weights.csv")
# print("Saved weights to scv file")
pygame.quit()
