import random
import numpy as np
import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from collections import OrderedDict

import numpy as np
#pip install faiss
import faiss
import Shared

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k + 1)
        predictions = self.y[indices[0][1:]]
        return predictions

class QNetwork(nn.Module):
    def __init__(self, params):
        super(QNetwork, self).__init__()
        self.f1 = nn.Linear(params['state_size'], params['first_layer_size'])
        self.f2 = nn.Linear(params['first_layer_size'], params['second_layer_size'])
        self.f3 = nn.Linear(params['second_layer_size'], params['number_of_actions']) # phidot, psidot actions

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, n_dim, n_actions, hidden_dim1=128, hidden_dim2=128):
        super().__init__()
        self.linear1 = nn.Linear(n_dim+n_actions, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, n_dim)

    def forward(self, abstract_states, actions):
        abstract_states_for_all_actions = abstract_states.unsqueeze(1).repeat(1, len(actions), 1)
        actions_for_all_states = actions.unsqueeze(0).repeat(abstract_states.shape[0], 1, 1)
        za = torch.cat([abstract_states_for_all_actions, actions_for_all_states], dim=-1)
        za = F.relu(self.linear1(za))
        za = F.relu(self.linear2(za))
        zt = self.linear3(za)
        return zt

class StateEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, mid=256, mid2=256,
                 activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, mid2)
        self.fc3 = nn.Linear(mid2, out_dim)
        self.act = activation

    def forward(self, obs):
        h = self.act(self.fc1(obs))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        return torch.sigmoid(h) # to restrict within 0 to 1

class Model(nn.Module):
    def __init__(self, state_encoder, action_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    def forward(self, x):
        raise NotImplementedError("This model is a placeholder")

    def train(self, boolean):
        self.state_encoder.train = boolean
        self.action_encoder.train = boolean

def square_dist(x, y, dim=1):
    return (x-y).pow(2).sum(dim=dim)

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, abstract_states, transitioned_abstract_states, abstract_next_states, action_embeddings, reward_fixations):
        # Transition loss
        transition_loss = square_dist(abstract_next_states, transitioned_abstract_states).mean()
        equivalence_loss = square_dist(abstract_states, abstract_states + action_embeddings.mean(dim=1)).mean()
        reward_fixation_loss = square_dist(abstract_states, reward_fixations).mean()
        return transition_loss, equivalence_loss, reward_fixation_loss

class DQNAgent():
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.action_bins = params['action_bins']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_minimum = params['epsilon_minimum']
        self.model = QNetwork(params)
        self.model.to(DEVICE)
        self.target_model = QNetwork(params)
        self.target_model.to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model_update_iterations = params['target_model_update_iterations']
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])
        self.params = params

        state_encoder = StateEncoder(params['state_size'], params['abstract_state_space_dimmension'])
        action_encoder = ActionEncoder(params['abstract_state_space_dimmension'], params['number_of_actions'])
        self.abstraction_model = Model(state_encoder, action_encoder)
        self.abstract_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.abstraction_model.parameters()), lr=self.params['abstraction_learning_rate'])
        self.abstraction_memory = collections.deque(maxlen=params['memory_size_for_abstraction'])
        self.abstraction_batch_size = params['batch_size_for_abstraction']
        self.loss_function = Loss()
        self.abstract_state_holders = OrderedDict()
        self.equivalence_weight = params['equivalence_weight']
        self.reward_fixation_in_abstraction = {}
        self.current_iteration = 0

    def get_abstract_reward(self, reward):
        return round(reward * 0.5) / 0.5

    def get_abstract_rewards(self, rewards):
        return tuple(self.get_abstract_reward(reward) for index, reward in enumerate(rewards))

    def on_new_sample(self, state, action, reward, next_state):
        abstract_reward = self.get_abstract_reward(reward)
        if abstract_reward not in self.reward_fixation_in_abstraction:
            with torch.no_grad():
                self.reward_fixation_in_abstraction[abstract_reward] = torch.rand(self.params['abstract_state_space_dimmension'])
        self.memory.append((state, action, reward, next_state))

        self.abstraction_memory.append((state, action, reward, next_state))
        abstract_state_holder_key = (tuple(state), action, abstract_reward)
        if abstract_state_holder_key in self.abstract_state_holders:
            self.abstract_state_holders.move_to_end(abstract_state_holder_key)
        else:
            with torch.no_grad():
                next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                abstract_next_state = self.abstraction_model.state_encoder(next_state_tensor)
                self.abstract_state_holders[abstract_state_holder_key] = (abstract_next_state, next_state)
                if len(self.abstract_state_holders) > self.params['abstract_state_holders_size']:
                    self.abstract_state_holders.popitem(last=False)

    def on_finished(self):
        pass

    def on_episode_start(self, episode_index):
        self.update_all_in_abstract_state_holders()
        if self.params['plot_t-sne'] == True and episode_index != 0 and (episode_index+1) % 20 == 0:
            self.draw_tsne(episode_index)

    def update_all_in_abstract_state_holders(self):
        if len(self.abstract_state_holders) == 0:
            return
        next_states = [value[1] for value in self.abstract_state_holders.values()]
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            updated_abstract_next_states = self.abstraction_model.state_encoder(next_states_tensor)
        updated_abstract_next_states = updated_abstract_next_states.unsqueeze(1)
        for (key, next_state), updated_abstract_next_state in zip(self.abstract_state_holders.items(), updated_abstract_next_states):
            self.abstract_state_holders[key] = (updated_abstract_next_state, next_state[1])

    def draw_tsne(self, episode_index):
        matplotlib.use("Qt5Agg")
        X = np.array([tensor[0].numpy().flatten() for tensor in self.abstract_state_holders.values()])

        # Perform t-SNE
        tsne = TSNE(n_components=2)
        tsne_states = tsne.fit_transform(X)
        if self.params['abstract_state_space_dimmension'] == 2:
            tsne_states = X

        tx_orig, ty_orig = tsne_states[:,0], tsne_states[:,1]
        tx = (tx_orig-np.min(tx_orig)) / (np.max(tx_orig) - np.min(tx_orig))
        ty = (ty_orig-np.min(ty_orig)) / (np.max(ty_orig) - np.min(ty_orig))

        labels = list(self.abstract_state_holders.keys())
        values = list(self.abstract_state_holders.values())
        width = 16000
        height = 16000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        if self.params['plot_reward_fixations'] == True:
            rewards = np.array([tensor.numpy().flatten() for tensor in self.reward_fixation_in_abstraction.values()])
            labels_rewards = list(self.reward_fixation_in_abstraction.keys())
            if self.params['abstract_state_space_dimmension'] == 2:
                tsne_rewards = rewards
            else:
                tsne_rewards = tsne.fit_transform(rewards)
                print("WARNING: tsne rewards on higher than 2 dimmension is inaccurate")
            tx_rewards_orig, ty_rewards_orig = tsne_rewards[:,0], tsne_rewards[:,1]
            tx_rewards = (tx_rewards_orig-np.min(tx_rewards_orig)) / (np.max(tx_rewards_orig) - np.min(tx_rewards_orig))
            ty_rewards = (ty_rewards_orig-np.min(ty_rewards_orig)) / (np.max(ty_rewards_orig) - np.min(ty_rewards_orig))
                
            tx = (tx_orig-np.min(tx_rewards_orig)) / (np.max(tx_rewards_orig) - np.min(tx_rewards_orig))
            ty = (ty_orig-np.min(ty_rewards_orig)) / (np.max(ty_rewards_orig) - np.min(ty_rewards_orig))
            for i, coord in enumerate(tsne_rewards):
                x = tx_rewards[i]
                y = ty_rewards[i]
                text = "r: " + str(labels_rewards[i])

                tile = Image.new("RGB", (1000, 1000), (0,0,0))
                draw = ImageDraw.Draw(tile)
                font = ImageFont.truetype("Arial.ttf",200)  # load font
                position = (10, 10)
                text_color = (255, 255, 255)  # Use RGB values for the desired color
                draw.text(position, text, font=font, fill=text_color)
                rs = max(1, tile.width/max_dim, tile.height/max_dim)
                tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        for i, coord in enumerate(tsne_states):
            x = tx[i]
            y = ty[i]
            abstract_state, next_state = values[i]
            state, action, reward = labels[i]

            color_code = 255 #0 to 255. 
            if self.params['t-sne_next_state'] == False:
                phidot, psidot = Shared.get_action_from_index(action, self.params['action_lowest'], self.params['action_highest'], self.params['action_bins'])
                text = f's:{round(state[0],1),round(state[1],1),round(state[2],1)}\na:{round(phidot, 2), round(psidot,2)}\n'
            else:
                text = f's:{round(next_state[0],1),round(next_state[1],1),round(next_state[2],1)}\n '
                folded_angle = abs(next_state[0])
                if folded_angle > math.pi/2.0: # x and y axis symmetry
                    folded_angle = math.pi - folded_angle
                color_code =  round(255 - 255 * (folded_angle / (math.pi/2.0)))

            tile = Image.new("RGB", (1000, 1000), (color_code,64,64))
            draw = ImageDraw.Draw(tile)
            font = ImageFont.truetype("Arial.ttf",130)  # load font
            position = (10, 10)
            text_color = (255, 255, 255)  # Use RGB values for the desired color
            draw.text(position, text, font=font, fill=text_color)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        plt.figure(figsize = (16,16))

        x_ticks = [0, 4000, 8000, 12000, 16000]  # Custom tick positions
        x_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']  # Custom tick labels for x-axis
        y_ticks = [0, 4000, 8000, 12000, 16000]  # Custom tick positions
        y_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']  # Custom tick labels for y-axis

        plt.xticks(x_ticks, x_labels)
        plt.yticks(y_ticks, y_labels)

        plt.title(f'Equivalent State Mapping on 2-D Euclidean Space After {episode_index+1}\'th episodes ')
        plt.imshow(full_image)
        plt.show(block=True)
        return

    def create_knn_model(self):
        kNNModel = FaissKNeighbors(self.params['K_for_KNN'])
        X = torch.stack([tuple[0] for tuple in self.abstract_state_holders.values()]).squeeze(1).cpu().numpy()
        kNNModel.fit(X, np.array(list(self.abstract_state_holders.keys()), dtype=object))
        return kNNModel

    def find_equivalences(self, state, action, reward, knn_model):
        target_key = (tuple(state), action, reward)
        target_tensor = self.abstract_state_holders[target_key][0]
        return knn_model.predict(target_tensor[0].cpu().numpy().flatten()[np.newaxis, :])

    def replay_abstract_model(self):
        if len(self.abstraction_memory) > self.abstraction_batch_size:
            minibatch = random.sample(self.abstraction_memory, self.abstraction_batch_size)
        else:
            minibatch = self.abstraction_memory
        self.abstraction_model.train(True)
        self.abstract_optimizer.zero_grad()
        states, actions, rewards, next_states = zip(*minibatch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        abstract_states_tensor = self.abstraction_model.state_encoder(states_tensor)

        possible_actions_onehot_tensor = torch.eye(self.params['number_of_actions'])[range(self.params['number_of_actions'])].to(DEVICE)
        action_embeddings_tensor = self.abstraction_model.action_encoder(abstract_states_tensor, possible_actions_onehot_tensor)

        transitioned_abstract_states_tensor = abstract_states_tensor + action_embeddings_tensor[torch.arange(action_embeddings_tensor.size(0)), actions]
        
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
        abstract_next_states_tensor = self.abstraction_model.state_encoder(next_states_tensor)
        for index, state in enumerate(states):
            self.abstract_state_holders[(tuple(state), actions[index], self.get_abstract_reward(rewards[index]))] = (abstract_next_states_tensor[index].unsqueeze(0).detach(), next_states[index])

        # Loss components
        abstract_rewards = self.get_abstract_rewards(rewards)
        reward_fixations = torch.stack([self.reward_fixation_in_abstraction[abstract_reward] for abstract_reward in abstract_rewards])
        trans_loss, equivalence_loss, reward_fixation_loss = self.loss_function(abstract_states_tensor,
                                                            transitioned_abstract_states_tensor, abstract_next_states_tensor,
                                                            action_embeddings_tensor, reward_fixations)
        loss = trans_loss + equivalence_loss + reward_fixation_loss
        loss.backward()
        self.abstract_optimizer.step()
        
    def replay_mem(self, batch_size):
        self.replay_abstract_model()
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        knn_model = self.create_knn_model()

        self.model.train()
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        states, actions, rewards, next_states = zip(*minibatch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            targets = self.get_targets(rewards, next_states)
        outputs = self.model.forward(states_tensor)
        outputs_selected = outputs[torch.arange(len(minibatch)), actions]

        equivalent_states = []
        equivalent_actions = []
        equivalent_targets = []
        for index, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            abstract_reward = self.get_abstract_reward(reward)
            equivalences = self.find_equivalences(state, action, abstract_reward, knn_model)
            for equivalence in equivalences:
                if self.params["reward_filter"] == True and equivalence[2] != abstract_reward:
                    continue
                equivalent_states.append(equivalence[0])
                equivalent_actions.append(equivalence[1])
                equivalent_targets.append(targets[index])
        equivalent_states_tensor = torch.tensor(equivalent_states, dtype=torch.float32).to(DEVICE)
        equivalent_outputs = self.model.forward(equivalent_states_tensor)

        equivalent_actions_tensor = torch.tensor(equivalent_actions, dtype=torch.int64).to(DEVICE)
        equivalent_actions_tensor = equivalent_actions_tensor.unsqueeze(-1)
        equivalent_outputs_selected = equivalent_outputs.gather(1, equivalent_actions_tensor)
        equivalent_outputs_selected = equivalent_outputs_selected.squeeze(dim=-1)
        equivalent_targets_tensor = torch.tensor(equivalent_targets).to(DEVICE)
        equivalence_loss = self.equivalence_weight * F.mse_loss(equivalent_outputs_selected, equivalent_targets_tensor)
        loss = F.mse_loss(outputs_selected, targets) + equivalence_loss
        loss.backward()
        self.optimizer.step()
 
        # epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)
        self.current_iteration = self.current_iteration + 1
        if self.current_iteration % self.target_model_update_iterations == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def select_action_index(self, state, apply_epsilon_random):
        if apply_epsilon_random == True and random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_bins ** 2) # phidot, psidot actions

        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            prediction = self.model(state_tensor)
            return np.argmax(prediction.detach().cpu().numpy()[0])

    def get_targets(self, rewards, next_states):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
            q_values_next_states = self.target_model.forward(next_states_tensor)
            max_values, _ = torch.max(q_values_next_states, dim=1)
            targets = rewards_tensor + self.gamma * max_values
        return targets

    def on_terminated(self):
        pass