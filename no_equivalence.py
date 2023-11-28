import Shared
from DQN import DQNAgent, DEVICE

params = Shared.parameters()
action = params['action_bins']
rewards = Shared.run(params, DQNAgent, f"result/{action}_no_equivalence")