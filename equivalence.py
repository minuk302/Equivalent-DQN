import Shared
from DQN_Equivalent import DQNAgent, DEVICE
import threading
import copy

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['memory_size']
	params['batch_size_for_abstraction'] = 32
	params['abstraction_learning_rate'] = 0.001
	params['abstract_state_space_dimmension'] = 20
	params['K_for_KNN'] = 11
	params['equivalence_weight'] = 1.0
	params['abstract_state_holders_size'] = params['memory_size']
	params['reward_filter'] = True

	# to debug mapping
	params['plot_t-sne'] = False
	params['t-sne_next_state'] = True
	params['plot_reward_fixations'] = False
	return params

params = define_parameters()
action = params['action_bins']
if params['plot_t-sne'] == True:	
	Shared.run(params, DQNAgent, f'equivalent_result/t-sne')

threads = []
for k in [3,7,11]:
	for reward_filter, weight in [(False, 0.6),(True,1.0)]:
		params['K_for_KNN'] = k
		params['equivalence_weight'] = weight
		params['reward_filter'] = reward_filter
		thread = threading.Thread(target=Shared.run, args=(copy.deepcopy(params), DQNAgent, f'result/{action}_equivalent({k})-{weight},filter({reward_filter})'))
		thread.start()
		threads.append(thread)

for thread in threads:
	thread.join()