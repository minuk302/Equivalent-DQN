import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_new_result_index(path):
	if not os.path.exists(path):
		os.makedirs(path)
	index = 0
	file_name = f"{index}.csv"
	file_path = os.path.join(path, file_name)
	while os.path.exists(file_path):
		index += 1
		file_name = f"{index}.csv"
		file_path = os.path.join(path, file_name)
	return index

def read_rewards(file_path):
	with open(file_path, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			return row

action = 9
num_episodes = 200
print_all = True
no_equivalence_path = f"result/{action}_no_equivalence"
number_of_results = 1#get_new_result_index(no_equivalence_path)

episodes = np.arange(1, num_episodes+1)
def add_to_plot(ax, path, label, color, number_of_results):
	equivalence_rewards = []
	for i in range(number_of_results):
		file_name = f"{i}.csv"
		equivalence_rewards.append(read_rewards(os.path.join(path, file_name)))
	equivalence_rewards = np.array([inner_array[:num_episodes] for inner_array in equivalence_rewards], dtype=float)
	equivalence_mean = np.mean(equivalence_rewards, axis=0)
	equivalence_std_deviation = np.std(equivalence_rewards, axis=0)
	ax.plot(episodes, equivalence_mean, label=label, color=color)
	ax.fill_between(episodes, equivalence_mean - equivalence_std_deviation, equivalence_mean + equivalence_std_deviation, alpha=0.3, color=color)

if print_all:
	fig, ax = plt.subplots()
	add_to_plot(ax, no_equivalence_path, "Naive DQN", 'b', number_of_results)
	index = 0
	colors = ["r", "g", "y", "c", "m", "k", "w", "0.5", "orange", "purple", "lime", "maroon"]
	for k in [3, 7, 11]:
		for reward_filter, weight in [(False, 0.6),(True,1.0)]:
			add_to_plot(ax, f'result/{action}_equivalent({k})-{weight},filter({reward_filter})', f"k:{k},equivalence weight:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	plt.xlabel("episodes")
	plt.ylabel("rewards")
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
else:
	for k in [3,7,11]:
		for reward_filter, weight in [(False, 0.6), (True,1.0)]:
			fig, ax = plt.subplots()
			equivalence_path = f'result/{action}_equivalent({k})-{weight},filter({reward_filter})'
			number_of_results = min(get_new_result_index(equivalence_path), get_new_result_index(no_equivalence_path))
			add_to_plot(ax, f'result/{action}_equivalent({k})-{weight},filter({reward_filter})', f"k:{k},weight:{weight},filter:{reward_filter}", 'r', number_of_results)
			add_to_plot(ax, no_equivalence_path, "no equivalent", 'b', number_of_results)
			plt.xlabel("episodes")
			plt.ylabel("rewards")
			plt.title(f'Rewards averaged from {number_of_results} runs')
			ax.legend()
			plt.show(block=True)