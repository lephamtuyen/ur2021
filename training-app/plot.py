from pathlib import Path
import os
import argparse
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

''' Get parameters from command line '''
parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type=str, default=os.path.join(Path(__file__).resolve().parent, 'logs'))
parser.add_argument("--trial_id", type=str, default="test")
params = vars(parser.parse_known_args()[0])

log_path = os.path.join(params['log_path'], params['trial_id'], 'logs')
csv_file = pd.read_csv(log_path + '/log.csv')


plot_reward = sns.relplot(x="Epoch", y="Avg Reward", kind="line", data=csv_file)
plot_reward.savefig(log_path + "/Reward.png", bbox_inches='tight')

plot_loss = sns.relplot(x="Epoch", y="Loss", kind="line", data=csv_file)
plot_loss.savefig(log_path + "/Loss.png", bbox_inches='tight')