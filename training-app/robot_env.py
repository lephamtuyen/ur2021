#  Copyright (C) AgileSoDA. All rights reserved.
#  Developer: tuyenple

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from utils import *

AGENT_TYPE_RANDOM = 0
AGENT_TYPE_FSFP = 1
AGENT_TYPE_SP = 2
AGENT_TYPE_RL = 3


class AllocateEnv:

    def __init__(self, exe_file, no_graphics=False, time_scale=1.0, episode_length=100, seed=123):

        channel = EngineConfigurationChannel()

        okay = False
        while not okay:
            try:
                base_port = np.random.randint(40000, 50000)
                self.unity_env = UnityEnvironment(
                    file_name=exe_file,
                    no_graphics=no_graphics,
                    base_port=base_port,
                    timeout_wait=9999,
                    seed=seed,
                    side_channels=[channel])
                okay = True
            except Exception:
                okay = False
        channel.set_configuration_parameters(time_scale=time_scale)
        self._episode_length = episode_length
        self.n_sussessors = 5
        self.n_other_states = 9
        self.n_features = 4
        self.n_robots = 3

        self.initialize_space()

    def initialize_space(self):
        self.action_dim = self.n_sussessors
        # self.state_dim = 7 + self.n_sussessors * 5 * 2 + self._episode_length
        self.state_dim = 6 + self.n_robots + self.n_sussessors * self.n_features + self.n_sussessors * 5 * 2 + self._episode_length

    def reset(self):
        self.unity_env.reset()
        for bn in self.unity_env.behavior_specs:
            self.behavior_name = bn
            break

        self.episode_step = 0
        self.total_collected_obj = 0

        initial_state, _, _ = self.get_next_state()

        return initial_state

    def get_next_state(self):
        # get first state
        # Check for end episode
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)

        if decision_steps.obs[0].shape[0] > 0:
            self.last_obs = decision_steps.obs[0][0]
        else:
            self.last_obs = terminal_steps.obs[0][0]

        obs = []
        obs.append(np.eye(self._episode_length)[self.episode_step])

        obs.append([self.last_obs[0] / 100.0])
        obs.append([self.last_obs[1] / 100.0])
        obs.append([self.last_obs[2] / 100.0])
        obs.append([self.last_obs[3] / 100.0])
        obs.append([self.last_obs[4] / 100.0])
        obs.append([self.last_obs[5] / 100.0])

        obs.append(np.eye(self.n_robots)[int(self.last_obs[8])])

        rank_displacement = [(i, np.sqrt((self.last_obs[6] - self.last_obs[
            self.n_other_states + i * self.n_features + 1]) ** 2 +
                                         (self.last_obs[7] - self.last_obs[
                                             self.n_other_states + i * self.n_features + 2]) ** 2))
                             for i in range(self.n_sussessors)]
        rank_maxreach = [(i, self.last_obs[self.n_other_states + i * self.n_features + 3]) for i in
                         range(self.n_sussessors)]
        rank_displacement.sort(key=lambda pair: pair[1])
        rank_maxreach.sort(key=lambda pair: pair[1])

        one_hot_vec = np.eye(self.n_sussessors)

        rank_displacement_indices = np.zeros(self.n_sussessors, dtype=int)
        rank_maxreach_indices = np.zeros(self.n_sussessors, dtype=int)

        for i in range(self.n_sussessors):
            rank_displacement_indices[rank_displacement[i][0]] = i
            rank_maxreach_indices[rank_maxreach[i][0]] = i

        for i in range(self.n_sussessors):
            obs.append([self.last_obs[self.n_other_states + i * self.n_features] / 100.0])
            obs.append([self.last_obs[self.n_other_states + i * self.n_features + 1]])
            obs.append([self.last_obs[self.n_other_states + i * self.n_features + 2]])
            obs.append([self.last_obs[self.n_other_states + i * self.n_features + 3]])

            obs.append(one_hot_vec[rank_displacement_indices[i]])
            obs.append(one_hot_vec[rank_maxreach_indices[i]])

        obs = np.concatenate(obs)

        # done or not
        done = False
        for _ in terminal_steps:
            done = True
            break

        if done:
            step_reward = terminal_steps.reward[0]
        else:
            step_reward = decision_steps.reward[0]

        reward = step_reward

        return obs, done, reward

    def step(self, action, agent_type=AGENT_TYPE_RL):
        # Set action
        continuous_actions = []
        discrete_actions = []

        # data = {'state': self.build_reach_state(action)}
        # response = requests.post('http://localhost:50000/inference', json=data)
        # continuous_actions.append(response.json()['action'])

        # x_ = self.last_obs[int(self.n_other_states + action * self.n_features + 1)]
        # z_ = self.last_obs[int(self.n_other_states + action * self.n_features + 2)]
        # max_reach = np.sqrt(0.4 ** 2 - (z_ + 0.25) ** 2)

        continuous_actions.append([0.15])
        target_idx = int(self.last_obs[self.n_other_states + self.n_features * action])

        discrete_actions.append([agent_type, target_idx])
        self.unity_env.set_actions(self.behavior_name,
                                   ActionTuple(continuous=np.array(continuous_actions),
                                               discrete=np.array(discrete_actions)))
        self.unity_env.step()
        self.episode_step += 1

        # next obs
        next_state, done, reward = self.get_next_state()

        return next_state, reward, done

    def close(self):
        self.unity_env.close()


if __name__ == '__main__':

    for epi in range(2):
        done = False

        states, actions, rewards, dones = [], [], [], []

        env = AllocateEnv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "./../unity-app/simulator.x86_64")),
            no_graphics=False, time_scale=1.0, episode_length=40, seed=123)

        state = env.reset()
        step = 0
        while not done:
            action = 0
            target_idx = int(env.last_obs[env.n_other_states + env.n_features * action])
            print(f"Robot: {int(env.last_obs[8])} "
                  f"| Target_id: {target_idx} "
                  f"| Action: {action}")

            next_state, reward, done = env.step(action, agent_type=AGENT_TYPE_FSFP)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            step += 1

        # update delay reward
        ptr = len(states)

        for i in range(len(states)):
            next_idx = i
            founded = False
            while next_idx < len(states):
                if int(states[i][6]) == 0:
                    if next_idx == 0 or next_idx == 1 or rewards[next_idx] > 5000:
                        next_idx += 1
                    else:
                        rewards[i] = get_true_reward(rewards[next_idx])
                        founded = True
                        break
                elif int(states[i][6]) == 1:
                    if next_idx == 1 or rewards[next_idx] < 5000 or rewards[next_idx] > 15000:
                        next_idx += 1
                    else:
                        rewards[i] = get_true_reward(rewards[next_idx])
                        founded = True
                        break
                else:
                    if rewards[next_idx] > 15000:
                        rewards[i] = get_true_reward(rewards[next_idx])
                        founded = True
                        break
                    else:
                        next_idx += 1

            if not founded:
                ptr = i
                break

            # print(f"Epi: {epi + 1} | "
            #       f"Step: {i} | "
            #       f"Robot: {states[i][6]} | "
            #       f"State: {states[i]} | "
            #       f"Reward: {rewards[i]} | "
            #       f"Done: {dones[i]}")

        print(f"Epi: {epi + 1} | "
              f"reward: {np.sum(rewards[:ptr])}")

        env.close()
        del env
