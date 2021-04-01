import time
import argparse
import itertools
import glob
import shutil
from copy import deepcopy
from pathlib import Path
from robot_env import AllocateEnv
from PPO import PPO, Buffer
from utils import *
from logger import *
import torch
from tqdm import tqdm


def main():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    # Algorihtm params
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_ratio", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_iters", type=int, default=1)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--n_episode_per_epoch", type=int, default=1)
    parser.add_argument("--n_training_epoch", type=int, default=100)
    parser.add_argument("--optimizer_func", type=str, default="adam")
    parser.add_argument("--value_loss_func", type=str, default="smoothl1")

    # Model params
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--recurrent_size", type=int, default=256)
    parser.add_argument("--sequence", type=int, default=2)
    parser.add_argument("--fc_activation", type=str, default='selu')
    parser.add_argument("--shared_net", type=int, default=0)

    # use cuda or not
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--cuda_idx", type=int, default=1)

    # Name of trial
    parser.add_argument("--trial_id", type=str, default="trial_id")

    # Pre_trained model path
    parser.add_argument("--pre_trained_model", type=str, default=None)

    params = vars(parser.parse_known_args()[0])

    seed = params['seed']
    clip_ratio = params['clip_ratio']
    discount_factor = params['discount_factor']
    gae_lambda = params['gae_lambda']
    learning_rate = params['learning_rate']
    train_iters = params['train_iters']
    hidden_size = params['hidden_size']
    clip_norm = params['clip_norm']
    entropy_ratio = params['entropy_ratio']
    shared_net = params['shared_net']
    device = torch.device(f"cuda:{params['cuda_idx']}" if torch.cuda.is_available() and params['use_cuda'] else "cpu")
    recurrent_size = params['recurrent_size']
    sequence = params['sequence']

    log_path = os.path.join(Path(__file__).resolve().parent, f"logs/{params['trial_id']}")

    fc_activation = get_activation_fn(params['fc_activation'])
    value_loss_func = get_loss_func(params['value_loss_func'])
    optimizer_func = get_optimizer(params['optimizer_func'])

    save_interval = params['save_interval']
    n_episode_per_epoch = params['n_episode_per_epoch']
    n_training_epoch = params['n_training_epoch']

    # Copy file to log for version backup
    if not os.path.exists(os.path.join(log_path, f"logs")):
        os.makedirs(os.path.join(log_path, f"logs"))
    for file in glob.glob(os.path.join(Path(__file__).resolve().parent, "*.py")):
        shutil.copy(file, os.path.join(log_path, "logs/"))

    robot_env = AllocateEnv(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "./../unity-app/simulator.x86_64")),
        no_graphics=True, time_scale=1.0, episode_length=40, seed=seed)
    max_buff_length = 40

    # Using PPO algorithm
    buff_length = max_buff_length * n_episode_per_epoch
    agent = PPO(fc_feature_size=robot_env.state_dim,
                act_dim=robot_env.action_dim,
                hidden_size=hidden_size,
                buff_length=buff_length,
                learning_rate=learning_rate,
                train_iters=train_iters,
                clipping_ratio=clip_ratio,
                fc_activation=fc_activation,
                discount=discount_factor,
                gae_lambda=gae_lambda,
                clip_norm=clip_norm,
                entropy_ratio=entropy_ratio,
                shared_net=shared_net,
                optimizer_func=optimizer_func,
                value_loss_func=value_loss_func,
                recurrent_size=recurrent_size,
                sequence=sequence
                )

    robot_env.reset()
    robot_env.close()
    del robot_env

    # Init from pre-trained model
    pre_trained_model = params['pre_trained_model']
    if pre_trained_model is not None and os.path.exists(pre_trained_model):
        print(f"Pre-trained model {pre_trained_model} is loaded")
        cp_model = torch.load(pre_trained_model)
        agent.load_state_dict(cp_model[f"model"])
        if cp_model.get(f"optimizer") is not None:
            agent.optimizer.load_state_dict(cp_model[f"optimizer"])

    # Log to file and console
    log_filename = f"log.csv"
    configure(folder=log_path + f'/logs', log_files=[log_filename])

    def data_collector(agent_, it, epoch, random_seed=123):
        # Random an env
        np.random.seed(random_seed + it + epoch * n_episode_per_epoch)

        env = AllocateEnv(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../unity-app/simulator.x86_64")),
            no_graphics=False, time_scale=1.0, episode_length=40, seed=123)

        obs = env.reset()

        buff = Buffer(fc_feature_size=env.state_dim,
                      act_dim=env.action_dim,
                      size=max_buff_length,
                      gamma=discount_factor,
                      gae_lambda=gae_lambda,
                      recurrent_size=recurrent_size)

        epi_r = 0
        done = False
        if recurrent_size is not None:
            memory = np.zeros([1, recurrent_size])
        else:
            memory = None

        while not done:
            act, val, logp, new_memory = agent_.step(obs, memory)

            next_obs, reward, done = env.step(act.item())

            buff.store(obs, act, reward, val, logp, done, memory)

            obs = next_obs
            memory = new_memory
            epi_r += reward

            if done:
                _, val, _, _ = agent_.step(obs, memory)
                buff.update_delayed_reward()
                buff.finish_path(last_val=val)

        print(f"Epi: {epoch}-{it} | "
              f"Total collected obj: {env.total_collected_obj}")

        env.close()

        rew = np.sum(buff.rew_buf[:buff.ptr])
        del env
        return buff, rew

    best_avg_reward = -999999.0
    start_epoch = 0
    log(f"Training started...")
    start_time = time.time()

    # Create an agent for collecting samples
    if torch.cuda.is_available() and params['use_cuda']:
        sample_agent = deepcopy(agent)

    for epoch in itertools.count(start=start_epoch):

        print(f"Collecting {n_episode_per_epoch} episodes at epoch {epoch}...")
        if torch.cuda.is_available() and params['use_cuda']:
            sample_agent.load_state_dict(agent.cpu().state_dict())
        else:
            sample_agent = agent

        # Running multiple CPUs for faster training
        results = []
        for it in tqdm(range(n_episode_per_epoch)):
            result = data_collector(sample_agent, it, epoch, seed)
            results.append(result)

        sum_r = 0
        for result in results:
            localBuffer = result[0]
            epi_r = result[1]
            agent.buf.store_from_other_buffer(localBuffer)
            sum_r += epi_r

            del localBuffer

        # Print average reward (average reward of current trained model)
        avg_reward = sum_r / n_episode_per_epoch
        print_log = f"Epoch: {epoch} | " \
                    f"Avg Reward: {avg_reward}"
        log(print_log)

        # Logging the log file
        logkv(key="Epoch", val=epoch, log_file=log_filename)
        logkv(key="Avg Reward", val=avg_reward, log_file=log_filename)

        # Saving models
        if epoch % save_interval == 0:
            if not os.path.exists(f"{log_path}/models"):
                os.makedirs(f"{log_path}/models")

            model_dict = dict()
            model_dict[f"model"] = agent.state_dict()
            model_dict[f"optimizer"] = agent.optimizer.state_dict()

            # Only save the best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(model_dict, f"{log_path}/models/model_{epoch}.pt")
                torch.save(model_dict, f"{log_path}/models/model_best.pt")

                time_to_get_best_model = time.time() - start_time
                log(f"Current best model: {epoch} | "
                    f"Time to get best model: {datetime.timedelta(seconds=time_to_get_best_model)}")

        # Perform PPO update!
        loss = agent.to(device).update(device)
        log(f"Optimizing agent at epoch {epoch} | " \
            f"Training Loss: {loss} | " \
            f"Training Time: {datetime.timedelta(seconds=(time.time() - start_time))}")

        # Logging the log file
        logkv(key="Loss", val=loss, log_file=log_filename)
        dumpkvs()

        if epoch - start_epoch + 1 >= n_training_epoch:
            break

    training_time = time.time() - start_time
    log(f"Training {n_training_epoch} ended... with training time: {datetime.timedelta(seconds=training_time)}")

    return 0


if __name__ == '__main__':
    main()
