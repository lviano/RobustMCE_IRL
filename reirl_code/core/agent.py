import multiprocessing
from utils.replay_memory import Memory, TwoPlayerMemory
from utils.torch import *
import math
import time
import copy


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, state_only = False,
                    opponent_policy = None, alpha = None, reward_type=None):
    torch.randn(pid)
    log = dict()
    if opponent_policy is None:
        memory = Memory()
    else:
        memory = TwoPlayerMemory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:
        state = env.reset()
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0

        for t in range(100000): #range(10000):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    action = policy(state_var)[0][0].numpy()
                else:
                    if opponent_policy is not None:
                        opponent_plays = np.random.choice(2, p=[alpha, 1 - alpha])
                        opponent_action = opponent_policy.select_action(state_var)[0].numpy()
                        player_action = policy.select_action(state_var)[0].numpy()
                        if opponent_plays:
                            action = copy.deepcopy(opponent_action)
                        else:
                            action = copy.deepcopy(player_action)

                        player_action = int(player_action) if policy.is_disc_action else player_action.astype(
                            np.float64)
                        opponent_action = int(opponent_action) if policy.is_disc_action else opponent_action.astype(
                            np.float64)
                        """if np.isnan(player_action).any():
                            print("Player Nan")
                            player_action = np.zeros_like(player_action)
                        if np.isnan(opponent_action).any():
                            print("Opponent Nan")
                            opponent_action = np.zeros_like(opponent_action)
                        action = (1 - alpha)*opponent_action.clip(-1.0, 1.0) + alpha*player_action.clip(-1.0, 1.0)"""
                    else:
                        action = policy.select_action(state_var)[0].numpy()

            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            if not policy.is_disc_action:
                action_to_play = action.clip(-1.0, 1.0)
                next_state, reward, done, _ = env.step(action_to_play)
            else:
                next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:

                if state_only:
                    reward = custom_reward(state, next_state, reward_type)
                else:
                    reward = custom_reward(state, action, reward_type)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1
            if opponent_policy is not None:
                memory.push(state, player_action, opponent_action, action, mask, next_state, reward)
            else:
                memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                if opponent_policy is not None:
                    memory.push(next_state, player_action, opponent_action, action,
                                mask, next_state, reward)
                else:
                    memory.push(next_state, action, mask, next_state, reward)
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1, state_only=False,
                 opponent_net = None, alpha= None, reward_type=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads
        self.state_only = state_only
        self.opponent_net = opponent_net
        self.alpha = alpha
        self.reward_type = reward_type

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, self.state_only)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, self.state_only,
                                      self.opponent_net, self.alpha, self.reward_type)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        if self.opponent_net is None:
            log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        else:
            log['action_mean'] = np.mean(np.vstack(batch.player_action), axis=0)
            log['action_min'] = np.min(np.vstack(batch.player_action), axis=0)
            log['action_max'] = np.max(np.vstack(batch.player_action), axis=0)
            log['opponent_action_mean'] = np.mean(np.vstack(batch.opponent_action), axis=0)
            log['opponent_action_min'] = np.min(np.vstack(batch.opponent_action), axis=0)
            log['opponent_action_max'] = np.max(np.vstack(batch.opponent_action), axis=0)
        return batch, log
