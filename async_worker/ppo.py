#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch.multiprocessing as mp
from network import *
from utils import *
from component import *
from async_worker import *
import pickle
import os
import time

class ProximalPolicyOptimization:
    def __init__(self, config, shared_network, extra):
        self.config = config
        self.task = config.task_fn()
        self.policy = config.policy_fn()

        self.shared_network = shared_network
        self.actor_opt = config.actor_optimizer_fn(shared_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(shared_network.critic.parameters())

        self.worker_network = config.network_fn()
        self.worker_network.load_state_dict(shared_network.state_dict())

        self.shared_state_normalizer = extra[0]
        self.state_normalizer = StaticNormalizer(self.task.state_dim)
        self.shared_reward_normalizer = extra[1]
        self.reward_normalizer = StaticNormalizer(1)

    def episode(self, deterministic=False):
        config = self.config
        self.state_normalizer.offline_stats.load(self.shared_state_normalizer.offline_stats)
        self.reward_normalizer.offline_stats.load(self.shared_reward_normalizer.offline_stats)

        replay = config.replay_fn()
        state = self.task.reset()
        state = self.state_normalizer(state)

        episode_length = 0
        batched_rewards = 0
        batched_steps = 0
        batched_episode = 0

        actor_net = self.worker_network.actor
        critic_net = self.worker_network.critic

        actor_net_old = config.actor_network_fn()
        actor_net_old.load_state_dict(actor_net.state_dict())

        self.worker_network.load_state_dict(self.shared_network.state_dict())
        while not replay.full():
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            for i in range(config.rollout_length):
                mean, std, log_std = actor_net.predict(np.stack([state]))
                if not np.isfinite(mean.data.numpy()).all():
                    print('NaN', state, actor_net.predict(np.stack([state])))
                value = critic_net.predict(np.stack([state]))
                assert np.isfinite(mean.data.numpy().flatten()).all()
                assert np.isfinite(std.data.numpy().flatten()).all()
                action = self.policy.sample(mean.data.numpy().flatten(), std.data.numpy().flatten(), deterministic)
                assert np.isfinite(action).all()
                assert np.isfinite(value.data.numpy()).all()
                action = self.config.action_shift_fn(action)
                states.append(state)
                actions.append(action)
                values.append(value)
                state, reward, done, _ = self.task.step(action)
                state = self.state_normalizer(state)
                done = (done or (config.max_episode_length and episode_length > config.max_episode_length))

                batched_rewards += reward
                batched_steps += 1
                episode_length += 1

                reward = self.reward_normalizer(reward)
                assert np.isfinite(reward)
                rewards.append(reward)

                # These seem to avoid NaN's I was getting that I couldn't replicate
                # even when debugging at the same point, and in the foreground
                mean = None
                std = None
                log_std = None
                value = None
                action = None

                if done:
                    episode_length = 0
                    batched_episode += 1
                    state = self.task.reset()
                    state = self.state_normalizer(state)
                    break

            R = torch.zeros((1, 1))
            if not done:
                R = critic_net.predict(np.stack([state])).data
                assert np.isfinite(R.numpy()).all()

            values.append(Variable(R))
            A = Variable(torch.zeros((1, 1)))
            for i in reversed(range(len(rewards))):
                R = Variable(torch.FloatTensor([[rewards[i]]]))
                ret = R + self.config.discount * values[i + 1]
                A = ret - values[i] + self.config.discount * self.config.gae_tau * A
                advantages.append(A.detach())
                returns.append(ret.detach())
            advantages = list(reversed(advantages))
            returns = list(reversed(returns))
            assert np.isfinite([a.data.numpy() for a in advantages]).all()
            assert np.isfinite([a.data.numpy() for a in returns]).all()
            replay.feed([states, actions, returns, advantages])

        batched_rewards /= batched_episode
        batched_steps /= batched_episode

        if deterministic:
            return batched_steps, batched_rewards

        with config.steps_lock:
            config.total_steps.value += replay.memory_size

        self.shared_state_normalizer.offline_stats.merge(self.state_normalizer.online_stats)
        self.state_normalizer.online_stats.zero()

        self.shared_reward_normalizer.offline_stats.merge(self.reward_normalizer.online_stats)
        self.reward_normalizer.online_stats.zero()

        for _ in np.arange(self.config.optimize_epochs):
            self.worker_network.load_state_dict(self.shared_network.state_dict())

            states, actions, returns, advantages = replay.sample()
            states = actor_net.to_torch_variable(np.stack(states))
            actions = actor_net.to_torch_variable(np.stack(actions))
            returns = torch.cat(returns, 0)
            advantages_raw = torch.cat(advantages, 0).squeeze(1)
            advantages = (advantages_raw - advantages_raw.mean()) / advantages_raw.std()
            assert np.isfinite(advantages.data.numpy()).all()
            assert np.isfinite(returns.data.numpy()).all()
            config.logger.debug('sampled returns=%s advantages=%s advantages_raw=%s', returns[:10], advantages[:10], advantages_raw[:10])

            mean_old, std_old, log_std_old = actor_net_old.predict(states)
            assert np.isfinite(mean_old.data.numpy()).all()
            assert np.isfinite(std_old.data.numpy()).all()
            assert np.isfinite(log_std_old.data.numpy()).all()
            probs_old = actor_net.log_density(actions, mean_old, log_std_old, std_old)
            mean, std, log_std = actor_net.predict(states)
            probs = actor_net.log_density(actions, mean, log_std, std)

            # avoid NaNs with small std I going to clamp this - mike
            probs_old = probs_old.clamp(-10,20)
            probs = probs.clamp(-10,20)

            ratio = (probs - probs_old).exp()
            obj = ratio * advantages
            obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip, 1.0 + self.config.ppo_ratio_clip) * advantages
            policy_loss = -torch.min(obj, obj_clipped).mean(0)
            if config.entropy_weight:
                policy_loss += -config.entropy_weight * actor_net.entropy(std)

            v = critic_net.predict(states)
            value_loss = 0.5 * (returns - v).pow(2).mean()
            actor_net_old.load_state_dict(actor_net.state_dict())

            self.worker_network.zero_grad()
            assert np.isfinite(value_loss.data.numpy())
            assert np.isfinite(policy_loss.data.numpy())
            policy_loss.backward()
            value_loss.backward()
            config.logger.debug('policy_loss=%s value_loss=%s', policy_loss, value_loss)
            nn.utils.clip_grad_norm(self.worker_network.parameters(), config.gradient_clip)
            with config.network_lock:
                self.shared_network.zero_grad()
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                for param, worker_param in zip(self.shared_network.parameters(), self.worker_network.parameters()):
                    assert np.isfinite(worker_param.grad.data.numpy()).all()
                    param._grad = worker_param.grad.clone()
                self.actor_opt.step()
                self.critic_opt.step()

        return batched_steps, batched_rewards
