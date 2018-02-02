#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from utils.normalizer import Normalizer


class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.worker_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.worker_network.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.worker_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.worker_network.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0

        self.state_normalizer = Normalizer(self.task.state_dim)
        self.reward_normalizer = Normalizer(1)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def state_dict(self):
        return {
            'worker_network': self.worker_network.state_dict(),
            'replay': self.replay.state_dict(),
            'state_normalizer': self.state_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict()
        }

    def load_state_dict(self, saved):
        self.worker_network.load_state_dict(saved['worker_network'])
        self.replay.load_state_dict(saved['replay'])
        self.state_normalizer.load_state_dict(saved['state_normalizer'])
        self.reward_normalizer.load_state_dict(saved['reward_normalizer'])

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.state_dict(), f)

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.state_normalizer(state)

        config = self.config
        actor = self.worker_network.actor
        critic = self.worker_network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic

        steps = 0
        total_reward = 0.0
        while True:
            actor.eval()
            action = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                noise = self.random_process.sample()
            else:
                noise = 0
            action += noise
            next_state, reward, done, info = self.task.step(action)
            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            next_state = self.state_normalizer(next_state)
            total_reward += reward

            assert np.isfinite(action), 'action should be finite'

            # tensorboard logging
            suffix = 'test_' if deterministic else ''
            if action.squeeze().ndim == 0:
                config.logger.scalar_summary(suffix + 'action', action, self.total_steps)
                config.logger.scalar_summary(suffix + 'noise', noise, self.total_steps)
            else:
                config.logger.histo_summary(suffix + 'action', action, self.total_steps)
                config.logger.histo_summary(suffix + 'noise', noise, self.total_steps)
            config.logger.scalar_summary(suffix + 'reward', reward, self.total_steps)
            for key in info:
                config.logger.scalar_summary('info_' + key, info[key], self.total_steps)

            reward = self.reward_normalizer(reward) * config.reward_scaling

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                self.worker_network.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                terminals = critic.to_torch_variable(terminals).unsqueeze(1)
                rewards = critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)

                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                if config.gradient_clip:
                    critic_grad_norm = nn.utils.clip_grad_norm(self.worker_network.critic.parameters(), config.gradient_clip)
                self.critic_opt.step()

                actions = actor.predict(states, False)
                var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, var_actions)
                q.backward(torch.ones(q.size()))
                critic.zero_grad()

                actor.zero_grad()
                self.actor_opt.zero_grad()
                actions.backward(-var_actions.grad.data)
                if config.gradient_clip:
                    actor_grad_norm = nn.utils.clip_grad_norm(self.worker_network.actor.parameters(), config.gradient_clip)
                self.actor_opt.step()
                actor.zero_grad()

                # tensorboard logging
                config.logger.scalar_summary('loss_policy', -var_actions.grad.data.sum(), self.total_steps)
                config.logger.scalar_summary('loss_critic', critic_loss, self.total_steps)
                config.logger.scalar_summary('lr_actor', torch.FloatTensor([self.actor_opt.param_groups[0]['lr']]), self.total_steps)
                config.logger.scalar_summary('lr_critic', torch.FloatTensor([self.critic_opt.param_groups[0]['lr']]), self.total_steps)
                if config.gradient_clip:
                    config.logger.histo_summary('grad_norm_actor', actor_grad_norm, self.total_steps)
                    config.logger.histo_summary('grad_norm_critic', critic_grad_norm, self.total_steps)

                self.soft_update(self.target_network, self.worker_network)

        config.logger.writer.file_writer.flush()

        return total_reward, steps
