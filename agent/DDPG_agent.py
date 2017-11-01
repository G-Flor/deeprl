#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from component import *
from utils import *
import pickle
import torch.nn as nn

class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.learning_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.learning_network.state_dict())
        self.target_network.eval()
        self.actor_opt = config.actor_optimizer_fn(self.learning_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.learning_network.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.epsilon = 1.0
        self.d_epsilon = 1.0 / config.noise_decay_interval

        self.state_normalizer = Normalizer(self.task.state_dim)
        self.reward_normalizer = Normalizer(1)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.target_network_mix) +
                                    param.data * self.config.target_network_mix)

    def episode(self, deterministic=False):
        self.random_process.reset_states()
        state = self.task.reset()
        state = self.state_normalizer(state)

        config = self.config
        actor = self.learning_network.actor
        critic = self.learning_network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic

        steps = 0
        total_reward = 0.0
        while True:
            actor.eval()
            action = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                if self.total_steps < config.exploration_steps:
                    action = self.task.random_action()
                else:
                    action += max(self.epsilon, config.min_epsilon) * self.random_process.sample()
                    self.epsilon -= self.d_epsilon
            next_state, reward, done, info = self.task.step(action)
            assert np.isfinite(reward)
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            next_state = self.state_normalizer(next_state)
            total_reward += reward
            reward = self.reward_normalizer(reward) # I turned this one - Mik
            assert np.isfinite(total_reward)

            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1
            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.total_steps > config.exploration_steps:
                self.learning_network.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                assert np.isfinite(rewards).all()
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                terminals = critic.to_torch_variable(terminals).unsqueeze(1)
                rewards = critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)

                # BUG Q blows up, it's wierd even thought when I calculate it
                # I get e.g. [0.1,0.2,0.3], when I look at stored values it's
                # [0.1,0.2,9e10] not sure why...
                # So let's clip it for now
                def clip(x, xmin, xmax):
                    x[x>xmax]=xmax
                    x[x<xmin]=xmin
                    return x
                qmax=1e5
                if np.abs(q.data.numpy()).max()>qmax:
                    config.logger.warning('q is above %s',qmax)
                    q = clip(q, -qmax, qmax)
                    q_next = clip(q_next, -qmax, qmax)
                if np.abs(q_next.data.numpy()).max()>qmax:
                    config.logger.warning('q_next is above %s',qmax)
                    q = clip(q, -qmax, qmax)
                    q_next = clip(q_next, -qmax, qmax)
                critic_loss = self.criterion(q, q_next)
                assert np.isfinite(critic_loss.data.numpy())
                critic.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                actions = actor.predict(states, False)
                var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, var_actions)
                critic.zero_grad() # is this something I need? Mike
                q.backward(torch.ones(q.size()))

                actor.zero_grad()
                actions.backward(-var_actions.grad.data)
                self.actor_opt.step()
                config.logger.debug('-var_actions.grad.data: %s', -var_actions.grad.data)
                config.logger.debug('q.size(): %s', q.size())
                config.logger.debug('critic_loss: %s', critic_loss)

                self.soft_update(self.target_network, self.learning_network)

                q = None
                q_next = None

        return total_reward, steps

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.learning_network.state_dict(), f)
