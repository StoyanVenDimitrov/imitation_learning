import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

import ppo.core as core
from ppo.utils.logx import EpochLogger
from ppo.utils.mpi_pytorch import (mpi_avg_grads, setup_pytorch_for_mpi,
                                   sync_params)
from ppo.utils.mpi_tools import (mpi_avg, mpi_fork, mpi_statistics_scalar,
                                 num_procs, proc_id)

"""
        Proximal Policy Optimization (by clipping), 

        with early stopping based on approximate KL

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with a 
                ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
                module. The ``step`` method should accept a batch of observations 
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``v``        (batch,)          | Numpy array of value estimates
                                            | for the provided observations.
                ``logp_a``   (batch,)          | Numpy array of log probs for the
                                            | actions in ``a``.
                ===========  ================  ======================================

                The ``act`` method behaves the same as ``step`` but only returns ``a``.

                The ``pi`` module's forward call should accept a batch of 
                observations and optionally a batch of actions, and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       N/A               | Torch Distribution object, containing
                                            | a batch of distributions describing
                                            | the policy for the provided observations.
                ``logp_a``   (batch,)          | Optional (only returned if batch of
                                            | actions is given). Tensor containing 
                                            | the log probability, according to 
                                            | the policy, of the provided actions.
                                            | If actions not given, will contain
                                            | ``None``.
                ===========  ================  ======================================

                The ``v`` module's forward call should accept a batch of observations
                and return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``v``        (batch,)          | Tensor containing the value estimates
                                            | for the provided observations. (Critical: 
                                            | make sure to flatten this!)
                ===========  ================  ======================================


            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to PPO.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs of interaction (equivalent to
                number of policy updates) to perform.

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while 
                still profiting (improving the objective function)? The new policy 
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
                denoted by :math:`\epsilon`. 

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take 
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used 
                for early stopping. (Usually small, 0.01 or 0.05.)

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """


class Generator:
    def __init__(self, env, discriminator, seed=0, steps_per_epoch=4000, max_ep_len=500, gamma=0.99, lam=0.97, pi_lr=3e-4,
            vf_lr=1e-3, ) -> None:
        self.env = env
        self.discriminator = discriminator
        self.policy = core.MLPActorCritic(env.observation_space, env.action_space)
         # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        self.logger = EpochLogger(**dict())
        # self.logger.save_config(locals())

        # Random seed
        seed += 10000 * proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #! mpi_fork(4)
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape
        self.max_ep_len = max_ep_len
        # Create actor-critic module
        # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

        # Sync params across processes
        sync_params(self.policy)
        # Set up model saving
        self.logger.setup_pytorch_saver(self.policy)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.policy.pi, self.policy.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.buf = PPOBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)
         # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.policy.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.policy.v.parameters(), lr=vf_lr)

        # Keep for plotting 
        self.avg_ep_len = 0.0
        self.avg_ep_return = 0.0

    def predict(self, state):
        """return action give a state

        Args:
            state (observation): observation 
        """
        return self.policy.act(state)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data, clip_ratio=0.2,):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.policy.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.policy.v(obs) - ret)**2).mean()

    def update(self, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
            data = self.buf.get()

            pi_l_old, pi_info_old = self.compute_loss_pi(data)
            pi_l_old = pi_l_old.item()
            v_l_old = self.compute_loss_v(data).item()

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data)
                kl = mpi_avg(pi_info['kl'])
                if kl > 1.5 * target_kl:
                    self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    break
                loss_pi.backward()
                mpi_avg_grads(self.policy.pi)    # average grads across MPI processes
                self.pi_optimizer.step()

            self.logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(data)
                loss_v.backward()
                mpi_avg_grads(self.policy.v)    # average grads across MPI processes
                self.vf_optimizer.step()

            # Log changes from update
            kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
            self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                        KL=kl, Entropy=ent, ClipFrac=cf,
                        DeltaLossPi=(loss_pi.item() - pi_l_old),
                        DeltaLossV=(loss_v.item() - v_l_old))

    def ppo(self, epochs=1, data=None, avg_reward=False):
        """train with ppo

        Args:
            epochs (int, optional): training epoch. Defaults to 1.
            data (list[Dict], optional): complete samples of (s,a,v,logp,s',r,d). Defaults to None.
            avg_reward (bool, optional): If using disc, whether to use the average for the rewards. Defaults to False.
        """
        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        buf = 0
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            for sample in  data:
                if self.discriminator:
                    with torch.no_grad(): 
                        logits = self.discriminator.forward(
                            torch.cat(
                                (
                                    torch.as_tensor(sample['state'], dtype=torch.float32), 
                                    torch.unsqueeze(torch.as_tensor(sample['action'], dtype=torch.float32),0)
                                )
                            )
                        )
                        score = -torch.sigmoid(logits)
                        sample['reward'] = score
            if avg_reward:
                avg = torch.mean(torch.cat([i['reward'] for i in data]))
                for sample in data:
                    sample['reward'] = torch.unsqueeze(avg,0)
            iterator = data if data else range(self.local_steps_per_epoch) 
            for i,t in enumerate(iterator):
                # when training the expert:
                if not data:
                    a, v, logp = self.policy.step(torch.as_tensor(o, dtype=torch.float32)) # ! keep for gradients

                    next_o, r, d, _ = self.env.step(a)
                    buf += 1
                    if d:
                        if buf<self.max_ep_len:
                            r = - 1.0 
                            buf = 0
                        else:
                            r = 0.0
                    else:
                        r = 0.0
                # ! edited the original implementatiom to first train the discriminator
                # when training the generator policy:
                else:
                    o, a, v, logp, next_o, r, d = t['state'], t['action'], t['value'], t['logprop'], t['next_state'], t['reward'], t['done']
                # Get the discriminator scores and keep them as reward
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                self.logger.store(VVals=v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                if data:
                    epoch_ended = i==len(data)-1
                else:
                    epoch_ended = t==self.local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.policy.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0



            # # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs-1):
            #     self.logger.save_state({'env': self.env}, None)

            # Perform PPO update!
            self.update()
            
            # Log info about epoch
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('StopIter', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.avg_ep_len = self.logger.log_current_row['EpLen']
            self.avg_ep_return = self.logger.log_current_row['AverageEpRet']
            self.logger.dump_tabular()


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.995, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

