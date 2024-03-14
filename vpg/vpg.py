from core_eqx import *
import jax.numpy as jnp
import numpy as np
import optax
from time import time


class VPGBuffer():
    def __init__(self,obs_dim,act_dim,size,gamma=0.99,lam=0.95) -> None:
        self.obs_buf = np.zeros((size,obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size,1), dtype=np.float32)
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
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return data


def vpg(env_fn, actor_critic=MlpActorCritic, ac_kwargs={'hidden_sizes':(4,2)},  seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10):
    
    @eqx.filter_jit
    def compute_loss_pi(model,obs,act,adv):
        pi, log_p = jax.vmap(model.pi)(obs,act)
        loss_pi = -log_p.reshape(-1)*adv
        # print(loss_pi[:5])
        # input()
        return loss_pi.mean()
    
    @eqx.filter_jit
    def compute_loss_v(model,obs,ret):
       
        v_vals = jax.vmap(model.v)(obs)
        return ((v_vals-ret)**2).mean()


    @eqx.filter_jit
    def update(ac,state,data):
        obs, act, adv,ret, logp_old = data['obs'], data['act'], data['adv'],data['ret'],data['logp'] 
        policy_loss, policy_grad = eqx.filter_value_and_grad(compute_loss_pi)(ac,obs,act,adv)
        updates, state = optim.update(policy_grad, state, ac)
        ac = eqx.apply_updates(ac, updates)

        for i in range(train_v_iters):
            value_loss, value_grad = eqx.filter_value_and_grad(compute_loss_v)(ac,obs,ret)
            updates, state = optim.update(value_grad, state, ac)
            ac = eqx.apply_updates(ac, updates)


        return ac,state

    key = jax.random.PRNGKey(seed)
    env = env_fn()
    obs_dim = 4#jnp.array(env.observation_space.shape)
    act_dim = 2#jnp.array(env.action_space.shape)

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
    
    buff = VPGBuffer(obs_dim,act_dim,steps_per_epoch)


    optim = optax.adamw(1e-2)
    state = optim.init(eqx.filter(ac, eqx.is_array))

    # optim_value = optax.adamw(1e-3)
    # opt_state_value = optim_value.init(eqx.filter(ac.v, eqx.is_array))

    def eval(env,agent,key):
        total_ret = 0
        obs = env.reset()
        while True:
            act = ac.act(obs)
            
            next_obs, rew, done, info  = env.step(act)
            env.render()
            obs = next_obs
            total_ret += rew
            if done:
                break
        print('EVAL', total_ret)
        return total_ret
    
    @eqx.filter_jit
    def forward(model,obs,key):
        act,log_p, v = model(obs,key)
        return act,log_p,v
    @jax.jit
    def split(key):
        key = jax.random.split(key)[0]
        return key

    for epoch in range(epochs):
        start = time() 

        print('Epoch',epoch)
      
        obs = env.reset()
        rew_ep =  0
        
        for step in range(steps_per_epoch):
  
            act,log_p, v = forward(ac,obs,key)
            act = int(act)
            next_obs, rew, done, info  = env.step(act)
            rew_ep += rew
            buff.store(obs,act,rew,v,log_p)
            
            obs = next_obs
            key = split(key)
            epoch_ended = step == steps_per_epoch - 1

            if done or epoch_ended:
                
                if not done:
                    act,log_p, v = ac(obs,key)
                    buff.finish_path(v)
                else:
                    buff.finish_path(0)

                obs = env.reset() 
                rew_ep = 0
            

        data = buff.get()

        ac, state= update(ac,state,data)
        print(time()-start)
        eval(env,ac,key)


import gym

env = lambda : gym.make('CartPole-v1')

vpg(env)