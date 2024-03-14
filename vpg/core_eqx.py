import jax
from jax import numpy as jnp
from jax import random as jrnd
from jax.random import normal as normal
import distrax
import equinox as eqx

import scipy
from itertools import chain
from functools import partial

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(eqx.Module):
    def __init__(self) -> None:
        pass
    def forward(self,obs:jnp.array, act:jnp.array=None):
        """
        Returns a distribution + loglikelihood of an action
        """
        pi = self._distribution(obs)
        log_pi = None
        if act is not None:
            log_pi = self._log_prob_from_distribution(pi,act)
        return pi, log_pi

class MLPCategoricalActor(Actor):
    layers: list

    def __init__(self,sizes:tuple,activation_fun:callable,seed:int) -> None:
        super().__init__()
        if type(seed) != int:
            raise "seed should be int"
        self.layers = []
        # self.layers += [(eqx.nn.Linear(in_features=sizes[i],
        #                                out_features=sizes[i+1],
        #                                key=jax.random.PRNGKey(i+seed)), 
        #                                activation_fun) for i in range(len(sizes)-1)]
        
        self.layers += [eqx.nn.Linear(in_features=sizes[0],
                                       out_features=64,
                                       key=jax.random.PRNGKey(1)), 
                                       activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=64,
                                       key=jax.random.PRNGKey(2)), 
                                        activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=sizes[-1],
                                       key=jax.random.PRNGKey(2)), 
                                       ]
 
        
        # self.layers = list(chain.from_iterable(self.layers))
    
    def _distribution(self,obs:jnp.array):
        for layer in self.layers:
            obs = layer(obs)
        dist = distrax.Categorical(logits=obs)
        return dist
    
    def _log_prob_from_distribution(self,pi,act):
        return pi.log_prob(act) 
    
    def __call__(self,obs,act=None):
        return self.forward(obs,act)
    
class MLPGaussianActor(Actor):
    layers: list

    def __init__(self,sizes:tuple,activation_fun:callable,seed:int) -> None:
        super().__init__()
        if type(seed) != int:
            raise "seed should be int"
        self.layers = []
        # self.layers += [(eqx.nn.Linear(in_features=sizes[i],
        #                                out_features=sizes[i+1],
        #                                key=jax.random.PRNGKey(i+seed)), 
        #                                activation_fun) for i in range(len(sizes)-1)]
        
        self.layers += [eqx.nn.Linear(in_features=sizes[0],
                                       out_features=64,
                                       key=jax.random.PRNGKey(1)), 
                                       activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=64,
                                       key=jax.random.PRNGKey(2)), 
                                        activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=2,
                                       key=jax.random.PRNGKey(2)), 
                                       ]
 
        
        # self.layers = list(chain.from_iterable(self.layers))
    
    def _distribution(self,obs:jnp.array):
        for layer in self.layers:
            # print(obs.shape)
            obs = layer(obs)

        dist = distrax.Normal(loc=obs[0],scale=obs[1])
        return dist
    
    def _log_prob_from_distribution(self,pi,act):
        return pi.log_prob(act) 
    
    def __call__(self,obs,act=None):
        return self.forward(obs,act)

class MLPCritic(eqx.Module):
    layers:list
    def __init__(self,sizes:tuple,activation_fun:callable,seed:int) -> None:
        super().__init__()
        if type(seed) != int:
            raise "seed should be int"
        self.layers = []
        # self.layers += [(eqx.nn.Linear(in_features=sizes[i],
        #                                out_features=sizes[i+1],
        #                                key=jax.random.PRNGKey(i+seed)), 
        #                                activation_fun) for i in range(len(sizes)-1)]
        self.layers += [eqx.nn.Linear(in_features=sizes[0],
                                       out_features=64,
                                       key=jax.random.PRNGKey(1)), 
                                       activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=64,
                                       key=jax.random.PRNGKey(2)), 
                                        activation_fun,
                        eqx.nn.Linear(in_features=64,
                                       out_features=sizes[-1],
                                       key=jax.random.PRNGKey(2)), 
                                       ]
        # self.layers = list(chain.from_iterable(self.layers))

    def __call__(self,obs):
        for layer in self.layers:
            obs = layer(obs)
      
        return obs

class MlpActorCritic(eqx.Module):
    pi: MLPGaussianActor
    v: MLPCritic
    pi_shape: list
    v_shape: list

    def __init__(self,observation_space: int, action_space:int, hidden_sizes:tuple, activation_fun:callable=jax.nn.tanh,
                 seed:int=42,): 
        
        self.pi_shape = [observation_space,] + list(hidden_sizes) + [action_space,]
        self.v_shape = [observation_space,] + list(hidden_sizes) + [1,]
        
        self.pi = MLPGaussianActor(sizes=self.pi_shape,activation_fun=activation_fun,seed=seed)
        self.v = MLPCritic(sizes=self.v_shape,activation_fun=activation_fun,seed=seed)
    
    # @eqx.filter_jit
    def __call__(self,obs,key):
        
        dist = self.pi._distribution(obs)
        act = dist.sample(seed=key)
        log_p = dist.log_prob(act)
        v = self.v(obs)
       
        return act, log_p, v
    
   
    def act(self,obs,key=42):
        # raise NotImplemented

        dist = self.pi._distribution(obs)
        act = dist.sample(seed=key)
        # for layer in self.pi.layers:
            # obs = layer(obs)
        return act

