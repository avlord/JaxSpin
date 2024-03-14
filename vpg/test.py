from core_eqx import *
import jax 
from jax.example_libraries import optimizers 
import equinox as eqx
import optax

x = jnp.ones(2)
ac = MlpActorCritic(2,4,(16,))

print(ac.pi_shape,ac.v_shape)

def loss(
    model: MlpActorCritic, x):
    loss_val = model.pi(x,1)[1].sum()
    return loss_val

print(loss(ac,x))

loss, grads = eqx.filter_value_and_grad(loss)(ac, x )

optim = optax.adamw(1e-3)
opt_state = optim.init(eqx.filter(ac, eqx.is_array))
updates, opt_state = optim.update(grads, opt_state, ac)
ac = eqx.apply_updates(ac, updates)

