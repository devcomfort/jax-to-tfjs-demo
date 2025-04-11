import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow_datasets as tfds
from datetime import datetime
from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import train_state
from typing import Any, Tuple

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, learning_rate):
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()

def load_mnist():
    ds_train = tfds.load('mnist', split='train', as_supervised=True)
    ds_test = tfds.load('mnist', split='test', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return {'image': image, 'label': label}
    
    ds_train = ds_train.map(preprocess).batch(32)
    ds_test = ds_test.map(preprocess).batch(32)
    
    return ds_train, ds_test

def train_and_evaluate(num_epochs=5, learning_rate=0.001):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate)
    
    ds_train, ds_test = load_mnist()
    
    # 체크포인트 저장을 위한 디렉토리 생성
    checkpoint_dir = Path('results/mnist_checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Orbax 체크포인트 매니저 생성
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1000
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        ocp.PyTreeCheckpointer(),
        options=options
    )
    
    for epoch in range(num_epochs):
        # 학습
        for batch in ds_train:
            state, loss = train_step(state, batch)
        
        # 평가
        test_loss = 0
        for batch in ds_test:
            test_loss += eval_step(state, batch)
        test_loss /= len(ds_test)
        
        print(f'Epoch {epoch + 1}, Test Loss: {test_loss:.4f}')
        
        # 체크포인트 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'mnist_cnn_epoch{epoch+1}_{timestamp}'
        checkpoint_manager.save(
            epoch + 1,
            args=ocp.args.StandardSave(state),
            save_kwargs={'save_args': ocp.args.StandardSave(state)}
        )
    
    return state

if __name__ == '__main__':
    train_and_evaluate() 