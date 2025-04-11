import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from pathlib import Path
import orbax.checkpoint as ocp
from flax.training import train_state
from typing import Any, Dict, Tuple

class CNN(nn.Module):
    """Flax를 사용한 CNN 모델"""
    
    @nn.compact
    def __call__(self, x):
        # 첫 번째 컨볼루션 레이어
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # 두 번째 컨볼루션 레이어
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # 완전 연결 레이어
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        
        return x

def create_train_state(rng: jax.random.PRNGKey, learning_rate: float) -> train_state.TrainState:
    """학습 상태 생성"""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """학습 스텝"""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """평가 스텝"""
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()

def load_mnist():
    """MNIST 데이터셋 로드"""
    ds_train = tfds.load('mnist', split='train', as_supervised=True)
    ds_test = tfds.load('mnist', split='test', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    ds_train = ds_train.map(preprocess).batch(32)
    ds_test = ds_test.map(preprocess).batch(32)
    
    return ds_train, ds_test

def load_checkpoint(checkpoint_dir: str = 'checkpoints/flax_mnist', step: int = None) -> train_state.TrainState:
    """체크포인트 로드
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로
        step: 로드할 체크포인트 스텝 (None이면 가장 최근 체크포인트)
    
    Returns:
        로드된 학습 상태
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=3)
    )
    
    if step is None:
        step = checkpoint_manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
    
    checkpoint = checkpoint_manager.restore(step)
    state = checkpoint['state']
    print(f"Loaded checkpoint from step {step}")
    return state

def train_and_evaluate(num_epochs: int = 5, learning_rate: float = 0.001):
    """모델 학습 및 평가"""
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate)
    
    ds_train, ds_test = load_mnist()
    
    # 체크포인트 저장을 위한 디렉토리 생성
    checkpoint_dir = Path('checkpoints/flax_mnist').absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Orbax 체크포인트 매니저 생성
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1000
    )
    checkpoint_manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        checkpointer,
        options=options
    )
    
    # 데이터셋을 메모리에 로드하고 JAX 배열로 변환
    def convert_to_jax(image, label):
        return {
            'image': jnp.array(image.numpy()),
            'label': jnp.array(label.numpy())
        }
    
    train_data = []
    for images, labels in ds_train:
        train_data.append(convert_to_jax(images, labels))
    
    test_data = []
    for images, labels in ds_test:
        test_data.append(convert_to_jax(images, labels))
    
    total_batches = len(train_data)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('=' * 50)
        
        # 학습
        train_loss = 0
        for batch_idx, batch in enumerate(train_data):
            state, loss = train_step(state, batch)
            train_loss += loss
            
            # 진행률 표시
            progress = (batch_idx + 1) / total_batches * 100
            print(f'\rTraining: [{progress:6.2f}%] Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}', end='')
        
        train_loss /= total_batches
        print(f'\nTraining Loss: {train_loss:.4f}')
        
        # 평가
        test_loss = 0
        for batch_idx, batch in enumerate(test_data):
            loss = eval_step(state, batch)
            test_loss += loss
            
            # 진행률 표시
            progress = (batch_idx + 1) / len(test_data) * 100
            print(f'\rEvaluating: [{progress:6.2f}%] Batch {batch_idx + 1}/{len(test_data)} - Loss: {loss:.4f}', end='')
        
        test_loss /= len(test_data)
        print(f'\nTest Loss: {test_loss:.4f}')
        
        # 체크포인트 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'flax_mnist_cnn_epoch{epoch+1}_{timestamp}'
        checkpoint_manager.save(
            epoch + 1,
            {'state': state}
        )
        print(f'Checkpoint saved: {checkpoint_name}')
    
    return state

if __name__ == '__main__':
    train_and_evaluate() 