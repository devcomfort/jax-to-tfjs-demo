import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from pathlib import Path
import orbax.checkpoint as ocp
from typing import Any, Dict, Tuple

def init_cnn_params(rng: jax.random.PRNGKey) -> Dict[str, Any]:
    """CNN 모델 파라미터 초기화"""
    keys = jax.random.split(rng, 5)
    
    # 첫 번째 컨볼루션 레이어
    conv1 = {
        'w': jax.random.normal(keys[0], (3, 3, 1, 32)) * 0.1,
        'b': jnp.zeros(32)
    }
    
    # 두 번째 컨볼루션 레이어
    conv2 = {
        'w': jax.random.normal(keys[1], (3, 3, 32, 64)) * 0.1,
        'b': jnp.zeros(64)
    }
    
    # 첫 번째 완전 연결 레이어
    dense1 = {
        'w': jax.random.normal(keys[2], (7 * 7 * 64, 128)) * 0.1,
        'b': jnp.zeros(128)
    }
    
    # 출력 레이어
    dense2 = {
        'w': jax.random.normal(keys[3], (128, 10)) * 0.1,
        'b': jnp.zeros(10)
    }
    
    return {
        'conv1': conv1,
        'conv2': conv2,
        'dense1': dense1,
        'dense2': dense2
    }

def train_and_evaluate(num_epochs: int = 5, learning_rate: float = 0.001):
    """모델 학습 및 평가"""
    rng = jax.random.PRNGKey(0)
    params = init_cnn_params(rng)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # update_step 함수 생성
    update_step = create_update_step(optimizer)
    
    train_data, test_data = load_mnist()
    
    # 체크포인트 저장을 위한 디렉토리 생성
    checkpoint_dir = Path('checkpoints/jax_mnist').absolute()
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
    
    total_batches = len(train_data)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('=' * 50)
        
        # 학습
        train_loss = 0
        for batch_idx, batch in enumerate(train_data):
            params, opt_state, loss = update_step(params, opt_state, batch)
            train_loss += loss
            
            # 진행률 표시
            progress = (batch_idx + 1) / total_batches * 100
            print(f'\rTraining: [{progress:6.2f}%] Batch {batch_idx + 1}/{total_batches} - Loss: {loss:.4f}', end='')
        
        train_loss /= total_batches
        print(f'\nTraining Loss: {train_loss:.4f}')
        
        # 평가
        test_loss = 0
        for batch_idx, batch in enumerate(test_data):
            loss, _ = loss_fn(params, batch)
            test_loss += loss
            
            # 진행률 표시
            progress = (batch_idx + 1) / len(test_data) * 100
            print(f'\rEvaluating: [{progress:6.2f}%] Batch {batch_idx + 1}/{len(test_data)} - Loss: {loss:.4f}', end='')
        
        test_loss /= len(test_data)
        print(f'\nTest Loss: {test_loss:.4f}')
        
        # 체크포인트 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f'jax_mnist_cnn_epoch{epoch+1}_{timestamp}'
        checkpoint_manager.save(
            epoch + 1,
            {'params': params, 'opt_state': opt_state}
        )
        print(f'Checkpoint saved: {checkpoint_name}')
    
    return params 

def cnn_forward(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """CNN 순전파"""
    # 입력 이미지 reshape (B, H, W) -> (B, H, W, C)
    if len(x.shape) == 3:
        x = x.reshape(*x.shape, 1)
    
    # 첫 번째 컨볼루션 레이어
    x = jax.lax.conv_general_dilated(
        x,
        params['conv1']['w'],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + params['conv1']['b'][None, None, None, :]
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
    
    # 두 번째 컨볼루션 레이어
    x = jax.lax.conv_general_dilated(
        x,
        params['conv2']['w'],
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = x + params['conv2']['b'][None, None, None, :]
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
    
    # 완전 연결 레이어
    x = x.reshape((x.shape[0], -1))
    x = jnp.dot(x, params['dense1']['w']) + params['dense1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['dense2']['w']) + params['dense2']['b']
    
    return x

@jax.jit
def loss_fn(params: Dict[str, Any], batch: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """손실 함수"""
    logits = cnn_forward(params, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

def create_update_step(optimizer: optax.GradientTransformation):
    """optimizer를 클로저로 처리하는 update_step 생성"""
    @jax.jit
    def update_step(
        params: Dict[str, Any],
        opt_state: optax.OptState,
        batch: Dict[str, jnp.ndarray]
    ) -> Tuple[Dict[str, Any], optax.OptState, jnp.ndarray]:
        """학습 스텝"""
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return update_step

def load_mnist():
    """MNIST 데이터셋 로드"""
    ds_train = tfds.load('mnist', split='train', as_supervised=True)
    ds_test = tfds.load('mnist', split='test', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    ds_train = ds_train.map(preprocess).batch(32)
    ds_test = ds_test.map(preprocess).batch(32)
    
    # TensorFlow 데이터셋을 JAX 배열로 변환
    def convert_to_jax(image, label):
        return {
            'image': jnp.array(image.numpy()),
            'label': jnp.array(label.numpy())
        }
    
    # 데이터셋을 메모리로 로드하고 JAX 배열로 변환
    train_data = []
    for images, labels in ds_train:
        train_data.append(convert_to_jax(images, labels))
    
    test_data = []
    for images, labels in ds_test:
        test_data.append(convert_to_jax(images, labels))
    
    return train_data, test_data 