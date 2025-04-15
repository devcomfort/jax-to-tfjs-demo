import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from pathlib import Path
import orbax.checkpoint as ocp
from typing import Any, Dict, Tuple
import argparse
import sys
import logging
import os
import numpy as np
from tqdm import tqdm

from jax_to_tfjs.paths import get_jax_checkpoint_path

# JAX 경고 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow 경고 비활성화
logging.getLogger('jax').setLevel(logging.ERROR)  # JAX 경고를 ERROR 레벨 이상으로만 표시

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

def train_and_evaluate(num_epochs: int = 5, learning_rate: float = 0.001, evaluate_model: bool = True, subdir: str = None):
    """모델 학습 및 평가"""
    rng = jax.random.PRNGKey(0)
    params = init_cnn_params(rng)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # update_step 함수 생성
    update_step = create_update_step(optimizer)
    
    train_data, test_data = load_mnist()
    
    # 체크포인트 경로 설정
    checkpoint_dir = get_jax_checkpoint_path(subdir)
    
    # Orbax 체크포인트 매니저 생성
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1000
    )
    checkpoint_manager = ocp.CheckpointManager(
        directory=str(checkpoint_dir),
        checkpointers={"model": checkpointer},
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
        
        # 체크포인트 저장
        checkpoint_manager.save(
            epoch + 1,
            {"model": {'params': params, 'opt_state': opt_state}}
        )
        print(f'Checkpoint saved: {os.path.join(checkpoint_dir, checkpoint_name)}')
    
    # 학습 완료 후 전체 평가 수행
    if evaluate_model:
        print("\n" + "=" * 50)
        print("모델 최종 평가 중...")
        evaluate_mnist(params)
    
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

def load_checkpoint(checkpoint_dir: str = None, step: int = None) -> Dict[str, Any]:
    """체크포인트 로드
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로 (None이면 기본 경로 사용)
        step: 로드할 체크포인트 스텝 (None이면 가장 최근 체크포인트)
    
    Returns:
        로드된 파라미터와 옵티마이저 상태
    """
    # 체크포인트 경로 설정
    if checkpoint_dir is None:
        checkpoint_dir = get_jax_checkpoint_path()
    
    print(f"JAX 모델 체크포인트 로드 중: {checkpoint_dir}")
    
    # Orbax 체크포인트 구조 확인
    model_dir = os.path.join(checkpoint_dir, "model")
    if os.path.isdir(model_dir):
        # model 디렉토리가 있는 경우 해당 디렉토리를 체크포인트 경로로 사용
        checkpoint_dir = model_dir
    
    checkpointer = ocp.PyTreeCheckpointer()
    
    try:
        # Orbax API 사용
        options = ocp.CheckpointManagerOptions(max_to_keep=3)
        checkpoint_manager = ocp.CheckpointManager(
            directory=str(checkpoint_dir),
            checkpointers={"model": checkpointer},
            options=options
        )
        
        if step is None:
            step = checkpoint_manager.latest_step()
            if step is None:
                # 모델 디렉토리 자체가 체크포인트인 경우
                step = 0
        
        if step == 0:
            # 모델 디렉토리 자체가 체크포인트인 경우 직접 로드
            try:
                checkpoint = checkpointer.restore(checkpoint_dir)
                print(f"모델 디렉토리로부터 체크포인트를 직접 로드했습니다: {checkpoint_dir}")
                return checkpoint
            except Exception as e:
                print(f"체크포인트 직접 로드 실패: {e}")
                raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        else:
            # 체크포인트 복원
            checkpoint = checkpoint_manager.restore(
                step, 
                items={"model": None}
            )["model"]
            print(f"Loaded checkpoint from step {step} in {checkpoint_dir}")
            return checkpoint
            
    except Exception as e:
        print(f"체크포인트 로드 오류: {e}")
        
        # 대체 방법 시도: 직접 체크포인트 로드
        try:
            checkpoint = checkpointer.restore(checkpoint_dir)
            print(f"대체 방법으로 체크포인트를 로드했습니다: {checkpoint_dir}")
            return checkpoint
        except Exception as e2:
            print(f"대체 체크포인트 로드 실패: {e2}")
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

def load_mnist_test():
    """MNIST 테스트 데이터셋 로드"""
    ds_test = tfds.load('mnist', split='test', as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    ds_test = ds_test.map(preprocess).batch(100)
    
    # TensorFlow 데이터셋을 NumPy 배열로 변환
    test_images = []
    test_labels = []
    
    for images, labels in ds_test:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())
    
    return np.concatenate(test_images), np.concatenate(test_labels)

def evaluate_mnist(params):
    """MNIST 테스트 세트에 대한 평가"""
    # 테스트 데이터셋 로드
    test_images, test_labels = load_mnist_test()
    print(f"테스트 데이터셋 로드 완료: {test_images.shape[0]} 샘플")
    
    # JAX 배열로 변환
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)
    
    # 배치 크기
    batch_size = 100
    num_samples = test_images_jax.shape[0]
    num_batches = num_samples // batch_size
    
    correct = 0
    all_predictions = []
    
    for i in tqdm(range(num_batches), desc="모델 평가"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_images = test_images_jax[start_idx:end_idx]
        batch_labels = test_labels_jax[start_idx:end_idx]
        
        # 예측
        logits = cnn_forward(params, batch_images)
        predictions = jnp.argmax(logits, axis=1)
        
        # 정확도 계산
        correct += jnp.sum(predictions == batch_labels)
        all_predictions.append(predictions)
    
    accuracy = float(correct) / num_samples
    predictions = jnp.concatenate(all_predictions)
    
    # 결과 출력
    print(f"테스트 정확도: {accuracy:.4f} ({int(correct)}/{num_samples})")
    
    # 클래스별 정확도 계산
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for i in range(num_samples):
        label = int(test_labels[i])
        class_total[label] += 1
        if predictions[i] == test_labels[i]:
            class_correct[label] += 1
    
    print("\n클래스별 정확도:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f"클래스 {i}: {accuracy:.4f} ({int(class_correct[i])}/{int(class_total[i])})")
    
    # 혼동 행렬 계산
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)
    for i in range(num_samples):
        confusion_matrix[int(test_labels[i]), int(predictions[i])] += 1
    
    print("\n혼동 행렬:")
    print(confusion_matrix)
    
    return accuracy, predictions

if __name__ == '__main__':
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='JAX MNIST 모델 학습')
    parser.add_argument('--epochs', type=int, default=5, help='훈련 에포크 수 (기본값: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률 (기본값: 0.001)')
    parser.add_argument('--evaluate', action='store_true', help='학습 후 모델 평가 수행')
    parser.add_argument('--subdir', type=str, default=None, help='체크포인트를 저장할 하위 디렉토리 (선택사항)')
    args = parser.parse_args()
    
    print(f"JAX MNIST 모델 학습 시작 (에포크: {args.epochs}, 학습률: {args.lr})...")
    params = train_and_evaluate(
        num_epochs=args.epochs, 
        learning_rate=args.lr, 
        evaluate_model=args.evaluate,
        subdir=args.subdir
    )
    print(f"JAX MNIST 모델 학습 완료!") 