"""
Flax 모델 평가 유틸리티

Flax 모델의 성능을 평가하는 기능을 제공합니다.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
from tqdm import tqdm
import logging
import pickle
import msgpack
from flax.training import checkpoints
from typing import Callable, Any, Dict, Optional, Union, cast, Tuple
from jax_to_tfjs.evaluation.metrics import calculate_metrics
from jax_to_tfjs.evaluation.types.metrics import ModelMetrics
from jax import random
from numpy.typing import NDArray

# 실제 CNN 모델 클래스 임포트
from jax_to_tfjs.models.flax.cnn_model import CNN

# 경고 비활성화
logging.getLogger("jax").setLevel(logging.ERROR)


def load_flax_checkpoint(checkpoint_path):
    """
    Flax 체크포인트를 로드합니다.

    인자:
        checkpoint_path: 체크포인트 파일 또는 디렉토리 경로

    반환:
        로드된 체크포인트 상태
    """
    logging.info(f"FLAX 모델 체크포인트 로드 중: {checkpoint_path}")

    try:
        # 디렉토리인 경우 Orbax 스타일의 체크포인트로 가정
        if os.path.isdir(checkpoint_path):
            try:
                # 먼저 표준 방식의 체크포인트 로드 시도
                state = checkpoints.restore_checkpoint(checkpoint_path, None)
                logging.info(f"표준 Orbax 체크포인트 로드 성공: {checkpoint_path}")
                return state
            except Exception as e:
                logging.warning(f"표준 방식 로드 실패: {e}, 다른 방법 시도 중...")
                # Orbax 체크포인트가 'model' 서브 디렉토리에 있는 경우
                model_dir = os.path.join(checkpoint_path, "model")
                if os.path.exists(model_dir):
                    state = checkpoints.restore_checkpoint(model_dir, None)
                    logging.info(
                        f"모델 디렉토리로부터 체크포인트를 직접 로드했습니다: {model_dir}"
                    )
                    return state
                else:
                    # msgpack 파일 검색
                    msgpack_files = [
                        f for f in os.listdir(checkpoint_path) if f.endswith(".msgpack")
                    ]
                    if msgpack_files:
                        with open(
                            os.path.join(checkpoint_path, msgpack_files[0]), "rb"
                        ) as f:
                            state = msgpack.unpack(f)
                            logging.info(
                                f"msgpack 파일에서 체크포인트 로드 성공: {msgpack_files[0]}"
                            )
                            return state

                    # 실패한 경우 디렉토리 내용 로깅
                    logging.error(f"디렉토리 내용: {os.listdir(checkpoint_path)}")
                    raise ValueError(
                        f"인식할 수 있는 체크포인트 형식을 찾을 수 없음: {checkpoint_path}"
                    )

        # 직접 파일 로드 시도
        if os.path.isfile(checkpoint_path):
            # 확장자 기반 로드
            if checkpoint_path.endswith(".msgpack"):
                with open(checkpoint_path, "rb") as f:
                    state = msgpack.unpack(f)
                    return state
            elif checkpoint_path.endswith(".pkl"):
                with open(checkpoint_path, "rb") as f:
                    state = pickle.load(f)
                    return state
            else:
                # 기본 restore_checkpoint 시도
                state = checkpoints.restore_checkpoint(
                    os.path.dirname(checkpoint_path),
                    None,
                    step=int(os.path.basename(checkpoint_path)),
                )
                return state

        # 번호가 지정된 경우 (예: '42')
        if checkpoint_path.isdigit():
            step = int(checkpoint_path)
            logging.info(f"체크포인트 스텝 {step}을 로드합니다.")
            # 현재 디렉토리 아래 checkpoints 폴더 검색
            possible_dirs = ["./checkpoints", "./flax_checkpoints", "./ckpt"]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    state = checkpoints.restore_checkpoint(dir_path, None, step=step)
                    if state is not None:
                        return state

        # 모든 시도 실패
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {checkpoint_path}")

    except Exception as e:
        logging.error(f"체크포인트 로드 중 오류 발생: {e}")
        raise


def extract_params_and_apply_fn(model_manager: Any) -> Tuple[Any, Callable]:
    """모델 매니저로부터 파라미터와 apply_fn을 추출합니다."""
    params = None
    apply_fn = None

    # 모델 매니저가 딕셔너리인 경우
    if isinstance(model_manager, dict):
        # Any 타입으로 처리하여 타입 체커 오류 방지
        manager_dict = cast(Dict[str, Any], model_manager)

        if "params" in manager_dict:
            params = manager_dict["params"]
        if "apply_fn" in manager_dict:
            apply_fn = manager_dict["apply_fn"]
        elif "model" in manager_dict and hasattr(manager_dict["model"], "apply"):
            apply_fn = manager_dict["model"].apply

    # TrainState 또는 유사한 객체인 경우
    elif hasattr(model_manager, "params"):
        params = model_manager.params
        if hasattr(model_manager, "apply_fn"):
            apply_fn = model_manager.apply_fn
        elif hasattr(model_manager, "model") and hasattr(model_manager.model, "apply"):
            apply_fn = model_manager.model.apply

    # Flax 모듈 자체인 경우
    elif hasattr(model_manager, "apply"):
        apply_fn = model_manager.apply
        # 파라미터가 없으면 기본 CNN으로 초기화
        if params is None:
            key = random.PRNGKey(0)
            dummy_input = jnp.ones((1, 28, 28, 1))  # 예시 입력 형태
            variables = model_manager.init(key, dummy_input)
            params = variables["params"]

    if params is None or apply_fn is None:
        raise ValueError("모델 매니저에서 파라미터나 apply_fn을 추출할 수 없습니다.")

    return params, apply_fn


def evaluate_flax_model(
    model_manager: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    save_dir: Optional[str] = None,
    with_probs: bool = False,
) -> Union[
    ModelMetrics,
    Tuple[ModelMetrics, np.ndarray, np.ndarray],
]:
    """
    Flax 모델 평가 함수. test_images와 test_labels에 대해 모델 정확도 및 성능 지표를 평가합니다.

    인자:
        model_manager: 모델 매니저 객체, 딕셔너리, 또는 체크포인트 경로
        test_images: 테스트 이미지 배열 (shape: [N, H, W, C])
        test_labels: 테스트 레이블 배열 (shape: [N, num_classes] 또는 [N,] - 원-핫 인코딩 또는 클래스 인덱스)
        save_dir: 결과를 저장할 디렉토리 경로
        with_probs: 예측 확률값도 함께 반환할지 여부

    반환:
        ModelMetrics 객체 또는 (metrics, predictions, probabilities) 튜플 (with_probs=True인 경우)
    """
    # 모델 매니저에서 파라미터와 apply_fn 추출
    try:
        params, apply_fn = extract_params_and_apply_fn(model_manager)
    except ValueError as e:
        logging.error(f"모델 평가 실패: {e}")
        # 오류 발생 시 빈 메트릭 객체 반환
        error_metrics = ModelMetrics(error=str(e))
        if with_probs:
            return (
                error_metrics,
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )
        return error_metrics

    # 테스트 이미지 형태 처리 및 예측 수행
    # 문자열이면 체크포인트 경로로 가정하고 로드
    if isinstance(model_manager, str):
        try:
            checkpoint_state = load_flax_checkpoint(model_manager)
            model_manager = checkpoint_state
            logging.info(
                f"체크포인트를 성공적으로 로드했습니다. 구조: {type(model_manager)}"
            )
            if isinstance(model_manager, dict):
                # 딕셔너리 경우 키 로깅
                logging.info(f"체크포인트 키: {list(model_manager.keys())}")
        except Exception as e:
            logging.error(f"체크포인트 로드 실패: {e}")
            raise

    # 프로젝트의 실제 CNN 모델 클래스 사용
    cnn_model = CNN()

    # JAX 배열로 변환
    test_images_jax = jnp.array(test_images)
    test_labels_jax = jnp.array(test_labels)

    # 배치 크기
    batch_size = 100
    num_samples = test_images_jax.shape[0]
    num_batches = num_samples // batch_size

    correct = 0
    all_predictions = []
    all_logits = []

    # JIT 컴파일을 통한 추론 성능 향상
    @jax.jit
    def predict_batch(images):
        # Flax 모델의 경우 {"params": params} 형태로 파라미터를 전달해야 함
        logits = apply_fn({"params": params}, images)
        predictions = jnp.argmax(logits, axis=1)
        return logits, predictions

    for i in tqdm(range(num_batches), desc="Flax 모델 평가"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_images = test_images_jax[start_idx:end_idx]
        batch_labels = test_labels_jax[start_idx:end_idx]

        try:
            # 예측
            logits, predictions = predict_batch(batch_images)

            # 정확도 계산
            correct += jnp.sum(predictions == batch_labels)
            all_predictions.append(predictions)
            all_logits.append(logits)
        except Exception as e:
            logging.error(f"Flax 모델 예측 중 오류 발생: {e}")
            logging.error(f"오류 타입: {type(e)}")
            raise

    accuracy = correct / num_samples
    predictions = jnp.concatenate(all_predictions)
    logits = jnp.concatenate(all_logits)

    # 확률 및 메트릭 계산
    probs = jax.nn.softmax(logits, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)

    # 이미 numpy 배열이 아니라면 변환
    test_labels_np = (
        np.array(test_labels)
        if not isinstance(test_labels, np.ndarray)
        else test_labels
    )
    predictions_np = (
        np.array(predictions)
        if not isinstance(predictions, np.ndarray)
        else predictions
    )
    probs_np = np.array(probs) if not isinstance(probs, np.ndarray) else probs

    # 메트릭 계산
    metrics = calculate_metrics(test_labels_np, predictions_np, probs_np)

    # 평가 결과 반환
    if with_probs:
        return metrics, predictions_np, probs_np

    return metrics
