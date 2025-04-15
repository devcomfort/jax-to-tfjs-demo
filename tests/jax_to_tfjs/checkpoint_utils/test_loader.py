"""
checkpoint_utils.loader 모듈에 대한 테스트

이 테스트 모듈은 체크포인트 로딩 유틸리티의 기능을 검증합니다.

모킹(Mocking) 정보:
- @patch 데코레이터를 통해 실제 체크포인트 파일을 로드하지 않고도 로딩 함수의 동작을 검증합니다.
- 모킹은 Orbax API와 같은 외부 의존성을 격리하여 테스트의 안정성과 일관성을 보장합니다.
- mock_checkpoint_dir 픽스처는 가상의 체크포인트 환경을 제공하여 파일 시스템 의존성을 제거합니다.
- mock_exists, mock_glob 등을 사용하여 파일 시스템 상태를 시뮬레이션하고, 예외 상황을 테스트합니다.
- 모킹을 통해 실제 체크포인트 데이터 없이도 로더의 오류 처리 기능을 검증할 수 있습니다.
"""
import os
import json
import pytest
import unittest.mock as mock
import tempfile
from pathlib import Path
import orbax.checkpoint as ocp

from jax_to_tfjs.checkpoint_utils.loader import (
    load_checkpoint_by_step,
    load_checkpoint_by_path,
    load_jax_checkpoint,
    load_flax_checkpoint,
    load_latest_checkpoint,
    get_orbax_checkpointer,
    create_checkpoint_manager
)

# 테스트 데이터 목업
MOCK_CHECKPOINT_INFO = {
    "path": "/checkpoints/jax_mnist/checkpoint_1",
    "name": "checkpoint_1",
    "model_type": "jax",
    "step": 1000,
    "timestamp": 1000000000,
    "datetime": "2000-01-01 00:00:00"
}

@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """임시 체크포인트 디렉토리 생성 및 테스트 데이터 설정"""
    # 임시 체크포인트 디렉토리 생성
    checkpoint_dir = tmp_path / "checkpoints" / "jax_mnist" / "checkpoint_1"
    checkpoint_dir.mkdir(parents=True)
    
    # 체크포인트 메타데이터 파일 생성
    with open(checkpoint_dir / "checkpoint", "w") as f:
        json.dump({
            "step": 1000,
            "timestamp": 1000000000,
            "model_type": "jax"
        }, f)
    
    return str(checkpoint_dir)

@mock.patch('jax_to_tfjs.checkpoint_utils.loader.get_checkpoint_by_step')
@mock.patch('jax_to_tfjs.checkpoint_utils.loader.load_checkpoint_by_path')
def test_load_checkpoint_by_step(mock_load_path, mock_get_step):
    """load_checkpoint_by_step 함수 테스트"""
    # 목업 설정
    mock_get_step.return_value = {"path": "/mock/path", "step": 2000}
    mock_load_path.return_value = {"weights": "mock_data"}
    
    # 함수 호출
    result = load_checkpoint_by_step("jax", 2000)
    
    # 검증
    mock_get_step.assert_called_once_with("jax", 2000, None)
    mock_load_path.assert_called_once_with("/mock/path", "jax")
    assert result == {"weights": "mock_data"}
    
    # 오류 시나리오
    mock_get_step.return_value = None
    with pytest.raises(ValueError):
        load_checkpoint_by_step("jax", 2000)

@mock.patch('jax_to_tfjs.checkpoint_utils.loader.get_orbax_checkpointer')
@mock.patch('pathlib.Path.exists')
def test_load_jax_checkpoint(mock_exists, mock_get_checkpointer, mock_checkpoint_dir):
    """load_jax_checkpoint 함수 테스트"""
    # 목업 설정
    mock_checkpointer = mock.MagicMock()
    mock_checkpointer.restore.return_value = {"weights": "mock_weights"}
    mock_get_checkpointer.return_value = mock_checkpointer
    mock_exists.return_value = True
    
    # 함수 호출
    checkpoint_path = os.path.join(mock_checkpoint_dir, "jax_checkpoint")
    result = load_jax_checkpoint(checkpoint_path)
    
    # 검증
    mock_get_checkpointer.assert_called_once()
    mock_checkpointer.restore.assert_called_once()
    assert result == {"weights": "mock_weights"}
    
    # 오류 시나리오
    mock_exists.return_value = False
    with pytest.raises(ValueError):
        load_jax_checkpoint("/invalid/path")

@mock.patch('pathlib.Path.glob')
@mock.patch('builtins.open', new_callable=mock.mock_open, read_data=b'msgpack_data')
@mock.patch('flax.serialization.from_bytes')
def test_load_flax_checkpoint(mock_from_bytes, mock_file, mock_glob):
    """load_flax_checkpoint 함수 테스트"""
    # 목업 설정
    mock_glob.return_value = ["checkpoint.msgpack"]
    mock_from_bytes.return_value = {"weights": "mock_weights"}
    
    # 함수 호출
    result = load_flax_checkpoint("/path/to/checkpoint")
    
    # 결과 검증
    assert result == {"weights": "mock_weights"}
    mock_from_bytes.assert_called_once()
    
    # 예외 케이스
    mock_glob.return_value = []
    with pytest.raises(ValueError):
        load_flax_checkpoint("/invalid/path")

@mock.patch('jax_to_tfjs.checkpoint_utils.loader.validation')
@mock.patch('jax_to_tfjs.checkpoint_utils.loader.load_jax_checkpoint')
@mock.patch('jax_to_tfjs.checkpoint_utils.loader.load_flax_checkpoint')
def test_load_checkpoint_by_path(mock_load_flax, mock_load_jax, mock_validation):
    """load_checkpoint_by_path 함수 테스트"""
    # JAX 모델 시나리오
    mock_validation.get_checkpoint_type.return_value = "jax"
    mock_load_jax.return_value = {"weights": "jax_weights"}
    
    result = load_checkpoint_by_path("/path/to/checkpoint")
    mock_load_jax.assert_called_once_with("/path/to/checkpoint")
    assert result == {"weights": "jax_weights"}
    
    # Flax 모델 시나리오
    mock_validation.get_checkpoint_type.return_value = "flax"
    mock_load_flax.return_value = {"weights": "flax_weights"}
    
    result = load_checkpoint_by_path("/path/to/checkpoint", "flax")
    mock_load_flax.assert_called_once_with("/path/to/checkpoint")
    assert result == {"weights": "flax_weights"}
    
    # 오류 시나리오
    mock_validation.get_checkpoint_type.return_value = "unknown"
    with pytest.raises(ValueError):
        load_checkpoint_by_path("/path/to/checkpoint")

@mock.patch('jax_to_tfjs.checkpoint_utils.loader.get_latest_checkpoint')
@mock.patch('jax_to_tfjs.checkpoint_utils.loader.load_checkpoint_by_path')
def test_load_latest_checkpoint(mock_load_path, mock_get_latest):
    """load_latest_checkpoint 함수 테스트"""
    # 목업 설정
    mock_get_latest.return_value = {"path": "/mock/latest", "step": 5000}
    mock_load_path.return_value = {"weights": "latest_weights"}
    
    # 함수 호출
    result = load_latest_checkpoint("jax")
    
    # 검증
    mock_get_latest.assert_called_once_with("jax", None)
    mock_load_path.assert_called_once_with("/mock/latest", "jax")
    assert result == {"weights": "latest_weights"}
    
    # 오류 시나리오
    mock_get_latest.return_value = None
    with pytest.raises(ValueError):
        load_latest_checkpoint("jax")

def test_get_orbax_checkpointer():
    """get_orbax_checkpointer 함수 테스트"""
    # 함수 호출
    result = get_orbax_checkpointer()
    
    # 결과 검증
    assert isinstance(result, ocp.PyTreeCheckpointer)

def test_create_checkpoint_manager():
    """create_checkpoint_manager 함수 테스트"""
    # 함수 호출
    manager, checkpointer = create_checkpoint_manager("/tmp/checkpoints")
    
    # 결과 검증
    assert isinstance(manager, ocp.CheckpointManager)
    assert isinstance(checkpointer, ocp.PyTreeCheckpointer) 