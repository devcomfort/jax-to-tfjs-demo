"""
checkpoint_utils.validation 모듈에 대한 테스트

이 테스트 모듈은 체크포인트 유효성 검증 유틸리티의 기능을 검증합니다.

모킹(Mocking) 정보:
- @patch 데코레이터를 사용하여 외부 시스템(파일 시스템, Orbax API 등)에 대한 의존성을 제거합니다.
- mock_open을 사용하여 파일 읽기 작업을 모킹하고 일관된 테스트 데이터를 제공합니다.
- MagicMock을 통해 복잡한 객체 동작을 시뮬레이션하여 다양한 조건에서의 함수 동작을 검증합니다.
- 모킹을 통해 테스트가 환경에 관계없이 항상 동일하게 작동하도록 보장합니다.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from jax_to_tfjs.checkpoint_utils.validation import (
    validate_checkpoint,
    is_checkpoint_directory,
    extract_step_from_checkpoint,
    get_checkpoint_type
)

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

def test_validate_checkpoint(mock_checkpoint_dir):
    """validate_checkpoint 함수 테스트"""
    # 유효한 체크포인트 경로
    assert validate_checkpoint(mock_checkpoint_dir, "jax") is True
    
    # 잘못된 모델 타입
    assert validate_checkpoint(mock_checkpoint_dir, "unknown") is False
    
    # 존재하지 않는 경로
    assert validate_checkpoint("/non/existent/path", "jax") is False

def test_is_checkpoint_directory(mock_checkpoint_dir):
    """is_checkpoint_directory 함수 테스트"""
    # 유효한 체크포인트 디렉토리
    assert is_checkpoint_directory(mock_checkpoint_dir) is True
    
    # 존재하지 않는 경로
    assert is_checkpoint_directory("/non/existent/path") is False
    
    # 체크포인트 디렉토리가 아닌 경우
    parent_dir = os.path.dirname(mock_checkpoint_dir)
    assert is_checkpoint_directory(parent_dir) is False

def test_extract_step_from_checkpoint_path():
    """extract_step_from_checkpoint 함수 테스트 - 경로에서 추출"""
    # 이름에서 스텝 추출
    assert extract_step_from_checkpoint("/path/to/checkpoint_1000") == 1000
    
    # 경로에 스텝이 없는 경우
    assert extract_step_from_checkpoint("/path/to/checkpoint") is None

@patch('os.path.exists')
@patch('builtins.open', new_callable=mock_open, read_data='{"step": 2000}')
def test_extract_step_from_checkpoint_metadata(mock_file, mock_exists):
    """extract_step_from_checkpoint 함수 테스트 - 메타데이터에서 추출"""
    # 목업 설정
    mock_exists.return_value = True
    
    # 함수 호출
    result = extract_step_from_checkpoint("/path/to/checkpoint")
    
    # 결과 검증
    assert result == 2000

@patch('os.path.exists')
@patch('os.path.isdir')
def test_get_checkpoint_type_by_path(mock_isdir, mock_exists):
    """get_checkpoint_type 함수 테스트 - 경로 기반"""
    # 목업 설정
    mock_exists.return_value = True
    mock_isdir.return_value = True
    
    # JAX 체크포인트 경로
    assert get_checkpoint_type("/checkpoints/jax_mnist/checkpoint_1") == "jax"
    
    # Flax 체크포인트 경로
    assert get_checkpoint_type("/checkpoints/flax_mnist/checkpoint_1") == "flax"

@patch('os.path.exists')
@patch('os.path.isdir')
@patch('builtins.open', new_callable=mock_open, read_data='{"model_type": "jax"}')
def test_get_checkpoint_type_by_metadata(mock_file, mock_isdir, mock_exists):
    """get_checkpoint_type 함수 테스트 - 메타데이터 기반"""
    # 목업 설정
    mock_exists.return_value = True
    mock_isdir.return_value = True
    
    # 함수 호출
    result = get_checkpoint_type("/path/to/checkpoint")
    
    # 결과 검증
    assert result == "jax"

@patch('os.path.exists')
@patch('os.path.isdir')
def test_get_checkpoint_type_nonexistent(mock_isdir, mock_exists):
    """get_checkpoint_type 함수 테스트 - 존재하지 않는 경로"""
    # 목업 설정
    mock_exists.return_value = False
    mock_isdir.return_value = False
    
    # 함수 호출
    result = get_checkpoint_type("/non/existent/path")
    
    # 결과 검증
    assert result is None 