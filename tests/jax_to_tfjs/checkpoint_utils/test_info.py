"""
checkpoint_utils.info 모듈에 대한 테스트

이 테스트 모듈은 체크포인트 정보 검색 유틸리티의 기능을 검증합니다.

모킹(Mocking) 정보:
- @patch 데코레이터를 사용하여 외부 의존성을 대체하고 테스트를 격리합니다.
- 모킹을 통해 파일 시스템 접근 없이 일관된 테스트 환경을 구성합니다.
- 각 테스트는 특정 조건과 예외 상황을 시뮬레이션하기 위해 모의 객체를 사용합니다.
- mock_checkpoint_dir 픽스처는 가상의 체크포인트 디렉토리 구조를 제공하여 파일 시스템 의존성을 제거합니다.
"""
import os
import json
import pytest
import tempfile
from pathlib import Path
import unittest.mock as mock
from datetime import datetime

from jax_to_tfjs.checkpoint_utils.info import (
    get_checkpoints_info,
    list_available_checkpoints,
    get_latest_checkpoint,
    get_checkpoint_by_index,
    get_checkpoint_by_step
)

# 테스트 데이터 목업
MOCK_CHECKPOINTS = [
    {
        "path": "/checkpoints/jax_mnist/checkpoint_1",
        "name": "checkpoint_1",
        "model_type": "jax",
        "step": 1000,
        "timestamp": 1000000000,
        "datetime": "2000-01-01 00:00:00"
    },
    {
        "path": "/checkpoints/jax_mnist/checkpoint_2",
        "name": "checkpoint_2",
        "model_type": "jax",
        "step": 2000,
        "timestamp": 2000000000,
        "datetime": "2000-01-01 00:00:01"
    }
]

@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """임시 체크포인트 디렉토리 생성 및 테스트 데이터 설정"""
    # 임시 체크포인트 디렉토리 생성
    checkpoint_dir = tmp_path / "checkpoints" / "jax_mnist"
    checkpoint_dir.mkdir(parents=True)
    
    # 체크포인트 1
    ckpt1_dir = checkpoint_dir / "checkpoint_1"
    ckpt1_dir.mkdir()
    with open(ckpt1_dir / "checkpoint", "w") as f:
        json.dump({"step": 1000, "timestamp": 1000000000}, f)
    
    # 체크포인트 2
    ckpt2_dir = checkpoint_dir / "checkpoint_2"
    ckpt2_dir.mkdir()
    with open(ckpt2_dir / "checkpoint", "w") as f:
        json.dump({"step": 2000, "timestamp": 2000000000}, f)
    
    return str(checkpoint_dir)

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_jax_checkpoint_path')
@mock.patch('jax_to_tfjs.checkpoint_utils.info.validate_checkpoint')
@mock.patch('jax_to_tfjs.checkpoint_utils.info.extract_step_from_checkpoint')
def test_get_checkpoints_info(mock_extract_step, mock_validate, mock_get_path, mock_checkpoint_dir):
    """get_checkpoints_info 함수 테스트"""
    # 목업 설정
    mock_get_path.return_value = mock_checkpoint_dir
    mock_validate.return_value = True
    mock_extract_step.side_effect = lambda path: 1000 if "checkpoint_1" in path else 2000
    
    # 함수 호출
    result = get_checkpoints_info("jax")
    
    # 결과 검증
    assert len(result) == 2
    # 체크포인트는 스텝 기준으로 내림차순 정렬되므로 순서가 변경됨
    assert result[0]["step"] == 2000
    assert result[0]["name"] == "checkpoint_2"
    assert result[1]["step"] == 1000
    assert result[1]["name"] == "checkpoint_1"

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_checkpoints_info')
def test_get_latest_checkpoint(mock_get_info):
    """get_latest_checkpoint 함수 테스트"""
    # 목업 설정 - 내림차순 정렬된 데이터로 설정
    sorted_checkpoints = sorted(MOCK_CHECKPOINTS.copy(), key=lambda x: x.get("step", 0), reverse=True)
    mock_get_info.return_value = sorted_checkpoints
    
    # 함수 호출
    result = get_latest_checkpoint("jax")
    
    # 결과 검증
    assert result is not None
    assert result["step"] == 2000  # 최신 체크포인트
    assert result["name"] == "checkpoint_2"

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_checkpoints_info')
def test_get_latest_checkpoint_empty(mock_get_info):
    """get_latest_checkpoint 함수 테스트 - 체크포인트 없음"""
    # 목업 설정
    mock_get_info.return_value = []
    
    # 함수 호출
    result = get_latest_checkpoint("jax")
    
    # 결과 검증
    assert result is None

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_checkpoints_info')
def test_get_checkpoint_by_index(mock_get_info):
    """get_checkpoint_by_index 함수 테스트"""
    # 목업 설정
    mock_get_info.return_value = MOCK_CHECKPOINTS.copy()
    
    # 함수 호출
    result = get_checkpoint_by_index("jax", 0)
    
    # 결과 검증
    assert result is not None
    assert result["step"] == 1000
    assert result["name"] == "checkpoint_1"
    
    # 함수 호출 - 두 번째 체크포인트
    result = get_checkpoint_by_index("jax", 1)
    
    # 결과 검증
    assert result is not None
    assert result["step"] == 2000
    assert result["name"] == "checkpoint_2"
    
    # 잘못된 인덱스
    result = get_checkpoint_by_index("jax", 2)
    assert result is None

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_checkpoints_info')
def test_get_checkpoint_by_step(mock_get_info):
    """get_checkpoint_by_step 함수 테스트"""
    # 목업 설정
    mock_get_info.return_value = MOCK_CHECKPOINTS.copy()
    
    # 함수 호출 - 존재하는 스텝
    result = get_checkpoint_by_step("jax", 1000)
    
    # 결과 검증
    assert result is not None
    assert result["step"] == 1000
    assert result["name"] == "checkpoint_1"
    
    # 함수 호출 - 존재하지 않는 스텝
    result = get_checkpoint_by_step("jax", 3000)
    assert result is None

@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_jax_checkpoint_path', return_value='/checkpoints/jax_mnist')
@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_flax_checkpoint_path', return_value='/checkpoints/flax_mnist')
@mock.patch('jax_to_tfjs.checkpoint_utils.info.get_checkpoints_info')
@mock.patch('os.path.isdir')
@mock.patch('os.listdir')
def test_list_available_checkpoints(mock_listdir, mock_isdir, mock_get_info, mock_flax_path, mock_jax_path):
    """list_available_checkpoints 함수 테스트"""
    # 목업 설정
    mock_get_info.return_value = [MOCK_CHECKPOINTS[0]]  # 기본 디렉토리
    mock_listdir.return_value = ["subdir1"]
    mock_isdir.return_value = True
    
    # 함수 호출
    result = list_available_checkpoints("jax", verbose=False)
    
    # 결과 검증
    assert isinstance(result, list)
    assert len(result) == 1  # 기본 디렉토리의 체크포인트만 반환
    
    # 출력 테스트 (verbose=True)
    with mock.patch('builtins.print') as mock_print:
        list_available_checkpoints("jax", verbose=True)
        # print가 호출되었는지 확인
        assert mock_print.call_count > 0 