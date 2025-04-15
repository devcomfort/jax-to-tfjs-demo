"""
체크포인트 로딩 유틸리티

체크포인트를 로딩하는 함수들을 제공합니다.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import orbax.checkpoint as ocp

from jax_to_tfjs.paths import (
    get_jax_checkpoint_path, 
    get_flax_checkpoint_path
)
from jax_to_tfjs.checkpoint_utils.info import get_checkpoint_by_step, get_latest_checkpoint
from jax_to_tfjs.checkpoint_utils import validation

logger = logging.getLogger(__name__)

def load_checkpoint_by_step(model_type: str, step: int, subdir: Optional[str] = None):
    """
    특정 스텝의 체크포인트를 로드합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        step (int): 로드할 체크포인트의 스텝
        subdir (Optional[str]): 체크포인트의 하위 디렉토리 (없을 경우 기본 경로 사용)
        
    반환:
        체크포인트 객체
    """
    logger.info(f"스텝 {step}에 해당하는 {model_type} 체크포인트를 로드합니다.")
    
    # 체크포인트 정보 가져오기
    checkpoint_info = get_checkpoint_by_step(model_type, step, subdir)
    if checkpoint_info is None:
        raise ValueError(f"스텝 {step}에 해당하는 {model_type} 체크포인트를 찾을 수 없습니다.")
    
    # 체크포인트 경로 가져오기
    checkpoint_path = checkpoint_info['path']
    
    # 체크포인트 로드
    return load_checkpoint_by_path(checkpoint_path, model_type)

def load_jax_checkpoint(checkpoint_path: str):
    """
    JAX 체크포인트를 로드합니다.
    
    인자:
        checkpoint_path (str): 체크포인트 경로
        
    반환:
        로드된 JAX 체크포인트 객체
    """
    logger.info(f"JAX 체크포인트를 로드합니다: {checkpoint_path}")
    
    # Orbax 체크포인터 가져오기
    checkpointer = get_orbax_checkpointer()
    
    try:
        from pathlib import Path
        # 체크포인트 경로가 올바른지 확인
        checkpoint_path = Path(checkpoint_path)
        
        # 메타데이터 경로 생성
        metadata_path = checkpoint_path / "checkpoint"
        
        # 메타데이터 존재 여부 확인
        if not metadata_path.exists():
            raise ValueError(f"체크포인트 메타데이터 파일이 존재하지 않습니다: {metadata_path}")
            
        # 체크포인트 로드
        return checkpointer.restore(str(checkpoint_path))
    except Exception as e:
        logger.error(f"JAX 체크포인트 로드 중 오류 발생: {e}")
        raise ValueError(f"JAX 체크포인트를 로드할 수 없습니다: {checkpoint_path} - {e}")

def load_flax_checkpoint(checkpoint_path: str):
    """
    Flax 체크포인트를 로드합니다.
    
    인자:
        checkpoint_path (str): 체크포인트 경로
        
    반환:
        로드된 Flax 체크포인트 객체
    """
    logger.info(f"Flax 체크포인트를 로드합니다: {checkpoint_path}")
    
    try:
        from pathlib import Path
        import flax.serialization as serialization
        
        # 체크포인트 경로가 올바른지 확인
        checkpoint_path = Path(checkpoint_path)
        
        # msgpack 파일 찾기
        msgpack_files = list(checkpoint_path.glob("*.msgpack"))
        if not msgpack_files:
            raise ValueError(f"Flax 체크포인트 파일(*.msgpack)을 찾을 수 없습니다: {checkpoint_path}")
        
        # 첫 번째 msgpack 파일 로드
        with open(msgpack_files[0], "rb") as f:
            return serialization.from_bytes(None, f.read())
    except Exception as e:
        logger.error(f"Flax 체크포인트 로드 중 오류 발생: {e}")
        raise ValueError(f"Flax 체크포인트를 로드할 수 없습니다: {checkpoint_path} - {e}")

def load_checkpoint_by_path(checkpoint_path: str, model_type: Optional[str] = None):
    """
    주어진 경로에서 체크포인트를 로드합니다.
    
    인자:
        checkpoint_path (str): 체크포인트 경로
        model_type (Optional[str]): 모델 타입 ("jax" 또는 "flax"). 없으면 자동 감지 시도
        
    반환:
        체크포인트 객체
    """
    if model_type is None:
        # 체크포인트 유형 자동 감지 시도
        model_type = validation.get_checkpoint_type(checkpoint_path)
        
    if model_type == "jax":
        return load_jax_checkpoint(checkpoint_path)
    elif model_type == "flax":
        return load_flax_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"지원되지 않는 모델 타입: {model_type}. 'jax' 또는 'flax'만 지원됩니다.")

def load_latest_checkpoint(model_type: str, subdir: Optional[str] = None):
    """
    최신 체크포인트를 로드합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        subdir (Optional[str]): 체크포인트의 하위 디렉토리 (없을 경우 기본 경로 사용)
        
    반환:
        최신 체크포인트 객체
    """
    logger.info(f"최신 {model_type} 체크포인트를 로드합니다.")
    
    # 최신 체크포인트 정보 가져오기
    checkpoint_info = get_latest_checkpoint(model_type, subdir)
    if checkpoint_info is None:
        raise ValueError(f"{model_type} 체크포인트를 찾을 수 없습니다.")
    
    # 체크포인트 경로 가져오기
    checkpoint_path = checkpoint_info['path']
    
    logger.info(f"최신 체크포인트 경로: {checkpoint_path}")
    
    # 체크포인트 로드
    return load_checkpoint_by_path(checkpoint_path, model_type)

def get_orbax_checkpointer(model_type: str = "jax"):
    """
    Orbax 체크포인터를 생성합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        
    반환:
        Orbax 체크포인터 객체
    """
    # 최신 Orbax API를 사용하여 체크포인터 생성
    # 주의: API 호환성 검사 필요
    try:
        # Orbax 2.0 이상 버전
        if hasattr(ocp, 'PyTreeCheckpointerOptions'):
            options = ocp.PyTreeCheckpointerOptions(
                save_keep_periods=[50, 200, 1000],
                use_ocdbt=False
            )
            return ocp.PyTreeCheckpointer(options=options)
        # 이전 버전 (옵션 없이 초기화)
        else:
            return ocp.PyTreeCheckpointer()
    except Exception as e:
        logger.warning(f"Orbax 체크포인터 생성 중 오류: {e}")
        return ocp.PyTreeCheckpointer()

def create_checkpoint_manager(checkpoint_dir: str, name: str = "model"):
    """
    Orbax 체크포인트 매니저를 생성합니다.
    
    인자:
        checkpoint_dir (str): 체크포인트 디렉토리 경로
        name (str): 체크포인트 항목 이름 (기본값: "model")
        
    반환:
        Tuple[ocp.CheckpointManager, ocp.PyTreeCheckpointer]: 체크포인트 매니저와 체크포인터
    """
    # 경로 객체 생성
    path_obj = Path(checkpoint_dir)
    
    # 체크포인터 생성
    checkpointer = get_orbax_checkpointer()
    
    try:
        # Orbax API 버전 호환성 처리
        # 체크포인트 매니저 옵션 설정
        manager_options = ocp.CheckpointManagerOptions(
            max_to_keep=5,
            save_interval_steps=1000
        )
        
        try:
            # 최신 API - ocp.args 사용
            if hasattr(ocp, 'args') and hasattr(ocp.args, 'CheckpointManagerArgs'):
                manager_args = ocp.args.CheckpointManagerArgs(
                    directory=str(path_obj),
                    options=manager_options
                )
                
                checkpoint_manager = ocp.CheckpointManager(
                    manager_args,
                    item_handlers={name: checkpointer}
                )
            # 이전 API
            else:
                checkpoint_manager = ocp.CheckpointManager(
                    directory=str(path_obj),
                    checkpointers={name: checkpointer},
                    options=manager_options
                )
        except Exception as e:
            logger.warning(f"CheckpointManager 생성 중 오류, 기본 초기화 시도: {e}")
            # 가장 기본적인 방법으로 시도
            checkpoint_manager = ocp.CheckpointManager(
                directory=str(path_obj),
                checkpointers={name: checkpointer}
            )
    except Exception as e:
        logger.error(f"CheckpointManager 생성 실패: {e}")
        raise ValueError(f"체크포인트 매니저를 생성할 수 없습니다: {e}")
    
    return checkpoint_manager, checkpointer 