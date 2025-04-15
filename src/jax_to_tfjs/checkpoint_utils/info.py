"""
체크포인트 정보 검색 유틸리티

체크포인트 목록 탐색, 최신 체크포인트 찾기, 체크포인트 정보 해석 등의 기능을 제공합니다.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import orbax.checkpoint as ocp
from jax_to_tfjs.paths import get_jax_checkpoint_path, get_flax_checkpoint_path
from jax_to_tfjs.checkpoint_utils.validation import (
    validate_checkpoint, 
    extract_step_from_checkpoint,
    get_checkpoint_type
)

# 로깅 설정
logger = logging.getLogger(__name__)

def get_checkpoints_info(model_type: str, subdir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    지정된 모델 타입의 모든 체크포인트 정보를 반환합니다.

    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        subdir (Optional[str], 기본값=None): 체크포인트 하위 디렉토리

    반환:
        List[Dict[str, Any]]: 체크포인트 정보 목록 (비어있을 수 있음)
    """
    checkpoints = []

    # 모델 타입에 따른 체크포인트 기본 경로 가져오기
    if model_type == "jax":
        base_dir = get_jax_checkpoint_path(subdir)
    elif model_type == "flax":
        base_dir = get_flax_checkpoint_path(subdir)
    else:
        logger.error(f"지원하지 않는 모델 타입: {model_type}")
        return []

    # 체크포인트 디렉토리가 존재하는지 확인
    if not os.path.exists(base_dir):
        logger.warning(f"체크포인트 디렉토리가 존재하지 않습니다: {base_dir}")
        return []

    # 1. 최신 Orbax API를 사용하여 체크포인트 정보 조회 시도
    try:
        base_dir_path = Path(base_dir)
        
        # Orbax 2.0 API를 사용하여 체크포인트 매니저 생성
        checkpoint_args = ocp.args.CheckpointManagerArgs(
            directory=str(base_dir_path),
            checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
            create=False
        )
        
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_args,
            item_handlers={"model": checkpointer}
        )
        
        # 체크포인트 목록 조회
        try:
            checkpoint_ids = checkpoint_manager.list()
            
            for checkpoint_id in checkpoint_ids:
                try:
                    # 스텝 추출
                    try:
                        step = int(checkpoint_id)
                    except (ValueError, TypeError):
                        step = 0
                    
                    # 체크포인트 경로 계산
                    checkpoint_path = os.path.join(base_dir, checkpoint_id)
                    
                    # 메타데이터 조회 시도
                    timestamp = None
                    time_str = ""
                    
                    try:
                        metadata = checkpoint_manager.metadata(checkpoint_id)
                        if metadata and "timestamp" in metadata:
                            timestamp = metadata["timestamp"]
                            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        logger.debug(f"메타데이터 조회 실패: {e}")
                    
                    # 체크포인트 정보 구성
                    checkpoint_info = {
                        "path": checkpoint_path,
                        "name": f"checkpoint_{checkpoint_id}",
                        "model_type": model_type,
                        "step": step,
                        "timestamp": timestamp,
                        "datetime": time_str,
                        "orbax_managed": True
                    }
                    
                    # 메타데이터 정보 추가
                    if 'metadata' in locals() and metadata:
                        for key, value in metadata.items():
                            if key not in checkpoint_info and key != "timestamp":
                                checkpoint_info[key] = value
                    
                    checkpoints.append(checkpoint_info)
                except Exception as e:
                    logger.debug(f"체크포인트 ID {checkpoint_id} 처리 중 오류: {str(e)}")
            
            # 체크포인트를 찾았으면 정렬 후 반환
            if checkpoints:
                checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
                return checkpoints
                
        except Exception as e:
            logger.debug(f"체크포인트 목록 조회 실패: {str(e)}")
    
    except (ImportError, Exception) as e:
        logger.debug(f"Orbax API 초기화 실패: {str(e)}")
    
    # 2. 기존 방식으로 폴백 (디렉토리 구조 기반)
    try:
        # 디렉토리 내용 스캔
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            
            # 디렉토리인 경우에만 체크포인트로 간주
            if os.path.isdir(item_path):
                # 유효한 체크포인트인지 확인
                if validate_checkpoint(item_path, model_type):
                    # 체크포인트 스텝 추출
                    step = extract_step_from_checkpoint(item_path)
                    
                    # 시간 정보 획득 시도
                    timestamp = None
                    time_str = ""
                    try:
                        # 메타데이터 파일에서 시간 정보 확인
                        metadata_path = os.path.join(item_path, "_CHECKPOINT_METADATA")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                if "timestamp" in metadata:
                                    timestamp = metadata["timestamp"]
                                    time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        # 시간 정보를 얻을 수 없는 경우 파일 시간 사용
                        timestamp = os.path.getmtime(item_path)
                        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 체크포인트 정보 구성
                    checkpoint_info = {
                        "path": item_path,
                        "name": item,
                        "model_type": model_type,
                        "step": step if step is not None else 0,
                        "timestamp": timestamp,
                        "datetime": time_str,
                        "orbax_managed": False
                    }
                    
                    checkpoints.append(checkpoint_info)
                
                # 하위 디렉토리 검사 (depth=1)
                elif os.path.isdir(item_path):
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path) and validate_checkpoint(subitem_path, model_type):
                            # 체크포인트 스텝 추출
                            step = extract_step_from_checkpoint(subitem_path)
                            
                            # 시간 정보 획득 시도
                            timestamp = None
                            time_str = ""
                            try:
                                # 메타데이터 파일에서 시간 정보 확인
                                metadata_path = os.path.join(subitem_path, "_CHECKPOINT_METADATA")
                                if os.path.exists(metadata_path):
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                        if "timestamp" in metadata:
                                            timestamp = metadata["timestamp"]
                                            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            except Exception:
                                # 시간 정보를 얻을 수 없는 경우 파일 시간 사용
                                timestamp = os.path.getmtime(subitem_path)
                                time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            
                            # 체크포인트 정보 구성
                            checkpoint_info = {
                                "path": subitem_path,
                                "name": f"{item}/{subitem}",
                                "model_type": model_type,
                                "step": step if step is not None else 0,
                                "timestamp": timestamp,
                                "datetime": time_str,
                                "orbax_managed": False
                            }
                            
                            checkpoints.append(checkpoint_info)
        
        # 스텝 기준으로 내림차순 정렬
        checkpoints.sort(key=lambda x: x.get("step", 0), reverse=True)
        
    except Exception as e:
        logger.error(f"체크포인트 검색 중 오류 발생: {str(e)}")
    
    return checkpoints

def list_available_checkpoints(model_type: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    사용 가능한 모든 체크포인트를 나열합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        verbose (bool, 기본값=True): 자세한 정보 출력 여부
        
    반환:
        List[Dict[str, Any]]: 체크포인트 정보 목록
    """
    checkpoints = get_checkpoints_info(model_type)
    
    if not checkpoints:
        print(f"{model_type} 모델 타입에 대한 체크포인트가 없습니다.")
        return []
    
    if verbose:
        print(f"\n===== {model_type.upper()} 체크포인트 목록 =====")
        for i, checkpoint in enumerate(checkpoints):
            step_info = f", 스텝: {checkpoint['step']}" if "step" in checkpoint else ""
            time_info = f", 시간: {checkpoint.get('datetime', '')}" if checkpoint.get('datetime') else ""
            orbax_info = " [Orbax]" if checkpoint.get("orbax_managed", False) else ""
            print(f"{i+1}. {checkpoint['name']}{step_info}{time_info}{orbax_info}")
        print("================================")
    
    return checkpoints

def get_latest_checkpoint(model_type: str, subdir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    지정된 모델 타입의 최신 체크포인트 정보를 반환합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        subdir (Optional[str], 기본값=None): 체크포인트 하위 디렉토리
        
    반환:
        Optional[Dict[str, Any]]: 최신 체크포인트 정보 또는 체크포인트가 없는 경우 None
    """
    # 1. 최신 Orbax API를 사용하여 체크포인트 정보 조회
    try:
        # 모델 타입에 따른 체크포인트 기본 경로 가져오기
        if model_type == "jax":
            base_dir = get_jax_checkpoint_path(subdir)
        elif model_type == "flax":
            base_dir = get_flax_checkpoint_path(subdir)
        else:
            logger.error(f"지원하지 않는 모델 타입: {model_type}")
            return None
            
        # 체크포인트 디렉토리가 존재하는지 확인
        if not os.path.exists(base_dir):
            logger.warning(f"체크포인트 디렉토리가 존재하지 않습니다: {base_dir}")
            return None
        
        base_dir_path = Path(base_dir)
        
        # Orbax 2.0 API를 사용하여 체크포인트 매니저 생성
        checkpoint_args = ocp.args.CheckpointManagerArgs(
            directory=str(base_dir_path),
            checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
            create=False
        )
        
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_args,
            item_handlers={"model": checkpointer}
        )
        
        # 최신 체크포인트 ID 조회
        try:
            last_checkpoint = checkpoint_manager.latest()
            
            if last_checkpoint:
                # 스텝 추출
                try:
                    step = int(last_checkpoint)
                except (ValueError, TypeError):
                    step = 0
                
                # 체크포인트 경로 계산
                checkpoint_path = os.path.join(base_dir, last_checkpoint)
                
                # 메타데이터 조회 시도
                timestamp = None
                time_str = ""
                
                try:
                    metadata = checkpoint_manager.metadata(last_checkpoint)
                    if metadata and "timestamp" in metadata:
                        timestamp = metadata["timestamp"]
                        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logger.debug(f"메타데이터 조회 실패: {e}")
                
                # 체크포인트 정보 구성
                checkpoint_info = {
                    "path": checkpoint_path,
                    "name": f"checkpoint_{last_checkpoint}",
                    "model_type": model_type,
                    "step": step,
                    "timestamp": timestamp,
                    "datetime": time_str,
                    "orbax_managed": True
                }
                
                # 메타데이터 정보 추가
                if 'metadata' in locals() and metadata:
                    for key, value in metadata.items():
                        if key not in checkpoint_info and key != "timestamp":
                            checkpoint_info[key] = value
                
                return checkpoint_info
                
        except Exception as e:
            logger.debug(f"최신 체크포인트 조회 실패: {str(e)}")
    
    except (ImportError, Exception) as e:
        logger.debug(f"Orbax API 초기화 실패: {str(e)}")
    
    # 2. 기존 방식으로 폴백
    checkpoints = get_checkpoints_info(model_type, subdir)
    
    if not checkpoints:
        return None
    
    # 이미 스텝 기준으로 내림차순 정렬된 상태
    return checkpoints[0]

def get_checkpoint_by_index(model_type: str, index: int, subdir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    인덱스로 체크포인트 정보를 검색합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        index (int): 체크포인트 인덱스 (0부터 시작)
        subdir (Optional[str], 기본값=None): 체크포인트 하위 디렉토리
        
    반환:
        Optional[Dict[str, Any]]: 체크포인트 정보 또는 인덱스가 범위를 벗어나는 경우 None
    """
    checkpoints = get_checkpoints_info(model_type, subdir)
    
    if not checkpoints:
        return None
    
    if index < 0 or index >= len(checkpoints):
        return None
    
    return checkpoints[index]

def get_checkpoint_by_step(model_type: str, step: int, subdir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    스텝 번호로 체크포인트 정보를 검색합니다.
    
    인자:
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        step (int): 체크포인트 스텝 번호
        subdir (Optional[str], 기본값=None): 체크포인트 하위 디렉토리
        
    반환:
        Optional[Dict[str, Any]]: 체크포인트 정보 또는 해당 스텝이 없는 경우 None
    """
    # 1. 최신 Orbax API를 사용하여 특정 스텝 체크포인트 조회
    try:
        # 모델 타입에 따른 체크포인트 기본 경로 가져오기
        if model_type == "jax":
            base_dir = get_jax_checkpoint_path(subdir)
        elif model_type == "flax":
            base_dir = get_flax_checkpoint_path(subdir)
        else:
            logger.error(f"지원하지 않는 모델 타입: {model_type}")
            return None
            
        # 체크포인트 디렉토리가 존재하는지 확인
        if not os.path.exists(base_dir):
            logger.warning(f"체크포인트 디렉토리가 존재하지 않습니다: {base_dir}")
            return None
        
        base_dir_path = Path(base_dir)
        
        # Orbax 2.0 API를 사용하여 체크포인트 매니저 생성
        checkpoint_args = ocp.args.CheckpointManagerArgs(
            directory=str(base_dir_path),
            checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
            create=False
        )
        
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_args,
            item_handlers={"model": checkpointer}
        )
        
        # 체크포인트 목록 조회
        try:
            checkpoint_ids = checkpoint_manager.list()
            
            # 스텝과 일치하는 체크포인트 ID 찾기
            step_str = str(step)
            if step_str in checkpoint_ids:
                # 체크포인트 경로 계산
                checkpoint_path = os.path.join(base_dir, step_str)
                
                # 메타데이터 조회 시도
                timestamp = None
                time_str = ""
                
                try:
                    metadata = checkpoint_manager.metadata(step_str)
                    if metadata and "timestamp" in metadata:
                        timestamp = metadata["timestamp"]
                        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logger.debug(f"메타데이터 조회 실패: {e}")
                
                # 체크포인트 정보 구성
                checkpoint_info = {
                    "path": checkpoint_path,
                    "name": f"checkpoint_{step}",
                    "model_type": model_type,
                    "step": step,
                    "timestamp": timestamp,
                    "datetime": time_str,
                    "orbax_managed": True
                }
                
                # 메타데이터 정보 추가
                if 'metadata' in locals() and metadata:
                    for key, value in metadata.items():
                        if key not in checkpoint_info and key != "timestamp":
                            checkpoint_info[key] = value
                
                return checkpoint_info
                
        except Exception as e:
            logger.debug(f"체크포인트 조회 실패: {str(e)}")
    
    except (ImportError, Exception) as e:
        logger.debug(f"Orbax API 초기화 실패: {str(e)}")
    
    # 2. 기존 방식으로 폴백
    checkpoints = get_checkpoints_info(model_type, subdir)
    
    for checkpoint in checkpoints:
        if checkpoint.get("step") == step:
            return checkpoint
    
    return None 