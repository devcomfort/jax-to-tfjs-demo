"""
체크포인트 유효성 검증 유틸리티

체크포인트가 유효한지 검사하는 기능을 제공합니다.
"""
import os
import re
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Optional, Tuple

def validate_checkpoint(checkpoint_path: str, model_type: str) -> bool:
    """
    주어진 체크포인트 경로가 유효한지 확인합니다.
    Orbax API를 사용하여 체크포인트의 유효성을 검증합니다.
    
    인자:
        checkpoint_path (str): 체크포인트 경로
        model_type (str): 모델 타입 ("jax" 또는 "flax")
        
    반환:
        bool: 체크포인트가 유효하면 True, 아니면 False
    """
    # 기본적인 경로 유효성 검사
    if not os.path.exists(checkpoint_path) or not os.path.isdir(checkpoint_path):
        return False
    
    # 모델 타입에 따른 추가 검증
    if model_type not in ["jax", "flax"]:
        return False
    
    # Orbax를 사용하여 유효성 검증 시도
    try:
        # 최신 Orbax API를 사용하여 체크포인트 유효성 검증
        path_obj = Path(checkpoint_path)
        
        try:
            # 1. 직접 체크포인트 매니저로 검증 시도
            checkpoint_args = ocp.args.CheckpointManagerArgs(
                directory=str(path_obj),
                checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
                create=False
            )
            
            checkpointer = ocp.PyTreeCheckpointer()
            checkpoint_manager = ocp.CheckpointManager(
                checkpoint_args,
                item_handlers={"model": checkpointer}
            )
            
            # 체크포인트 목록 조회 시도
            checkpoint_ids = checkpoint_manager.list()
            if checkpoint_ids:
                return True
        except Exception:
            pass
        
        # 2. 메타데이터 파일 검증
        metadata_files = ["checkpoint", "_CHECKPOINT_METADATA"]
        for file in metadata_files:
            if os.path.exists(os.path.join(checkpoint_path, file)):
                return True
        
        # 3. model 디렉토리가 유효한 체크포인트인지 확인
        model_dir = os.path.join(checkpoint_path, "model")
        if os.path.isdir(model_dir):
            try:
                # model 디렉토리에서 체크포인트 매니저 초기화 시도
                model_path = Path(model_dir)
                model_args = ocp.args.CheckpointManagerArgs(
                    directory=str(model_path),
                    checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
                    create=False
                )
                
                model_manager = ocp.CheckpointManager(
                    model_args,
                    item_handlers={"model": checkpointer}
                )
                
                # 체크포인트 목록 조회 시도
                model_checkpoint_ids = model_manager.list()
                if model_checkpoint_ids:
                    return True
            except Exception:
                # 실패하더라도 model 디렉토리에 필요한 파일이 있는지 확인
                try:
                    # 직접 체크포인트 유효성 검증
                    restore_args = ocp.args.PyTreeCheckpointReaderArgs(directory=model_dir)
                    checkpoint = checkpointer.read(restore_args)
                    return True
                except Exception:
                    # 메타데이터 파일 확인
                    for file in metadata_files:
                        if os.path.exists(os.path.join(model_dir, file)):
                            return True
                    
                    # OCDBT 형식 체크
                    if os.path.exists(os.path.join(model_dir, "manifest.ocdbt")):
                        return True
        
        return False
            
    except Exception:
        # Orbax API가 실패하면 기존 검증 방식으로 폴백
        
        # 메타데이터 파일 확인
        metadata_files = ["checkpoint", "_CHECKPOINT_METADATA"]
        metadata_found = False
        
        for file in metadata_files:
            if os.path.exists(os.path.join(checkpoint_path, file)):
                metadata_found = True
                break
                
        if metadata_found:
            return True
            
        # model 디렉토리 확인
        model_dir = os.path.join(checkpoint_path, "model")
        if os.path.isdir(model_dir):
            return True
            
        return False

def is_checkpoint_directory(directory_path: str) -> bool:
    """
    주어진 디렉토리가 체크포인트 디렉토리인지 확인합니다.
    
    인자:
        directory_path: 검사할 디렉토리 경로
        
    반환:
        bool: 체크포인트 디렉토리이면 True, 아니면 False
    """
    # Orbax를 사용하여 검증 시도
    try:
        path_obj = Path(directory_path)
        
        # 1. 최신 Orbax API를 사용하여 체크포인트 매니저 초기화 시도
        try:
            checkpoint_args = ocp.args.CheckpointManagerArgs(
                directory=str(path_obj),
                checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
                create=False
            )
            
            checkpointer = ocp.PyTreeCheckpointer()
            checkpoint_manager = ocp.CheckpointManager(
                checkpoint_args,
                item_handlers={"model": checkpointer}
            )
            
            # 체크포인트 목록 조회 시도
            checkpoint_ids = checkpoint_manager.list()
            if checkpoint_ids:
                return True
        except Exception:
            pass
        
        # 2. 직접 체크포인트 구조 확인 시도
        try:
            restore_args = ocp.args.PyTreeCheckpointReaderArgs(directory=directory_path)
            checkpoint = ocp.PyTreeCheckpointer().read(restore_args)
            return True
        except Exception:
            pass
            
    except Exception:
        pass
    
    # 기존 방식으로 폴백
    # 디렉토리 존재 여부 확인
    if not os.path.isdir(directory_path):
        return False
    
    # 체크포인트 파일 존재 여부 확인
    metadata_files = ["checkpoint", "_CHECKPOINT_METADATA"]
    metadata_found = False
    
    for file in metadata_files:
        if os.path.exists(os.path.join(directory_path, file)):
            metadata_found = True
            break
            
    if metadata_found:
        return True
        
    # model 디렉토리 체크
    model_dir = os.path.join(directory_path, "model")
    if os.path.isdir(model_dir):
        return True
        
    return False

def extract_step_from_checkpoint(checkpoint_path: str) -> Optional[int]:
    """
    체크포인트 경로에서 스텝 번호를 추출합니다.
    
    인자:
        checkpoint_path: 체크포인트 경로
        
    반환:
        Optional[int]: 스텝 번호 또는 스텝 번호를 추출할 수 없는 경우 None
    """
    # Orbax 체크포인트 매니저에서 스텝 추출 시도
    try:
        path_obj = Path(checkpoint_path)
        
        # 1. 최신 Orbax API를 사용하여 체크포인트 매니저 초기화 시도
        try:
            checkpoint_args = ocp.args.CheckpointManagerArgs(
                directory=str(path_obj),
                checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
                create=False
            )
            
            checkpointer = ocp.PyTreeCheckpointer()
            checkpoint_manager = ocp.CheckpointManager(
                checkpoint_args,
                item_handlers={"model": checkpointer}
            )
            
            # 최신 체크포인트 ID 조회 시도
            latest_id = checkpoint_manager.latest()
            if latest_id:
                try:
                    return int(latest_id)
                except ValueError:
                    pass
        except Exception:
            pass
        
        # 2. model 디렉토리에서 스텝 추출 시도
        model_dir = os.path.join(checkpoint_path, "model")
        if os.path.isdir(model_dir):
            model_path = Path(model_dir)
            try:
                model_args = ocp.args.CheckpointManagerArgs(
                    directory=str(model_path),
                    checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
                    create=False
                )
                
                model_manager = ocp.CheckpointManager(
                    model_args,
                    item_handlers={"model": checkpointer}
                )
                
                # 최신 체크포인트 ID 조회 시도
                latest_model_id = model_manager.latest()
                if latest_model_id:
                    try:
                        return int(latest_model_id)
                    except ValueError:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    
    # 기존 방식으로 폴백
    # 경로에서 스텝 번호 추출 시도
    match = re.search(r'checkpoint_(\d+)$', checkpoint_path)
    if match:
        return int(match.group(1))
    
    # 메타데이터 파일에서 스텝 번호 추출 시도
    metadata_files = ["checkpoint", "_CHECKPOINT_METADATA"]
    
    for metadata_file in metadata_files:
        metadata_path = os.path.join(checkpoint_path, metadata_file)
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if "step" in metadata:
                        return metadata["step"]
            except (json.JSONDecodeError, IOError):
                pass
    
    return None

def get_checkpoint_type(checkpoint_path: str) -> Optional[str]:
    """
    체크포인트의 타입을 자동으로 감지합니다.
    
    인자:
        checkpoint_path: 체크포인트 경로
        
    반환:
        Optional[str]: "jax", "flax", 또는 None (감지할 수 없는 경우)
    """
    # 체크포인트가 존재하는지 확인
    if not os.path.exists(checkpoint_path) or not os.path.isdir(checkpoint_path):
        return None
    
    # 경로 기반 추론
    if "/jax_mnist/" in checkpoint_path or "jax" in os.path.basename(checkpoint_path).lower():
        return "jax"
    elif "/flax_mnist/" in checkpoint_path or "flax" in os.path.basename(checkpoint_path).lower():
        return "flax"
    
    # Orbax 메타데이터에서 모델 타입 추출 시도
    try:
        path_obj = Path(checkpoint_path)
        
        # 최신 Orbax API를 사용하여 체크포인트 매니저 초기화 시도
        checkpoint_args = ocp.args.CheckpointManagerArgs(
            directory=str(path_obj),
            checkpoint_id_format=ocp.args.StandardCheckpointIdFormat(),
            create=False
        )
        
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_args,
            item_handlers={"model": checkpointer}
        )
        
        # 최신 체크포인트 ID 조회 시도
        latest_id = checkpoint_manager.latest()
        if latest_id:
            try:
                # 메타데이터 조회 시도
                metadata = checkpoint_manager.metadata(latest_id)
                if metadata and "model_type" in metadata:
                    model_type = metadata["model_type"].lower()
                    if model_type in ["jax", "flax"]:
                        return model_type
            except Exception:
                pass
    except Exception:
        pass
    
    # 메타데이터 파일에서 모델 타입 추출 시도
    metadata_files = ["checkpoint", "_CHECKPOINT_METADATA"]
    
    for metadata_file in metadata_files:
        metadata_path = os.path.join(checkpoint_path, metadata_file)
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if "model_type" in metadata:
                        model_type = metadata["model_type"].lower()
                        if model_type in ["jax", "flax"]:
                            return model_type
            except (json.JSONDecodeError, IOError):
                pass
    
    # 구조 기반 추론: 직접 체크포인트를 로드해서 모델 구조 검사
    try:
        # JAX 체크포인트 로드 시도
        from jax_to_tfjs.models.jax_mnist_cnn import load_checkpoint as load_jax_checkpoint
        try:
            jax_model = load_jax_checkpoint(checkpoint_path)
            if jax_model is not None:
                return "jax"
        except Exception:
            pass
            
        # Flax 체크포인트 로드 시도
        from jax_to_tfjs.models.flax_mnist_cnn import load_checkpoint as load_flax_checkpoint
        try:
            flax_model = load_flax_checkpoint(checkpoint_path)
            if flax_model is not None:
                return "flax"
        except Exception:
            pass
    except ImportError:
        pass
    
    return None 