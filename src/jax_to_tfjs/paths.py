"""
체크포인트 및 모델 파일 경로 관리 모듈
"""
import os
from pathlib import Path

# 현재 파일의 디렉토리 경로 (src/jax_to_tfjs)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 디렉토리 (src의 상위 디렉토리)
ROOT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))

# 체크포인트 디렉토리 경로
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# JAX 모델 체크포인트 경로
JAX_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "jax_mnist")

# Flax 모델 체크포인트 경로
FLAX_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "flax_mnist")

# 결과 디렉토리 경로
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# 웹 모델 출력 디렉토리 경로
WEB_MODEL_DIR = os.path.join(ROOT_DIR, "web", "model")

# 필요한 디렉토리 생성 함수
def ensure_directories():
    """필요한 디렉토리가 존재하는지 확인하고 없으면 생성"""
    for dir_path in [CHECKPOINT_DIR, JAX_CHECKPOINT_DIR, FLAX_CHECKPOINT_DIR, RESULTS_DIR, WEB_MODEL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

# 모듈 임포트 시 디렉토리 생성
ensure_directories()

def get_jax_checkpoint_path(subdir=None):
    """JAX 모델 체크포인트 경로 반환
    
    Args:
        subdir: 추가 하위 디렉토리 (예: "experiment1")
    
    Returns:
        완전한 경로
    """
    if subdir:
        path = os.path.join(JAX_CHECKPOINT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    return JAX_CHECKPOINT_DIR

def get_flax_checkpoint_path(subdir=None):
    """Flax 모델 체크포인트 경로 반환
    
    Args:
        subdir: 추가 하위 디렉토리 (예: "experiment1")
    
    Returns:
        완전한 경로
    """
    if subdir:
        path = os.path.join(FLAX_CHECKPOINT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    return FLAX_CHECKPOINT_DIR

def get_results_path(filename=None):
    """결과 파일 경로 반환
    
    Args:
        filename: 결과 파일 이름 (예: "confusion_matrix.png")
    
    Returns:
        완전한 경로
    """
    if filename:
        return os.path.join(RESULTS_DIR, filename)
    return RESULTS_DIR

def get_web_model_path(subdir=None):
    """웹 모델 경로 반환
    
    Args:
        subdir: 추가 하위 디렉토리 (예: "jax" 또는 "flax")
    
    Returns:
        완전한 경로
    """
    if subdir:
        path = os.path.join(WEB_MODEL_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    return WEB_MODEL_DIR 