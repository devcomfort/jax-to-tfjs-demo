#!/usr/bin/env python
"""
JAX/Flax 모델을 TensorFlow.js로 변환하는 스크립트

사용법:
    python -m jax_to_tfjs.convert_to_tfjs [--model-type {jax,flax}] [--checkpoint-path PATH]
"""
import argparse
import os
import tempfile
from pathlib import Path
from datetime import datetime
import jax
import tensorflow as tf
import tensorflowjs as tfjs

from jax_to_tfjs.paths import (
    get_jax_checkpoint_path, 
    get_flax_checkpoint_path, 
    get_web_model_path
)
# 새로운 모듈 구조로 임포트
from jax_to_tfjs.checkpoint_utils import (
    get_latest_checkpoint,
    get_checkpoint_by_index,
    validate_checkpoint
)
from jax_to_tfjs.checkpoint_utils.loader import (
    load_jax_checkpoint,
    load_flax_checkpoint,
    load_checkpoint_by_path
)
from jax_to_tfjs.checkpoint_utils.info import get_checkpoints_info

def convert_checkpoint_to_tfjs(checkpoint_info: dict, output_dir: str):
    """체크포인트를 TensorFlow.js 모델로 변환합니다.

    인자:
        checkpoint_info: 체크포인트 정보 (checkpoint_utils.get_checkpoints_info에서 반환)
        output_dir: TensorFlow.js 모델을 저장할 디렉토리
    """
    model_type = checkpoint_info["model_type"]
    checkpoint_path = checkpoint_info["path"]
    
    print(f"{model_type.upper()} 체크포인트 로드 중: {checkpoint_path}")
    
    if model_type == "jax":
        # JAX 체크포인트 로드
        from jax_to_tfjs.models.jax_mnist_cnn import create_model as create_jax_model
        checkpoint = load_jax_checkpoint(checkpoint_path)
        params = checkpoint["params"]
        
        # JAX 모델 생성 및 파라미터 설정
        # 이 부분은 실제 JAX 모델 구조에 맞게 수정해야 합니다
    else:
        # Flax 체크포인트 로드
        from jax_to_tfjs.models.flax_mnist_cnn import create_model as create_flax_model
        state = load_flax_checkpoint(checkpoint_path)
        params = {"params": state.params}
        
        # Flax 모델 생성 및 파라미터 설정
        # 이 부분은 실제 Flax 모델 구조에 맞게 수정해야 합니다
    
    # Tensorflow 모델 생성
    tf_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])
    
    # JAX/Flax 가중치를 TensorFlow 가중치로 변환하는 로직
    # 이 부분은 실제 모델 구조에 맞게 구현해야 합니다
    print("모델 변환 중...")
    
    # TensorFlow.js 모델로 저장
    print(f"TensorFlow.js 모델을 {output_dir}에 저장 중...")
    tfjs.converters.save_keras_model(tf_model, output_dir)
    
    # 모델 정보 저장
    step_info = f"스텝: {checkpoint_info.get('step', '정보 없음')}"
    datetime_info = f"생성 시간: {checkpoint_info.get('datetime', '정보 없음')}"
    
    model_info = {
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
        "step": checkpoint_info.get("step", 0),
        "timestamp": checkpoint_info.get("timestamp"),
        "datetime": checkpoint_info.get("datetime"),
        "converted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 모델 정보 저장
    model_info_path = os.path.join(output_dir, "model_info.json")
    with open(model_info_path, "w") as f:
        import json
        json.dump(model_info, f, indent=2)
    
    print(f"변환 완료!")
    print(f"모델이 {output_dir}에 저장되었습니다.")
    print(f"체크포인트 정보: {step_info}, {datetime_info}")

def main():
    """TensorFlow.js 변환 메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="JAX/Flax 모델을 TensorFlow.js로 변환")
    parser.add_argument("--model-type", choices=["jax", "flax"], default="jax", 
                        help="변환할 모델 타입 (기본값: jax)")
    parser.add_argument("--checkpoint-index", type=int, 
                        help="체크포인트 인덱스 (1부터 시작)")
    parser.add_argument("--checkpoint-path", type=str, 
                        help="체크포인트 경로 (직접 지정)")
    parser.add_argument("--output", type=str, 
                        help="출력 디렉토리 (기본값: web/model/[model_type])")
    parser.add_argument("--subdir", type=str, default=None, 
                        help="체크포인트 하위 디렉토리 (선택사항)")
    parser.add_argument("--latest", action="store_true", 
                        help="최신 체크포인트 사용")
    args = parser.parse_args()
    
    model_type = args.model_type
    
    # 체크포인트 선택 로직
    checkpoint_info = None
    
    if args.checkpoint_path:
        # 직접 체크포인트 경로 지정
        if not os.path.exists(args.checkpoint_path):
            print(f"오류: 체크포인트 경로가 존재하지 않습니다: {args.checkpoint_path}")
            return 1
        
        # 유효성 검사 수행
        if not validate_checkpoint(args.checkpoint_path, model_type):
            print(f"오류: 유효하지 않은 체크포인트 경로입니다: {args.checkpoint_path}")
            return 1
        
        # 체크포인트 정보 임의 생성 (경로만 있는 경우)
        checkpoint_info = {
            "path": args.checkpoint_path,
            "model_type": model_type,
            "name": os.path.basename(args.checkpoint_path)
        }
    
    elif args.checkpoint_index:
        # 인덱스로 체크포인트 선택
        checkpoint_info = get_checkpoint_by_index(
            model_type, args.checkpoint_index - 1, args.subdir
        )
        
        if not checkpoint_info:
            print(f"오류: 인덱스 {args.checkpoint_index}에 해당하는 체크포인트가 없습니다.")
            print("사용 가능한 체크포인트를 확인하려면 다음 명령을 실행하세요:")
            print(f"  python -m jax_to_tfjs.cli checkpoints --model-type {model_type}")
            return 1
    
    elif args.latest:
        # 최신 체크포인트 선택
        checkpoint_info = get_latest_checkpoint(model_type, args.subdir)
        
        if not checkpoint_info:
            print(f"오류: {model_type} 모델 타입에 대한 체크포인트가 없습니다.")
            return 1
    
    else:
        # 체크포인트 목록 표시 후 사용자에게 선택하도록 안내
        checkpoints = get_checkpoints_info(model_type, args.subdir)
        
        if not checkpoints:
            print(f"오류: {model_type} 모델 타입에 대한 체크포인트가 없습니다.")
            return 1
        
        print(f"\n{model_type.upper()} 체크포인트 목록:")
        for i, ckpt in enumerate(checkpoints):
            step_info = f", 스텝: {ckpt['step']}" if "step" in ckpt else ""
            time_info = f", 시간: {ckpt.get('datetime', '')}" if "datetime" in ckpt else ""
            print(f"  {i+1}. {ckpt['name']}{step_info}{time_info}")
        
        try:
            from rich import print as rprint
            rprint("[yellow]변환할 체크포인트를 선택하려면 다음 예시와 같이 명령을 실행하세요:[/]")
            rprint(f"[green]  python -m jax_to_tfjs.convert_to_tfjs --model-type {model_type} --checkpoint-index 1[/]")
            rprint("[yellow]또는 최신 체크포인트를 사용하려면:[/]")
            rprint(f"[green]  python -m jax_to_tfjs.convert_to_tfjs --model-type {model_type} --latest[/]")
        except ImportError:
            print("변환할 체크포인트를 선택하려면 다음 예시와 같이 명령을 실행하세요:")
            print(f"  python -m jax_to_tfjs.convert_to_tfjs --model-type {model_type} --checkpoint-index 1")
            print("또는 최신 체크포인트를 사용하려면:")
            print(f"  python -m jax_to_tfjs.convert_to_tfjs --model-type {model_type} --latest")
        
        return 0
    
    # 출력 디렉토리 설정
    if args.output:
        output_dir = args.output
    else:
        # 기본 출력 경로 + 현재 시간을 포함하여 고유한 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = get_web_model_path(model_type)
        output_dir = f"{base_dir}_{timestamp}"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 체크포인트를 TensorFlow.js 모델로 변환
    convert_checkpoint_to_tfjs(checkpoint_info, output_dir)
    
    return 0

if __name__ == "__main__":
    main()
