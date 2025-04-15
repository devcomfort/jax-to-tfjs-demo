#!/usr/bin/env python
"""
JAX-to-TFJS 변환 CLI 도구

이 명령줄 인터페이스는 JAX 및 FLAX 모델 관리와 TFJS 변환 작업을 위한 도구를 제공합니다.

사용법:
    python -m jax_to_tfjs.cli checkpoints [--model-type jax|flax|all]
    python -m jax_to_tfjs.cli convert [--model-type jax|flax] [--checkpoint-path PATH] [--output-dir DIR]

명령:
    checkpoints    사용 가능한 체크포인트 목록 표시
    convert        JAX/FLAX 모델을 TFJS 형식으로 변환
"""

import argparse
import os
import subprocess
import sys
import traceback
from typing import List, Dict, Optional

# 새로운 checkpoint_utils 모듈에서 함수 임포트
from .checkpoint_utils import list_available_checkpoints

def list_checkpoints_command(args) -> None:
    """
    사용 가능한 체크포인트 목록을 표시합니다.
    
    인자:
        args: 명령줄 인수 (model_type 포함)
    """
    model_type = args.model_type
    list_available_checkpoints(model_type)

def convert_command(args) -> None:
    """
    JAX/FLAX 모델을 TFJS 형식으로 변환합니다.
    
    인자:
        args: 명령줄 인수 (model_type, checkpoint_path, output_dir 포함)
    """
    from jax_to_tfjs.convert_to_tfjs import main as convert_main
    
    # 명령줄 인수 준비
    original_argv = sys.argv
    
    # convert_to_tfjs.py에 전달할 인수 설정
    sys.argv = [
        'convert_to_tfjs.py',
        f'--model-type={args.model_type}',
    ]
    
    # 체크포인트 경로가 제공된 경우 추가
    if args.checkpoint_path:
        sys.argv.append(f'--checkpoint-path={args.checkpoint_path}')
        
    # 출력 디렉토리가 제공된 경우 추가
    if args.output_dir:
        sys.argv.append(f'--output={args.output_dir}')
    
    # 최신 체크포인트 사용 플래그 추가
    if args.latest:
        sys.argv.append('--latest')
    
    # 체크포인트 인덱스가 제공된 경우 추가
    if args.checkpoint_index:
        sys.argv.append(f'--checkpoint-index={args.checkpoint_index}')
    
    try:
        # convert_to_tfjs.py의 main 함수 호출
        convert_main()
    except Exception as e:
        print(f"변환 중 오류 발생: {str(e)}")
        traceback.print_exc()
    finally:
        # 원래 명령줄 인수 복원
        sys.argv = original_argv

def main():
    """
    JAX-to-TFJS CLI 도구의 메인 함수
    """
    parser = argparse.ArgumentParser(
        description="JAX 및 FLAX 모델 관리 및 TFJS 변환 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="수행할 명령")
    
    # 체크포인트 목록 명령
    checkpoint_parser = subparsers.add_parser(
        "checkpoints", 
        help="사용 가능한 체크포인트 목록 표시"
    )
    checkpoint_parser.add_argument(
        "--model-type", 
        choices=["jax", "flax", "all"],
        default="all",
        help="표시할 체크포인트의 모델 타입"
    )
    checkpoint_parser.set_defaults(func=list_checkpoints_command)
    
    # 변환 명령
    convert_parser = subparsers.add_parser(
        "convert", 
        help="JAX/FLAX 모델을 TFJS 형식으로 변환"
    )
    convert_parser.add_argument(
        "--model-type", 
        choices=["jax", "flax"],
        default="jax",
        help="변환할 모델 타입"
    )
    convert_parser.add_argument(
        "--checkpoint-path", 
        type=str,
        help="변환할 체크포인트 경로"
    )
    convert_parser.add_argument(
        "--output-dir", 
        type=str,
        default="./tfjs_model",
        help="TFJS 모델 출력 디렉토리"
    )
    convert_parser.add_argument(
        "--latest",
        action="store_true",
        help="최신 체크포인트 사용"
    )
    convert_parser.add_argument(
        "--checkpoint-index",
        type=int,
        help="특정 체크포인트 인덱스 사용"
    )
    convert_parser.set_defaults(func=convert_command)
    
    # 인수 파싱 및 명령 실행
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 