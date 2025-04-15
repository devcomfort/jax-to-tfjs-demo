#!/usr/bin/env python
"""
JAX/Flax MNIST 모델 평가 실행 스크립트

rye run evaluate 명령으로 실행할 수 있는 진입점입니다.
"""
import sys
from jax_to_tfjs.evaluation.cli import main

if __name__ == "__main__":
    print("JAX/Flax MNIST 모델 평가 도구 실행 중...")
    sys.exit(main()) 