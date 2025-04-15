"""
JAX/Flax 모델 평가 모듈의 __main__ 진입점

이 모듈을 직접 실행할 수 있도록 합니다:
python -m jax_to_tfjs.evaluation
"""
import sys
from jax_to_tfjs.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main()) 