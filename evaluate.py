#!/usr/bin/env python
"""
JAX/Flax MNIST 모델 평가 실행 스크립트

rye run evaluate 명령으로 실행할 수 있는 진입점입니다.
"""
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    평가 도구 진입점 함수
    """
    try:
        from jax_to_tfjs.evaluation.cli import main as eval_main
        logger.info("JAX/Flax MNIST 모델 평가 도구를 시작합니다")
        return eval_main()
    except ImportError as e:
        logger.error(f"모듈 임포트 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 