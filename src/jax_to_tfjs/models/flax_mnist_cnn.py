"""
FLAX MNIST CNN 모델 실행 모듈

이 모듈은 FLAX 모델을 사용하여 MNIST 데이터셋을 학습하는 실행 스크립트를 제공합니다.
모델 구현과 유틸리티 함수는 별도 모듈로 분리되어 가독성과 유지보수성을 개선했습니다.
"""

import argparse
import logging
import os

from .flax.model_manager import FlaxModelManager
from .common.train_utils import train_and_evaluate


# JAX/FLAX 경고 비활성화
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow 경고 비활성화
logging.getLogger("flax").setLevel(
    logging.ERROR
)  # FLAX 경고를 ERROR 레벨 이상으로만 표시


def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="FLAX MNIST 모델 학습")
    parser.add_argument(
        "--epochs", type=int, default=5, help="훈련 에포크 수 (기본값: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="학습률 (기본값: 0.001)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="학습 후 모델 평가 수행"
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="체크포인트를 저장할 하위 디렉토리 (선택사항)",
    )
    args = parser.parse_args()

    print(f"FLAX MNIST 모델 학습 시작 (에포크: {args.epochs}, 학습률: {args.lr})...")

    # 모델 매니저 초기화
    model_manager = FlaxModelManager()
    model_manager.init_model()

    # 학습 및 평가 실행
    state = train_and_evaluate(
        model_manager,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        evaluate_model=args.evaluate,
        subdir=args.subdir,
    )

    print("FLAX MNIST 모델 학습 완료!")


if __name__ == "__main__":
    main()
