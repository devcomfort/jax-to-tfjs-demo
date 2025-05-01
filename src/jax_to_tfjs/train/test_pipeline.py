#!/usr/bin/env python3
"""
JAX/FLAX 체크포인트 저장 회귀 테스트 파이프라인

이 스크립트는 JAX 및 FLAX 학습 모델의 체크포인트 저장 기능을 자동으로 테스트하고
결과를 로깅 및 보고하는 파이프라인을 제공합니다.

실행 방법:
    python -m jax_to_tfjs.train.test_pipeline --model-type=jax --epochs=1
"""

import json
import time
import argparse
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import pytest

# 테스트할 모듈 import
from jax_to_tfjs.train.jax_trainer import JAXTrainer
from jax_to_tfjs.train.flax_trainer import FlaxTrainer
from jax_to_tfjs.checkpoint_utils.jax_checkpointer import JAXCheckpointer
from jax_to_tfjs.models.jax.cnn_model import CNNModel
from jax_to_tfjs.models.flax.model_manager import FlaxModelManager


# ---------- 테스트 유틸리티 클래스 구현 ----------


@pytest.fixture
def epochs():
    """테스트에 사용할 기본 에포크 수"""
    return 1


@pytest.fixture
def logger():
    """테스트에 사용할 로거"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    return logger


@pytest.fixture
def collector(tmp_path):
    """테스트 결과 수집기"""
    return TestResultCollector(tmp_path)


@pytest.fixture
def checkpoint_dir(tmp_path):
    """테스트용 체크포인트 디렉토리"""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


class CheckpointDebugger:
    """체크포인트 디버깅 유틸리티"""

    @staticmethod
    def inspect_checkpoint(checkpoint_dir: Path, step: Optional[int] = None):
        """체크포인트 검사"""
        checkpointer = JAXCheckpointer()

        # 체크포인트 목록 확인
        steps = checkpointer.list_checkpoints(checkpoint_dir)

        if not steps:
            return {
                "error": "체크포인트가 없습니다",
                "directory": str(checkpoint_dir),
                "exists": checkpoint_dir.exists(),
            }

        if step is None:
            step = max(steps)

        try:
            # 체크포인트 로드
            ckpt_data = checkpointer.load(checkpoint_dir, step)

            # 체크포인트 메타데이터
            result = {
                "step": step,
                "available_steps": steps,
                "keys": list(ckpt_data.keys()),
                "structure": {},
            }

            # 파라미터 구조 분석
            if "params" in ckpt_data:
                params = ckpt_data["params"]
                if isinstance(params, dict):
                    result["structure"]["params"] = {
                        k: {
                            "type": type(v).__name__,
                            "shape": v["kernel"].shape
                            if isinstance(v, dict) and "kernel" in v
                            else "unknown",
                        }
                        for k, v in params.items()
                    }

            return result

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}


# 로그 설정
def setup_logging(
    log_dir: Optional[Path] = None,
) -> Tuple[logging.Logger, Optional[Path]]:
    """로깅 설정을 초기화합니다"""
    logger = logging.getLogger("checkpoint_test_pipeline")
    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (옵션)
    log_file = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"checkpoint_test_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, log_file


# 테스트 결과 저장
class TestResultCollector:
    """테스트 결과 수집 및 보고"""

    # pytest가 이 클래스를 테스트 클래스로 수집하지 않도록 설정
    __test__ = False

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(
        self, test_name: str, success: bool, details: Dict[str, Any], duration: float
    ):
        """테스트 결과 추가"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)

    def save_results(self) -> Path:
        """테스트 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"checkpoint_test_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(
                {
                    "results": self.results,
                    "summary": {
                        "total": len(self.results),
                        "succeeded": sum(1 for r in self.results if r["success"]),
                        "failed": sum(1 for r in self.results if not r["success"]),
                        "total_duration": sum(r["duration"] for r in self.results),
                    },
                },
                f,
                indent=2,
            )

        return report_file

    def print_summary(self, logger: logging.Logger):
        """테스트 결과 요약 출력"""
        succeeded = sum(1 for r in self.results if r["success"])
        failed = sum(1 for r in self.results if not r["success"])
        total = len(self.results)

        logger.info("=" * 50)
        logger.info(f"테스트 결과 요약: {succeeded}/{total} 성공 ({failed} 실패)")
        logger.info("-" * 50)

        for result in self.results:
            status = "성공" if result["success"] else "실패"
            logger.info(f"{result['test_name']}: {status} ({result['duration']:.2f}초)")

            if not result["success"] and "error" in result["details"]:
                logger.info(f"  오류: {result['details']['error']}")

        logger.info("=" * 50)


# JAX 모델 테스트
def test_jax_checkpoint(epochs, collector, logger, checkpoint_dir):
    """JAX 모델의 체크포인트 저장 및 로드 테스트"""
    start_time = time.time()
    test_success = False
    details = {}

    try:
        logger.info("JAX 체크포인트 테스트 시작")

        # 실제 CNNModel 사용
        model = CNNModel()
        model.init_params()
        trainer = JAXTrainer(model)

        # 체크포인트 저장
        jax_checkpoint_dir = checkpoint_dir / "jax"
        jax_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 학습 실행
        logger.info(f"JAX 모델 학습 - {epochs} 에포크")
        trainer.train(num_epochs=epochs, subdir=str(jax_checkpoint_dir))

        # 체크포인트 존재 확인
        logger.info("체크포인트 검증")

        # 체크포인트 디렉토리 확인
        if not jax_checkpoint_dir.exists():
            raise ValueError(
                f"체크포인트 디렉토리가 존재하지 않음: {jax_checkpoint_dir}"
            )

        # 체크포인트 파일 확인
        checkpointer = JAXCheckpointer()
        steps = checkpointer.list_checkpoints(jax_checkpoint_dir)

        if not steps:
            raise ValueError("체크포인트가 생성되지 않음")

        # 가장 최근 체크포인트 확인
        last_step = max(steps)
        checkpoint_path = jax_checkpoint_dir / str(last_step)

        if not checkpoint_path.exists():
            raise ValueError(f"체크포인트 디렉토리가 존재하지 않음: {checkpoint_path}")

        logger.info(f"체크포인트 디렉토리 확인 완료: {checkpoint_path}")

        # 상세 메트릭 파일 존재 확인
        metrics_path = jax_checkpoint_dir / "evaluation" / "metrics.json"
        if metrics_path.exists():
            logger.info(f"평가 메트릭 파일 존재: {metrics_path}")
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                logger.info(f"메트릭: accuracy={metrics.get('accuracy', 'N/A')}")
                details["metrics"] = metrics
        else:
            logger.warning(f"평가 메트릭 파일이 없음: {metrics_path}")

        # 테스트 성공
        test_success = True
        details["checkpoint_info"] = {
            "directory": str(jax_checkpoint_dir),
            "steps": steps,
            "last_step": last_step,
        }
        logger.info("JAX 체크포인트 테스트 성공")

    except Exception as e:
        logger.error(f"JAX 체크포인트 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        details["error"] = str(e)
        details["traceback"] = traceback.format_exc()

    duration = time.time() - start_time
    collector.add_result("jax_checkpoint", test_success, details, duration)

    # True를 반환하는 대신 assert 사용
    assert test_success, "JAX 체크포인트 테스트 실패"


# FLAX 모델 테스트
def test_flax_checkpoint(epochs, collector, logger, checkpoint_dir):
    """FLAX 모델의 체크포인트 저장 및 로드 테스트"""
    start_time = time.time()
    test_success = False
    details = {}

    try:
        logger.info("FLAX 체크포인트 테스트 시작")

        # 실제 FlaxModelManager 사용
        model_manager = FlaxModelManager()
        model_manager.init_model()
        trainer = FlaxTrainer(model_manager)

        # 체크포인트 저장
        flax_checkpoint_dir = checkpoint_dir / "flax"
        flax_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 학습 실행
        logger.info(f"FLAX 모델 학습 - {epochs} 에포크")
        trainer.train(num_epochs=epochs, subdir=str(flax_checkpoint_dir))

        # 체크포인트 디렉토리 확인
        if not flax_checkpoint_dir.exists():
            raise ValueError(
                f"체크포인트 디렉토리가 존재하지 않음: {flax_checkpoint_dir}"
            )

        # 체크포인트 파일 확인 - Flax는 에포크 번호로 디렉토리 생성
        checkpoint_dirs = [
            d for d in flax_checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
        ]

        if not checkpoint_dirs:
            raise ValueError(f"체크포인트 디렉토리가 없음: {flax_checkpoint_dir}")

        # 가장 최근 체크포인트 확인
        last_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))

        if not last_checkpoint.exists():
            raise ValueError(
                f"최신 체크포인트 디렉토리가 존재하지 않음: {last_checkpoint}"
            )

        logger.info(f"체크포인트 디렉토리 확인 완료: {last_checkpoint}")

        # 상세 메트릭 파일 존재 확인
        metrics_path = flax_checkpoint_dir / "evaluation" / "metrics.json"
        if metrics_path.exists():
            logger.info(f"평가 메트릭 파일 존재: {metrics_path}")
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                logger.info(f"메트릭: accuracy={metrics.get('accuracy', 'N/A')}")
                details["metrics"] = metrics
        else:
            logger.warning(f"평가 메트릭 파일이 없음: {metrics_path}")

        # 테스트 성공
        test_success = True
        details["checkpoint_info"] = {
            "directory": str(flax_checkpoint_dir),
            "last_checkpoint": str(last_checkpoint),
        }
        logger.info("FLAX 체크포인트 테스트 성공")

    except Exception as e:
        logger.error(f"FLAX 체크포인트 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        details["error"] = str(e)
        details["traceback"] = traceback.format_exc()

    duration = time.time() - start_time
    collector.add_result("flax_checkpoint", test_success, details, duration)

    # True를 반환하는 대신 assert 사용
    assert test_success, "FLAX 체크포인트 테스트 실패"


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="체크포인트 테스트 파이프라인")
    parser.add_argument(
        "--model-type",
        choices=["jax", "flax", "all"],
        default="all",
        help="테스트할 모델 유형 (기본값: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="학습 에포크 수 (기본값: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/checkpoint_tests",
        help="테스트 결과 저장 경로 (기본값: logs/checkpoint_tests)",
    )
    args = parser.parse_args()

    # 로깅 설정
    output_dir = Path(args.output_dir)
    logger, log_file = setup_logging(output_dir)
    logger.info(
        f"체크포인트 테스트 시작 - 모델: {args.model_type}, 에포크: {args.epochs}"
    )

    # 결과 수집기 초기화
    collector = TestResultCollector(output_dir)

    # 임시 체크포인트 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # 테스트 실행
        if args.model_type in ["jax", "all"]:
            test_jax_checkpoint(args.epochs, collector, logger, checkpoint_dir)

        if args.model_type in ["flax", "all"]:
            test_flax_checkpoint(args.epochs, collector, logger, checkpoint_dir)

    # 결과 저장 및 출력
    results_path = collector.save_results()
    logger.info(f"테스트 결과 저장 완료: {results_path}")
    collector.print_summary(logger)

    if log_file:
        logger.info(f"로그 파일: {log_file}")


if __name__ == "__main__":
    main()
