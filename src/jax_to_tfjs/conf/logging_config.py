"""
로깅 시스템 설정 모듈

구조화된 로깅 시스템과 다양한 로깅 레벨을 지원합니다.
학습 진행 상황, 모델 파라미터, 성능 지표 등을 체계적으로 기록합니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union


# 로깅 수준 정의
VERBOSE = 5  # VERBOSE 레벨 (DEBUG보다 더 상세한 로깅을 위한 커스텀 레벨)
logging.addLevelName(VERBOSE, "VERBOSE")


def verbose(self, message, *args, **kwargs):
    """VERBOSE 레벨 로깅 메서드"""
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)


# Logger 클래스에 verbose 메서드 추가
logging.Logger.verbose = verbose  # type: ignore


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    json_format: bool = False,
    log_dir: Optional[Union[str, Path]] = None,
):
    """
    로깅 시스템 설정

    Args:
        log_file: 로그 파일 경로 (기본값: None, 자동 생성)
        console_level: 콘솔 로깅 레벨 (기본값: INFO)
        file_level: 파일 로깅 레벨 (기본값: DEBUG)
        json_format: JSON 형식 로깅 사용 여부 (기본값: False)
        log_dir: 로그 디렉토리 경로 (기본값: None, 'logs' 디렉토리 사용)
    """
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)  # 모든 핸들러가 자체적으로 필터링하도록 설정

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    # 파일 핸들러 설정 (지정된 경우)
    if log_file is None and log_dir is not None:
        # 로그 디렉토리가 지정된 경우 자동 생성
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 타임스탬프로 로그 파일 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"log_{timestamp}.log"

    # 포매터 설정
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 추가 (지정된 경우)
    if log_file:
        # 디렉토리가 없으면 생성
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


class JsonFormatter(logging.Formatter):
    """JSON 형식으로 로그를 포맷팅하는 포매터"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # 예외 정보 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 추가 데이터가 있으면 포함
        if hasattr(record, "data"):
            data = getattr(record, "data", None)
            if data and isinstance(data, dict):
                log_data.update(data)

        return json.dumps(log_data)


class LoggerAdapter(logging.LoggerAdapter):
    """추가 데이터를 로깅할 수 있도록 확장된 로거 어댑터"""

    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        # 추가 데이터가 있으면 기록
        if "extra" in kwargs and kwargs["extra"] is not None:
            # 기존 extra 딕셔너리가 있으면 업데이트, 없으면 새로 생성
            if "extra" not in kwargs:
                kwargs["extra"] = {}

            # 로그 레코드에 data 속성 추가
            if "data" not in kwargs["extra"]:
                kwargs["extra"]["data"] = {}

            # extra 딕셔너리의 내용을 data에 추가
            if isinstance(kwargs["extra"]["data"], dict):
                kwargs["extra"]["data"].update(self.extra)

        return msg, kwargs

    def verbose(self, msg, *args, **kwargs):
        """VERBOSE 레벨 로깅"""
        if self.isEnabledFor(VERBOSE):
            self.log(VERBOSE, msg, *args, **kwargs)


def get_logger(name: str, extra: Optional[Dict[str, Any]] = None):
    """
    로거 인스턴스 가져오기

    Args:
        name: 로거 이름
        extra: 추가 로깅 데이터

    Returns:
        로거 어댑터 인스턴스
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, extra or {})


def log_metrics(
    logger,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    level: int = logging.INFO,
):
    """
    성능 지표 로깅

    Args:
        logger: 로거 인스턴스
        metrics: 로깅할 지표
        step: 현재 스텝 (선택 사항)
        epoch: 현재 에포크 (선택 사항)
        level: 로깅 레벨 (기본값: INFO)
    """
    # 메트릭 문자열 생성
    metrics_str_parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            metrics_str_parts.append(f"{name}: {value:.6f}")
        else:
            metrics_str_parts.append(f"{name}: {value}")

    metrics_str = ", ".join(metrics_str_parts)

    # 로깅 메시지 생성
    message_parts = []
    if epoch is not None:
        message_parts.append(f"Epoch: {epoch}")
    if step is not None:
        message_parts.append(f"Step: {step}")
    message_parts.append(f"Metrics: {metrics_str}")

    message = " - ".join(message_parts)

    # 추가 데이터를 포함하여 로깅
    extra_data: Dict[str, Any] = {"metrics": metrics}
    if step is not None:
        extra_data["step"] = step
    if epoch is not None:
        extra_data["epoch"] = epoch

    logger.log(level, message, extra={"data": extra_data})


# 고급 로깅 기능은 별도로 정의
def setup_json_logging(log_file=None, log_dir="logs"):
    """JSON 형식 로깅 설정"""
    return setup_logging(log_file=log_file, log_dir=log_dir, json_format=True)


__all__ = [
    "VERBOSE",
    "setup_logging",
    "setup_json_logging",
    "get_logger",
    "log_metrics",
    "LoggerAdapter",
    "JsonFormatter",
]
