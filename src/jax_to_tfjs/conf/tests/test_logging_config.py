"""
로깅 시스템 설정 모듈 테스트

로깅 설정 및 유틸리티 함수를 테스트합니다.
"""

import unittest
import os
import tempfile
import logging
import json
import io
import sys
from pathlib import Path

import pytest

from jax_to_tfjs.conf.logging_config import (
    setup_logging,
    JsonFormatter,
    get_logger,
    log_metrics,
    setup_json_logging,
    VERBOSE,
)


class TestLoggingConfig(unittest.TestCase):
    """로깅 설정 테스트"""

    def setUp(self):
        """각 테스트 전에 실행"""
        # 기존 로거 핸들러 백업 및 제거
        self.root_logger = logging.getLogger()
        self.original_handlers = self.root_logger.handlers.copy()
        self.root_logger.handlers.clear()

        # 로그 캡처를 위한 스트림
        self.log_stream = io.StringIO()

        # 임시 로그 디렉토리
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """각 테스트 후에 실행"""
        # 로거 핸들러 복원
        self.root_logger.handlers.clear()
        for handler in self.original_handlers:
            self.root_logger.addHandler(handler)

        # 캡처 스트림 닫기
        self.log_stream.close()

        # 임시 파일 정리 (있는 경우)
        for handler in self.root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def test_setup_logging_basic(self):
        """기본 로깅 설정 테스트"""
        logger = setup_logging(console_level=logging.INFO)

        # 로거 핸들러 확인
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        self.assertEqual(logger.handlers[0].level, logging.INFO)

    def test_setup_logging_with_file(self):
        """파일 출력 로깅 설정 테스트"""
        # 임시 로그 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
            log_path = temp_file.name

        try:
            logger = setup_logging(
                log_file=log_path,
                console_level=logging.WARNING,
                file_level=logging.DEBUG,
            )

            # 핸들러 확인
            self.assertEqual(len(logger.handlers), 2)

            # 콘솔 핸들러
            console_handler = next(
                h
                for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            )
            self.assertEqual(console_handler.level, logging.WARNING)

            # 파일 핸들러
            file_handler = next(
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            )
            self.assertEqual(file_handler.level, logging.DEBUG)
            self.assertEqual(file_handler.baseFilename, log_path)

            # 로그 기록 테스트
            test_logger = logging.getLogger("test_file_logger")
            test_logger.warning("Test warning message")
            test_logger.debug("Test debug message")

            # 파일에 로그가 기록되었는지 확인
            with open(log_path, "r") as f:
                log_content = f.read()
                self.assertIn("Test warning message", log_content)
                self.assertIn("Test debug message", log_content)
        finally:
            # 임시 파일 정리
            os.unlink(log_path)

    def test_json_formatter(self):
        """JSON 포맷터 테스트"""
        formatter = JsonFormatter()

        # 로그 레코드 생성
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # 데이터 속성 추가
        record.data = {"user_id": 123, "action": "login"}

        # 포맷 변환
        formatted = formatter.format(record)

        # JSON 파싱 확인
        log_data = json.loads(formatted)

        # 필수 필드 확인
        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["name"], "test_logger")
        self.assertEqual(log_data["message"], "Test message")
        self.assertIn("timestamp", log_data)

        # 추가 데이터 확인
        self.assertEqual(log_data["user_id"], 123)
        self.assertEqual(log_data["action"], "login")

    @pytest.mark.debug_logging
    def test_log_metrics(self, debug_logger=None):
        """지표 로깅 테스트"""
        print("\n[디버깅] test_log_metrics 시작")

        # 직접 테스트 로거 생성
        test_stream = io.StringIO()
        test_handler = logging.StreamHandler(test_stream)
        test_logger = logging.getLogger("direct_test_logger")

        # 기존 핸들러 제거 및 새 핸들러 설정
        for h in test_logger.handlers:
            test_logger.removeHandler(h)

        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.INFO)
        print(f"[디버깅] 테스트 로거 레벨: {logging.getLevelName(test_logger.level)}")

        # 지표 로깅
        metrics = {"loss": 0.123456, "accuracy": 0.98, "learning_rate": 0.001}

        print("[디버깅] log_metrics 함수 직접 호출")
        log_metrics(test_logger, metrics, step=100, epoch=5)

        # 로그 출력 확인
        log_output = test_stream.getvalue()
        print(f"[디버깅] 직접 테스트 로그 출력: {log_output!r}")

        # 실제 출력 형식에 맞게 검증
        # 실제 출력 예: 'Epoch: 5 - Step: 100 - Metrics: loss: 0.123456, ...'
        self.assertIn("Epoch: 5", log_output)
        self.assertIn("Step: 100", log_output)
        self.assertIn("loss: 0.123456", log_output)
        self.assertIn("accuracy: 0.98", log_output)
        self.assertIn("learning_rate: 0.001", log_output)

    def test_verbose_level(self):
        """VERBOSE 레벨 테스트"""
        # 출력 캡처를 위한 핸들러 설정
        handler = logging.StreamHandler(self.log_stream)
        handler.setLevel(VERBOSE)
        self.root_logger.addHandler(handler)

        # 로거 가져오기
        logger = get_logger("verbose_test")

        # 다양한 레벨로 로깅
        logger.verbose("Verbose message")
        logger.debug("Debug message")
        logger.info("Info message")

        # 로그 출력 확인
        log_output = self.log_stream.getvalue()
        self.assertIn("Verbose message", log_output)
        self.assertIn("Debug message", log_output)
        self.assertIn("Info message", log_output)


if __name__ == "__main__":
    unittest.main()
