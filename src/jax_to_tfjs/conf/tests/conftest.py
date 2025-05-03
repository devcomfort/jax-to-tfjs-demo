"""
pytest 설정 파일

테스트 환경 설정 및 공통 기능을 제공합니다.
"""

import pytest
import sys
import os
from pathlib import Path
import traceback
import logging
from _pytest.runner import CallInfo


@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """
    테스트 실행 전 Python 경로 설정

    프로젝트 루트 디렉토리를 파이썬 경로에 추가하여 모듈 임포트가
    정상적으로 작동하도록 합니다.
    """
    # 프로젝트 루트 디렉토리 찾기
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir

    # src 디렉토리를 찾을 때까지 상위 디렉토리로 이동
    while not (project_root / "src").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:  # 루트에 도달하면 중단
            break

    # 프로젝트 루트를 파이썬 경로에 추가
    sys.path.insert(0, str(project_root))

    yield

    # 테스트 종료 후 경로 제거 (선택 사항)
    if str(project_root) in sys.path:
        sys.path.remove(str(project_root))


# 이 디렉토리에만 적용되는 테스트 경로 설정
def pytest_configure(config):
    # 테스트 경로 설정
    config.addinivalue_line("testpaths", ".")
    config.addinivalue_line("testpaths", "test_path.py")
    config.addinivalue_line("testpaths", "test_logging_config.py")
    config.addinivalue_line("testpaths", "test_config")
    config.addinivalue_line("testpaths", "test_types")

    # 이 디렉토리에서 사용하는 마커 등록
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (deselect with '-m \"not integration\"')",
    )


# 테스트 실패 시 상세 정보 출력
@pytest.hookimpl(trylast=True)
def pytest_runtest_protocol(item, nextitem):
    """테스트 실행 중 예외 발생 시 상세 정보 출력"""
    return None


@pytest.hookimpl(trylast=True)
def pytest_runtest_makereport(item, call):
    """테스트 결과 보고서에 상세 정보 추가"""
    if call.when == "call" and call.excinfo:
        print("\n" + "=" * 80)
        print(f"테스트 실패: {item.nodeid}")
        print("-" * 80)

        # 실패 유형에 따른 추가 정보 출력
        if call.excinfo.type == AssertionError:
            print("실패 원인: 어설션 오류")
            print(f"기대값과 실제값이 다릅니다: {str(call.excinfo.value)}")

            # 테스트 함수 내부 변수 상태 출력 (가능한 경우)
            if hasattr(item, "obj") and hasattr(item.obj, "__self__"):
                test_instance = item.obj.__self__
                if hasattr(test_instance, "_outcome"):
                    # unittest.TestCase 객체의 경우
                    for key, value in test_instance.__dict__.items():
                        if not key.startswith("_") and not callable(value):
                            print(f"변수 {key}: {repr(value)}")

        print("상세 오류 정보:")
        traceback.print_exception(*call.excinfo._excinfo)
        print("=" * 80)


@pytest.fixture
def debug_logger():
    """디버깅용 로거를 제공하는 픽스처"""
    logger = logging.getLogger("debug")

    # 콘솔 핸들러 설정
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # 포맷터 설정
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s"
    )
    handler.setFormatter(formatter)

    # 기존 핸들러 제거 및 새 핸들러 추가
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)

    # 로그 레벨 설정
    logger.setLevel(logging.DEBUG)

    return logger
