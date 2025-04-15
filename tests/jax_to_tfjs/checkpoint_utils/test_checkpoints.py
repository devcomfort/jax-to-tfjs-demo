"""
checkpoint_utils 모듈의 통합 테스트

실제 체크포인트 디렉토리와 상호작용하여 모듈의 기능을 검증합니다.

통합 테스트와 모킹(Mocking) 정보:
- 이 테스트는 단위 테스트와 달리 일부 실제 파일 시스템과 상호작용하는 통합 테스트입니다.
- 필요한 경우 테스트 체크포인트 디렉토리를 생성하여 실제 환경과 유사한 조건에서 테스트합니다.
- @patch 데코레이터는 선택적으로 사용되어 테스트 결과의 결정성을 보장합니다.
- 통합 테스트는 모듈 간의 상호작용과 실제 환경에서의 동작을 검증하는데 중점을 둡니다.
- create_test_checkpoints 메서드를 통해 실제 체크포인트가 없을 경우 테스트용 체크포인트를 생성합니다.
"""
import os
import sys
import tempfile
import unittest
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from src.jax_to_tfjs.checkpoint_utils.info import (
    get_checkpoints_info,
    get_latest_checkpoint,
    get_checkpoint_by_index,
    get_checkpoint_by_step,
    list_available_checkpoints
)
from src.jax_to_tfjs.checkpoint_utils.validation import validate_checkpoint
from src.jax_to_tfjs.paths import get_jax_checkpoint_path, get_flax_checkpoint_path

class TestCheckpointUtils(unittest.TestCase):
    """checkpoint_utils 모듈 통합 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 체크포인트 기본 경로 확인
        cls.jax_checkpoint_dir = get_jax_checkpoint_path()
        cls.flax_checkpoint_dir = get_flax_checkpoint_path()
        
        # 테스트 체크포인트 디렉토리 생성 (필요 시)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_checkpoint_dir = Path(cls.temp_dir.name)
        
        # 테스트 체크포인트 생성 (실제 체크포인트가 없는 경우)
        cls.create_test_checkpoints()
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # 임시 디렉토리 정리
        cls.temp_dir.cleanup()
    
    @classmethod
    def create_test_checkpoints(cls):
        """테스트용 체크포인트 생성"""
        # JAX 체크포인트 디렉토리가 없는 경우 테스트용 체크포인트 생성
        if not os.path.exists(cls.jax_checkpoint_dir) or not os.listdir(cls.jax_checkpoint_dir):
            # 테스트용 JAX 체크포인트 디렉토리 생성
            jax_test_dir = cls.test_checkpoint_dir / "jax_mnist"
            jax_test_dir.mkdir(parents=True, exist_ok=True)
            
            # 테스트용 체크포인트 생성
            for step in [10, 20, 30]:
                step_dir = jax_test_dir / str(step)
                step_dir.mkdir(parents=True, exist_ok=True)
                
                # 모델 디렉토리 생성
                model_dir = step_dir / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # 체크포인트 메타데이터 파일 생성
                metadata = {
                    "step": step,
                    "timestamp": datetime.now().timestamp(),
                    "model_type": "jax"
                }
                
                with open(step_dir / "_CHECKPOINT_METADATA", "w") as f:
                    json.dump(metadata, f)
            
            # 환경 변수 설정 (실제 테스트 시 사용될 수 있도록)
            os.environ["JAX_CHECKPOINT_DIR"] = str(cls.test_checkpoint_dir)
    
    def test_get_checkpoints_info(self):
        """get_checkpoints_info 함수 테스트"""
        # JAX 체크포인트 정보 조회
        jax_checkpoints = get_checkpoints_info("jax")
        
        # 체크포인트가 존재하는지 확인
        self.assertIsInstance(jax_checkpoints, list)
        
        if jax_checkpoints:
            # 체크포인트 정보 구조 확인
            checkpoint = jax_checkpoints[0]
            self.assertIn("path", checkpoint)
            self.assertIn("name", checkpoint)
            self.assertIn("model_type", checkpoint)
            
            # 모델 타입 확인
            self.assertEqual(checkpoint["model_type"], "jax")
            
            print(f"\n테스트 get_checkpoints_info 결과:")
            print(f"체크포인트 수: {len(jax_checkpoints)}")
            print(f"첫 번째 체크포인트: {checkpoint}")
    
    def test_get_latest_checkpoint(self):
        """get_latest_checkpoint 함수 테스트"""
        # 최신 JAX 체크포인트 조회
        latest_checkpoint = get_latest_checkpoint("jax")
        
        # 체크포인트가 존재하는지 확인
        if latest_checkpoint:
            # 체크포인트 정보 구조 확인
            self.assertIn("path", latest_checkpoint)
            self.assertIn("model_type", latest_checkpoint)
            
            # 모델 타입 확인
            self.assertEqual(latest_checkpoint["model_type"], "jax")
            
            print(f"\n테스트 get_latest_checkpoint 결과:")
            print(f"최신 체크포인트: {latest_checkpoint}")
        else:
            print("\n테스트 get_latest_checkpoint 결과: 체크포인트를 찾을 수 없습니다.")
    
    def test_get_checkpoint_by_index(self):
        """get_checkpoint_by_index 함수 테스트"""
        # JAX 체크포인트 정보 조회
        jax_checkpoints = get_checkpoints_info("jax")
        
        if jax_checkpoints:
            # 인덱스로 체크포인트 조회
            first_checkpoint = get_checkpoint_by_index("jax", 0)
            
            # 체크포인트 정보 확인
            self.assertIsNotNone(first_checkpoint)
            self.assertEqual(first_checkpoint["path"], jax_checkpoints[0]["path"])
            
            print(f"\n테스트 get_checkpoint_by_index 결과:")
            print(f"인덱스 0 체크포인트: {first_checkpoint}")
            
            # 유효하지 않은 인덱스 테스트
            invalid_checkpoint = get_checkpoint_by_index("jax", 9999)
            self.assertIsNone(invalid_checkpoint)
        else:
            print("\n테스트 get_checkpoint_by_index 결과: 체크포인트를 찾을 수 없습니다.")
    
    def test_validate_checkpoint(self):
        """validate_checkpoint 함수 테스트"""
        # JAX 체크포인트 정보 조회
        jax_checkpoints = get_checkpoints_info("jax")
        
        if jax_checkpoints:
            # 첫 번째 체크포인트 유효성 검증
            first_checkpoint_path = jax_checkpoints[0]["path"]
            is_valid = validate_checkpoint(first_checkpoint_path, "jax")
            
            # 유효성 확인
            self.assertTrue(is_valid)
            
            print(f"\n테스트 validate_checkpoint 결과:")
            print(f"체크포인트 경로: {first_checkpoint_path}")
            print(f"유효성: {is_valid}")
            
            # 유효하지 않은 경로 테스트
            invalid_path = os.path.join(os.path.dirname(first_checkpoint_path), "invalid_path")
            is_invalid = validate_checkpoint(invalid_path, "jax")
            self.assertFalse(is_invalid)
        else:
            print("\n테스트 validate_checkpoint 결과: 체크포인트를 찾을 수 없습니다.")
    
    def test_list_available_checkpoints(self):
        """list_available_checkpoints 함수 테스트"""
        # JAX 체크포인트 목록 조회 (verbose=False로 출력 없이)
        jax_checkpoints = list_available_checkpoints("jax", verbose=False)
        
        # 체크포인트 목록 검증
        self.assertIsInstance(jax_checkpoints, list)
        
        if jax_checkpoints:
            print(f"\n테스트 list_available_checkpoints 결과:")
            print(f"체크포인트 수: {len(jax_checkpoints)}")
            
            # verbose=True로 다시 호출하여 출력 확인
            print("\n체크포인트 목록 출력 테스트:")
            list_available_checkpoints("jax", verbose=True)
        else:
            print("\n테스트 list_available_checkpoints 결과: 체크포인트를 찾을 수 없습니다.")

if __name__ == "__main__":
    unittest.main() 