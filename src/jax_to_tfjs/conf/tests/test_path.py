"""
경로 관리 모듈 테스트

Path 클래스 및 경로 관련 기능을 테스트합니다.
"""

import unittest
from pathlib import Path as PathLib
import os

from jax_to_tfjs.conf.path import Path, get_path


class TestPath(unittest.TestCase):
    """Path 클래스 테스트"""

    def test_path_singleton(self):
        """싱글톤 패턴 테스트"""
        path1 = get_path()
        path2 = get_path()
        self.assertIs(path1, path2, "get_path는 싱글톤 인스턴스를 반환해야 합니다")

    def test_path_structure(self):
        """경로 구조 테스트"""
        path = Path()

        # 기본 디렉토리명 정의
        checkpoint_dirname = "checkpoints"
        jax_dirname = "jax"
        flax_dirname = "flax"
        onnx_dirname = "onnx"
        tensorflow_dirname = "tensorflow"
        tfjs_dirname = "tfjs"

        # 기본 경로 확인
        self.assertIsInstance(path.project_root, PathLib)
        self.assertIsInstance(path.checkpoint_dir, PathLib)
        self.assertIsInstance(path.jax_checkpoint_dir, PathLib)
        self.assertIsInstance(path.flax_checkpoint_dir, PathLib)
        self.assertIsInstance(path.onnx_dir, PathLib)
        self.assertIsInstance(path.tensorflow_dir, PathLib)
        self.assertIsInstance(path.tfjs_dir, PathLib)

        # 경로 관계 확인
        self.assertEqual(path.checkpoint_dir, path.project_root / checkpoint_dirname)
        self.assertEqual(path.jax_checkpoint_dir, path.checkpoint_dir / jax_dirname)
        self.assertEqual(path.flax_checkpoint_dir, path.checkpoint_dir / flax_dirname)
        self.assertEqual(path.onnx_dir, path.project_root / onnx_dirname)
        self.assertEqual(path.tensorflow_dir, path.project_root / tensorflow_dirname)
        self.assertEqual(path.tfjs_dir, path.project_root / tfjs_dirname)

    def test_get_jax_checkpoint_path(self):
        """JAX 체크포인트 경로 테스트"""
        path = Path()
        test_subdir = "model_v1"

        # 기본 경로
        jax_path = path.get_jax_checkpoint_path()
        self.assertEqual(jax_path, path.jax_checkpoint_dir)

        # 하위 디렉토리 지정
        jax_subdir_path = path.get_jax_checkpoint_path(test_subdir)
        self.assertEqual(jax_subdir_path, path.jax_checkpoint_dir / test_subdir)

    def test_get_flax_checkpoint_path(self):
        """Flax 체크포인트 경로 테스트"""
        path = Path()
        test_subdir = "model_v1"

        # 기본 경로
        flax_path = path.get_flax_checkpoint_path()
        self.assertEqual(flax_path, path.flax_checkpoint_dir)

        # 하위 디렉토리 지정
        flax_subdir_path = path.get_flax_checkpoint_path(test_subdir)
        self.assertEqual(flax_subdir_path, path.flax_checkpoint_dir / test_subdir)

    def test_get_onnx_path(self):
        """ONNX 모델 경로 테스트"""
        path = Path()
        test_model_name = "mnist_model"

        # 기본 경로 (디렉토리)
        onnx_path = path.get_onnx_path()
        self.assertEqual(onnx_path, path.onnx_dir)

        # 모델 이름 지정 (파일 경로)
        onnx_model_path = path.get_onnx_path(test_model_name)
        self.assertEqual(onnx_model_path, path.onnx_dir / f"{test_model_name}.onnx")

    def test_get_tensorflow_path(self):
        """TensorFlow 모델 경로 테스트"""
        path = Path()
        test_model_name = "keras_model"

        # 기본 경로 (디렉토리)
        tf_path = path.get_tensorflow_path()
        self.assertEqual(tf_path, path.tensorflow_dir)

        # 모델 이름 지정 (서브디렉토리)
        tf_model_path = path.get_tensorflow_path(test_model_name)
        self.assertEqual(tf_model_path, path.tensorflow_dir / test_model_name)

    def test_get_tfjs_path(self):
        """TensorFlow.js 모델 경로 테스트"""
        path = Path()
        test_model_name = "web_model"

        # 기본 경로 (디렉토리)
        tfjs_path = path.get_tfjs_path()
        self.assertEqual(tfjs_path, path.tfjs_dir)

        # 모델 이름 지정 (서브디렉토리)
        tfjs_model_path = path.get_tfjs_path(test_model_name)
        self.assertEqual(tfjs_model_path, path.tfjs_dir / test_model_name)


if __name__ == "__main__":
    unittest.main()
