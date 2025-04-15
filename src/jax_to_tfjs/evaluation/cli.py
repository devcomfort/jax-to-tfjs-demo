"""
JAX/Flax 모델 평가를 위한 명령행 인터페이스

모델 평가 명령어를 처리하고 평가 기능을 실행하는 CLI 모듈입니다.
"""
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import jax

from jax_to_tfjs.paths import (
    get_jax_checkpoint_path, 
    get_flax_checkpoint_path, 
    get_results_path
)
from jax_to_tfjs.checkpoint_utils import (
    get_latest_checkpoint,
    get_checkpoints_info,
    list_available_checkpoints,
    validate_checkpoint,
    load_checkpoint_by_step
)
from jax_to_tfjs.models.jax_mnist_cnn import (
    load_checkpoint as load_jax_checkpoint
)
from jax_to_tfjs.models.flax_mnist_cnn import (
    load_checkpoint as load_flax_checkpoint
)
from jax_to_tfjs.evaluation import (
    load_mnist_test,
    evaluate_jax_model,
    evaluate_flax_model,
    visualize_image_predictions,
    visualize_confusion_matrix,
    visualize_roc_curve,
    visualize_precision_recall_curve,
    visualize_metrics_by_class,
    calculate_confusion_matrix,
    calculate_metrics,
    save_metrics_to_json,
    print_metrics_table
)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """
    명령행 인자 파서를 생성합니다.
    
    반환값:
        argparse.ArgumentParser: 설정된 인자 파서
    """
    parser = argparse.ArgumentParser(
        description='JAX/Flax MNIST 모델 평가 도구',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 기본 인자
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['jax', 'flax'], 
        help='평가할 모델 유형'
    )
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        help='평가할 체크포인트 경로'
    )
    
    parser.add_argument(
        '--step', 
        type=int, 
        help='평가할 체크포인트 스텝 번호'
    )
    
    parser.add_argument(
        '--subdir', 
        type=str, 
        default=None, 
        help='체크포인트 하위 디렉토리'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true', 
        help='시각화 활성화'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true', 
        help='대화형 모드 활성화'
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true', 
        help='체크포인트 비교 모드 활성화'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results', 
        help='결과 저장 디렉토리'
    )
    
    return parser

def interactive_mode(args: argparse.Namespace) -> int:
    """
    대화형 모드에서 평가를 실행합니다.
    
    인자:
        args: 명령행 인자
        
    반환값:
        int: 종료 코드
    """
    logger.info("대화형 모드를 시작합니다...")
    
    # 모델 타입 선택 강제
    model_type = args.model
    if not model_type:
        model_options = ['jax', 'flax']
        print("\n평가할 모델 유형을 선택하세요:")
        for i, option in enumerate(model_options):
            print(f"  {i+1}. {option}")
            
        while True:
            try:
                choice = int(input("\n모델 번호 입력: "))
                if 1 <= choice <= len(model_options):
                    model_type = model_options[choice-1]
                    args.model = model_type  # args 업데이트
                    break
                else:
                    print(f"1에서 {len(model_options)} 사이의 번호를 입력하세요.")
            except ValueError:
                print("유효한 숫자를 입력하세요.")
    
    # 모델 타입이 설정되었는지 확인
    if not model_type:
        logger.error("모델 타입을 지정해야 합니다. --model 옵션 또는 대화형 모드에서 선택하세요.")
        return 1
    
    logger.info(f"선택된 모델 타입: {model_type}")
    
    # 체크포인트 선택
    checkpoint_path = args.checkpoint
    if not checkpoint_path and not args.step:
        # 체크포인트 정보 가져오기
        print(f"\n{model_type} 체크포인트 검색 중...")
        checkpoints = list_available_checkpoints(model_type, verbose=True)
        
        if not checkpoints:
            print(f"오류: '{model_type}' 모델 타입에 대한 체크포인트를 찾을 수 없습니다.")
            return 1
        
        # 체크포인트 선택
        while True:
            try:
                choice = int(input("\n체크포인트 번호 입력 (최신 체크포인트는 0): "))
                if choice == 0:
                    # 최신 체크포인트 선택
                    ckpt = get_latest_checkpoint(model_type, args.subdir)
                    if not ckpt:
                        print("최신 체크포인트를 찾지 못했습니다.")
                        continue
                    checkpoint_path = ckpt['path']
                    break
                elif 1 <= choice <= len(checkpoints):
                    checkpoint_path = checkpoints[choice-1]['path']
                    break
                else:
                    print(f"0에서 {len(checkpoints)} 사이의 번호를 입력하세요.")
            except ValueError:
                print("유효한 숫자를 입력하세요.")
    
    # 선택된 체크포인트로 args 업데이트
    if checkpoint_path:
        args.checkpoint = checkpoint_path
        logger.info(f"선택된 체크포인트: {checkpoint_path}")
    
    # 평가 실행
    return evaluate_model(args)

def compare_checkpoints(args: argparse.Namespace) -> int:
    """
    여러 체크포인트를 비교 평가합니다.
    
    인자:
        args: 명령행 인자
        
    반환값:
        int: 종료 코드
    """
    logger.info("체크포인트 비교 모드를 시작합니다...")
    
    # 모델 타입 선택 강제
    model_type = args.model
    if not model_type:
        model_options = ['jax', 'flax']
        print("\n평가할 모델 유형을 선택하세요:")
        for i, option in enumerate(model_options):
            print(f"  {i+1}. {option}")
            
        while True:
            try:
                choice = int(input("\n모델 번호 입력: "))
                if 1 <= choice <= len(model_options):
                    model_type = model_options[choice-1]
                    args.model = model_type  # args 업데이트
                    break
                else:
                    print(f"1에서 {len(model_options)} 사이의 번호를 입력하세요.")
            except ValueError:
                print("유효한 숫자를 입력하세요.")
    
    # 체크포인트 정보 가져오기
    checkpoints = get_checkpoints_info(model_type, args.subdir)
    
    if not checkpoints:
        # 하위 디렉토리 검색
        if model_type == 'jax':
            base_dir = get_jax_checkpoint_path(args.subdir)
        else:
            base_dir = get_flax_checkpoint_path(args.subdir)
        
        try:
            # 하위 디렉토리 목록 가져오기
            subdirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))]
            
            # 각 하위 디렉토리에서 체크포인트 검색
            for subdir in subdirs:
                subdir_checkpoints = get_checkpoints_info(model_type, subdir)
                if subdir_checkpoints:
                    for ckpt in subdir_checkpoints:
                        ckpt["subdir"] = subdir
                    checkpoints.extend(subdir_checkpoints)
        except (FileNotFoundError, PermissionError) as e:
            print(f"하위 디렉토리 검색 중 오류: {e}")
    
    if not checkpoints:
        print(f"오류: '{model_type}' 모델 타입에 대한 체크포인트를 찾을 수 없습니다.")
        return 1
    
    # 체크포인트 선택을 위한 목록 표시
    print("\n사용 가능한 체크포인트:")
    for i, ckpt in enumerate(checkpoints):
        step_info = f", 스텝: {ckpt['step']}" if "step" in ckpt else ""
        time_info = f", 시간: {ckpt.get('datetime', '')}" if "datetime" in ckpt and ckpt.get('datetime') else ""
        subdir_info = f", 하위디렉토리: {ckpt.get('subdir', '')}" if "subdir" in ckpt and ckpt.get('subdir') else ""
        print(f"  {i+1}. {ckpt['name']}{step_info}{time_info}{subdir_info}")
    
    # 비교할 체크포인트 선택
    selected_indices = []
    print("\n비교할 체크포인트를 선택하세요 (쉼표로 구분된 번호 입력, 또는 'all'로 모두 선택):")
    selection = input().strip()
    
    if selection.lower() == 'all':
        selected_indices = list(range(len(checkpoints)))
    else:
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
            # 인덱스 유효성 검사
            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(checkpoints)]
            if not valid_indices:
                print("유효한 체크포인트를 선택하지 않았습니다.")
                return 1
            selected_indices = valid_indices
        except ValueError:
            print("잘못된 선택 형식입니다. 쉼표로 구분된 번호를 입력하세요.")
            return 1
    
    # 선택된 체크포인트 비교
    compare_results = []
    
    # 테스트 데이터 한 번만 로드
    print("\nMNIST 테스트 데이터셋 로드 중...")
    images, labels = load_mnist_test()
    
    for idx in selected_indices:
        ckpt = checkpoints[idx]
        print(f"\n체크포인트 평가 중 {idx+1}/{len(selected_indices)}: {ckpt['name']}")
        
        # 체크포인트 경로 가져오기
        checkpoint_path = ckpt['path']
        
        # 모델 타입에 따른 평가
        if model_type == 'jax':
            # JAX 모델 로드
            try:
                model = load_jax_checkpoint(checkpoint_path)
                metrics, predicted_labels, predicted_probs = evaluate_jax_model(
                    model, images, labels, with_probs=True
                )
            except Exception as e:
                print(f"JAX 체크포인트 {ckpt['name']} 평가 중 오류: {e}")
                continue
        else:
            # Flax 모델 로드
            try:
                model = load_flax_checkpoint(checkpoint_path)
                metrics, predicted_labels, predicted_probs = evaluate_flax_model(
                    model, images, labels, with_probs=True
                )
            except Exception as e:
                print(f"Flax 체크포인트 {ckpt['name']} 평가 중 오류: {e}")
                continue
        
        # 결과 추가
        result = {
            'checkpoint': ckpt['name'],
            'path': checkpoint_path,
            'step': ckpt.get('step', 0),
            'datetime': ckpt.get('datetime', ''),
            'metrics': metrics
        }
        compare_results.append(result)
        
        # 메트릭 출력
        print_metrics_table(metrics)
    
    # 비교 결과 저장
    if compare_results:
        output_dir = os.path.join(get_results_path(), args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON으로 저장
        comparison_file = os.path.join(output_dir, f"{model_type}_comparison.json")
        import json
        with open(comparison_file, 'w') as f:
            json.dump(compare_results, f, indent=2)
        
        print(f"\n비교 결과가 다음 경로에 저장되었습니다: {comparison_file}")
        
        # 비교 시각화 생성
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # 비교용 데이터 추출
            df_data = []
            for result in compare_results:
                metrics = result['metrics']
                row = {
                    'Checkpoint': result['checkpoint'],
                    'Step': result['step'],
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # 비교 플롯
            plt.figure(figsize=(12, 8))
            
            # 가능한 경우 스텝별 정렬
            if all(r['step'] for r in compare_results):
                df = df.sort_values('Step')
            
            # 메트릭 플롯
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            for metric in metrics_to_plot:
                plt.plot(df['Checkpoint'], df[metric], marker='o', label=metric)
            
            plt.title(f'{model_type.upper()} Model Checkpoint Comparison')
            plt.xlabel('Checkpoint')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 플롯 저장
            comparison_plot = os.path.join(output_dir, f"{model_type}_comparison.png")
            plt.savefig(comparison_plot)
            print(f"비교 플롯이 다음 경로에 저장되었습니다: {comparison_plot}")
            
        except Exception as e:
            print(f"비교 시각화 생성 중 오류 발생: {e}")
    
    return 0

def evaluate_model(args: argparse.Namespace) -> int:
    """
    모델을 평가합니다.
    
    인자:
        args: 명령행 인자
        
    반환값:
        int: 종료 코드
    """
    # 모델 타입이 지정되었는지 확인
    model_type = args.model
    if not model_type:
        logger.error("모델 타입을 지정해야 합니다. --model 옵션을 사용하세요.")
        return 1
    
    logger.info(f"{model_type} 모델 평가를 시작합니다...")
    
    # 체크포인트 로드
    try:
        if args.step:
            # 스텝으로 로드
            logger.info(f"스텝 {args.step}의 체크포인트 로드 중...")
            checkpoint = load_checkpoint_by_step(model_type, args.step, args.subdir)
        elif args.checkpoint:
            # 경로로 로드
            logger.info(f"{args.checkpoint}에서 체크포인트 로드 중...")
            if model_type == 'jax':
                checkpoint = load_jax_checkpoint(args.checkpoint)
            else:
                checkpoint = load_flax_checkpoint(args.checkpoint)
        else:
            # 최신 체크포인트 로드
            logger.info("최신 체크포인트 로드 중...")
            ckpt_info = get_latest_checkpoint(model_type, args.subdir)
            if not ckpt_info:
                print(f"오류: {model_type} 모델에 대한 체크포인트를 찾을 수 없습니다.")
                return 1
            
            if model_type == 'jax':
                checkpoint = load_jax_checkpoint(ckpt_info['path'])
            else:
                checkpoint = load_flax_checkpoint(ckpt_info['path'])
    except Exception as e:
        logger.error(f"체크포인트 로드 실패: {e}")
        return 1
    
    # MNIST 테스트 데이터셋 로드
    logger.info("MNIST 테스트 데이터셋 로드 중...")
    images, labels = load_mnist_test()
    
    # 모델 평가
    logger.info("모델 평가 중...")
    if model_type == 'jax':
        metrics, predicted_labels, predicted_probs = evaluate_jax_model(
            checkpoint, images, labels, with_probs=True
        )
    else:
        metrics, predicted_labels, predicted_probs = evaluate_flax_model(
            checkpoint, images, labels, with_probs=True
        )
    
    # 메트릭 출력
    print("\n평가 결과:")
    print_metrics_table(metrics)
    
    # 메트릭 저장
    output_dir = os.path.join(get_results_path(), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, f"{model_type}_metrics.json")
    save_metrics_to_json(metrics, metrics_file)
    
    logger.info(f"메트릭이 다음 경로에 저장되었습니다: {metrics_file}")
    
    # 요청된 경우 시각화 생성
    if args.visualize:
        logger.info("시각화 생성 중...")
        
        # 혼동 행렬
        cm = calculate_confusion_matrix(labels, predicted_labels)
        visualize_confusion_matrix(
            cm, output_dir=output_dir, 
            filename=f"{model_type}_confusion_matrix.png"
        )
        
        # ROC 곡선
        visualize_roc_curve(
            labels, predicted_probs, 
            output_dir=output_dir,
            filename=f"{model_type}_roc_curve.png"
        )
        
        # 정밀도-재현율 곡선
        visualize_precision_recall_curve(
            labels, predicted_probs, 
            output_dir=output_dir,
            filename=f"{model_type}_precision_recall.png"
        )
        
        # 클래스별 메트릭
        visualize_metrics_by_class(
            metrics, output_dir=output_dir,
            filename=f"{model_type}_class_metrics.png"
        )
        
        # 샘플 예측
        visualize_image_predictions(
            images, labels, predicted_labels, 
            output_dir=output_dir,
            filename=f"{model_type}_samples.png"
        )
        
        logger.info(f"시각화가 다음 경로에 저장되었습니다: {output_dir}")
    
    logger.info("평가가 성공적으로 완료되었습니다.")
    return 0

def main() -> int:
    """
    CLI의 메인 진입점입니다.
    
    반환값:
        int: 종료 코드
    """
    logger.info("JAX/Flax 모델 평가 도구 시작")
    
    # 인자 파싱
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # 다양한 모드 처리
        if args.compare:
            return compare_checkpoints(args)
        elif args.interactive:
            return interactive_mode(args)
        else:
            # 필수 인자가 제공되었는지 확인
            if not args.model:
                parser.print_help()
                logger.error("모델 타입을 지정해야 합니다. --model 옵션을 사용하세요.")
                return 1
            
            # 일반 평가 실행
            return evaluate_model(args)
    except KeyboardInterrupt:
        print("\n사용자에 의해 작업이 취소되었습니다.")
        return 130
    except Exception as e:
        logger.exception(f"예상치 못한 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 