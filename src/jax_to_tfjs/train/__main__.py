import typer
from .data_loader import MNISTDataLoader
from .flax_trainer import FlaxTrainer
from .jax_trainer import JAXTrainer
from ..models.flax_mnist_cnn import FlaxModelManager
from ..models.jax_mnist_cnn import CNNModel

# evaluate 함수들을 import
from ..evaluation.models.flax_evaluator import evaluate_flax_model
from ..evaluation.models.jax_evaluator import evaluate_jax_model

app = typer.Typer()


@app.command()
def train(
    framework: str = typer.Option(
        ..., "--framework", help="사용할 프레임워크 (flax 또는 jax)"
    ),
    epochs: int = typer.Option(5, "--epochs", help="훈련 에포크 수"),
    learning_rate: float = typer.Option(0.001, "--learning-rate", help="학습률"),
    subdir: str = typer.Option(None, "--subdir", help="체크포인트 저장 하위 디렉토리"),
    evaluate: bool = typer.Option(
        True, "--evaluate/--no-evaluate", help="학습 후 상세 평가 수행 여부"
    ),
):
    """
    모델 훈련 명령어

    학습 후 자동으로 정확도, 정밀도, 재현율, F1 점수 등의 다양한 지표를 계산하고 저장합니다.
    """
    if framework == "flax":
        model_manager = FlaxModelManager()
        model_manager.init_model()
        trainer = FlaxTrainer(model_manager, learning_rate=learning_rate)
    elif framework == "jax":
        model = CNNModel()
        if model.params is None:
            model.init_params()  # 모델 파라미터 초기화
        trainer = JAXTrainer(model, learning_rate=learning_rate)
    else:
        typer.echo("지원하지 않는 프레임워크입니다. '--framework' 옵션을 확인하세요.")
        typer.echo("Usage: python -m jax_to_tfjs.train train --help")
        raise typer.Exit(code=1)

    training_state = trainer.train(
        num_epochs=epochs, subdir=subdir, evaluate_after_training=evaluate
    )
    typer.echo(f"훈련 완료: 최종 손실 = {getattr(training_state, 'loss', None)}")


@app.command()
def evaluate(
    framework: str = typer.Option(
        ..., "--framework", help="사용할 프레임워크 (flax 또는 jax)"
    ),
):
    """
    모델 평가 명령어
    """
    # 테스트 데이터 로드
    test_images, test_labels = MNISTDataLoader.load_mnist_test()

    if framework == "flax":
        model_manager = FlaxModelManager()
        model_manager.init_model()
        state = model_manager.create_train_state(0.001)

        # 평가 수행
        metrics, _, _ = evaluate_flax_model(
            state, test_images, test_labels, with_probs=True
        )
        accuracy = metrics["accuracy"]
    elif framework == "jax":
        model = CNNModel()
        if model.params is None:
            model.init_params()  # 모델 파라미터 초기화

        params = model.params
        if params is not None:
            # 평가 수행
            metrics, _, _ = evaluate_jax_model(
                params, test_images, test_labels, with_probs=True
            )
            accuracy = metrics["accuracy"]
        else:
            typer.echo("모델 파라미터가 초기화되지 않았습니다.")
            raise typer.Exit(code=1)
    else:
        typer.echo("지원하지 않는 프레임워크입니다. '--framework' 옵션을 확인하세요.")
        typer.echo("Usage: python -m jax_to_tfjs.train evaluate --help")
        raise typer.Exit(code=1)

    typer.echo(f"평가 완료: 정확도 = {accuracy}")


if __name__ == "__main__":
    try:
        app()
    except SystemExit as e:
        if e.code != 0:  # 에러 코드가 0이 아닌 경우 도움말 표시
            typer.echo("명령어에 문제가 있습니다. 도움말을 참조하세요.")
            typer.echo("Usage: python -m jax_to_tfjs.train [OPTIONS] COMMAND [ARGS]...")
            typer.echo("Try 'python -m jax_to_tfjs.train --help' for help.")
