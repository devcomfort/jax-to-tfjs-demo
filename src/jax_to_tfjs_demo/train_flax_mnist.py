from jax_to_tfjs_demo.models.flax_mnist_cnn import train_and_evaluate

if __name__ == '__main__':
    # Flax 모델 학습 및 평가 실행
    state = train_and_evaluate(num_epochs=5, learning_rate=0.001) 