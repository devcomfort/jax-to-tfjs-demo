# JAX/FLAX → TensorFlow.js 변환 프로젝트 관례

## 파라미터 구조 및 변환 관례

### JAX/FLAX 파라미터 구조
- **중첩 구조**: 계층적 파라미터 구조 사용 (`params['layer1']['weights']`)
- **명명 규칙**: 일반적으로 `kernel`, `bias` 명명 규칙 사용
- **배치 정규화 파라미터**: `mean`, `var`, `scale`, `bias`

### 파라미터 매핑 관례
- **차원 순서 변환**
  - 합성곱 필터: JAX(OIHW) ↔ TensorFlow(HWIO)
  - Dense 레이어: JAX([출력, 입력]) ↔ TensorFlow([입력, 출력])

- **변환 패턴**
  ```python
  # Dense 레이어 가중치 변환 예시
  dense1_kernel = flax_params['params']['Dense_0']['kernel'].T  # 전치 필요
  dense1_bias = flax_params['params']['Dense_0']['bias']
  
  # TensorFlow 모델에 가중치 설정
  tf_model.layers[0].set_weights([dense1_kernel, dense1_bias])
  ```

### 공통 네이밍 패턴 매핑
- **레이어 명명 규칙**
  - JAX/FLAX: `Conv_0`, `Dense_0`, `BatchNorm_0`
  - TensorFlow: `conv1`, `dense1`, `batch_normalization_1`

- **기타 명명 패턴 매핑**
  ```
  conv(\d+) ↔ Conv_\g<1>
  fc(\d+) ↔ Dense_\g<1>
  dense(\d+) ↔ Dense_\g<1>
  ```

## 체크포인트 관리 관례

### 체크포인트 구조
- **체크포인트 디렉토리**: `checkpoints/{model_name}/{step}`
- **메타데이터 포함**: 모델 이름, 정확도, 학습 설정 등

### 체크포인트 사용 패턴
```python
# 권장 패턴: 컨텍스트 매니저 사용
with JAXCheckpointer() as checkpointer:
    # 체크포인트 저장
    checkpointer.save(1000, model_state, "path/to/checkpoints")
    
    # 체크포인트 로드
    state = checkpointer.load("path/to/checkpoints")

# 대체 패턴: 명시적 리소스 정리
checkpointer = JAXCheckpointer(max_to_keep=5)
try:
    checkpointer.save(step, state, directory)
finally:
    checkpointer.close()
```

## 변환 오류 처리 관례

### 오류 분류 및 처리
- **파라미터 키 불일치**: 유사 이름 매핑, 네이밍 패턴 인식
- **형태 불일치**: 차원 순서 조정(NHWC↔NCHW), 브로드캐스팅
- **데이터 타입 불일치**: 호환 가능한 타입으로 자동 변환

### 오류 로깅 형식
```python
{
    "error_type": "KeyError",
    "category": "parameter_key_mismatch",
    "description": "모델 파라미터 키가 예상 구조와 일치하지 않습니다.",
    "message": "키 'kernel'이 없습니다.",
    "context": {...},
    "can_auto_fix": True
}
```

## TensorFlow.js 모델 관례

### 변환 명령어 패턴
```bash
# TensorFlow 모델 → TensorFlow.js
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  --saved_model_tags=serve \
  --quantize_uint8 \
  saved_model_dir/ \
  tfjs_model/

# 도구를 통한 변환
rye run app convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1
```

### 웹 환경 로딩 패턴
```javascript
// 모델 로드
const model = await tf.loadLayersModel('path/to/model.json');

// 모델 추론
const inputTensor = tf.zeros([1, 28, 28, 1]);
const predictions = model.predict(inputTensor);

// 결과 데이터 접근 및 메모리 관리
const resultData = await predictions.data();
inputTensor.dispose();
predictions.dispose();
``` 