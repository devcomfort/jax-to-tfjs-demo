# TensorFlow.js 모델 로딩 가이드

이 문서는 JAX/Flax 모델을 TensorFlow.js로 변환한 후 다양한 환경에서 로드하고 사용하는 방법에 대해 설명합니다. TensorFlow와 TensorFlow.js 간의 모델 로딩 문법 차이를 중점적으로 다룹니다.

## 목차
1. [모델 변환 과정 개요](#모델-변환-과정-개요)
2. [TensorFlow와 TensorFlow.js 로딩 비교](#tensorflow와-tensorflowjs-로딩-비교)
3. [다양한 환경에서의 모델 로딩](#다양한-환경에서의-모델-로딩)
   - [브라우저 환경](#브라우저-환경)
   - [Node.js 환경](#nodejs-환경)
4. [모델 포맷 비교](#모델-포맷-비교)
5. [로딩 시 주의사항](#로딩-시-주의사항)

## 모델 변환 과정 개요

JAX/Flax 모델을 TensorFlow.js에서 사용하기 위해서는 다음과 같은 변환 단계를 거칩니다:

1. JAX/Flax 모델 체크포인트 로드
2. TensorFlow 모델로 변환 (가중치 구조 매핑)
3. TensorFlow.js 포맷으로 변환
4. 선택적으로 모델 양자화 수행

```bash
# 변환 예시 명령어
rye run app convert --model-type jax_mnist --checkpoint-path checkpoints/jax_mnist/1
```

## TensorFlow와 TensorFlow.js 로딩 비교

### TensorFlow 모델 로딩 (Python)

```python
# TensorFlow에서 모델 로딩
import tensorflow as tf

# SavedModel 포맷 로드
model = tf.keras.models.load_model('path/to/model')

# h5 포맷 로드
model = tf.keras.models.load_model('path/to/model.h5')

# 모델 추론
input_data = tf.zeros([1, 28, 28, 1])  # 예: MNIST 입력
predictions = model.predict(input_data)
```

### TensorFlow.js 모델 로딩 (JavaScript)

```javascript
// TensorFlow.js에서 모델 로딩
import * as tf from '@tensorflow/tfjs';
// 또는 브라우저 환경에서:
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0"></script>

// Layers 모델 로드 (Keras 구조 유지)
const model = await tf.loadLayersModel('path/to/model.json');

// Graph 모델 로드 (TensorFlow SavedModel에서 변환된 경우)
// const model = await tf.loadGraphModel('path/to/model.json');

// 모델 추론
const inputTensor = tf.zeros([1, 28, 28, 1]);  // 예: MNIST 입력
const predictions = model.predict(inputTensor);

// 결과 데이터 접근
const resultData = await predictions.data();
console.log(resultData);

// 메모리 정리 (TensorFlow.js에서는 수동으로 메모리 관리)
inputTensor.dispose();
predictions.dispose();
```

## 다양한 환경에서의 모델 로딩

### 브라우저 환경

```html
<!DOCTYPE html>
<html>
<head>
  <title>TensorFlow.js 모델 로딩 예제</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0"></script>
</head>
<body>
  <div id="output"></div>
  
  <script>
    async function loadAndRunModel() {
      const outputDiv = document.getElementById('output');
      outputDiv.innerHTML = '모델 로딩 중...';
      
      try {
        // 모델 로드 - 로컬 또는 CDN 경로 지정
        const model = await tf.loadLayersModel('tfjs_models/jax_mnist/model.json');
        
        // 입력 데이터 생성
        const inputTensor = tf.zeros([1, 28, 28, 1]);
        
        // 예측 수행
        const result = model.predict(inputTensor);
        
        // 결과 표시
        const resultData = await result.data();
        outputDiv.innerHTML = `<p>모델 로드 및 추론 성공!</p>
                              <p>결과: ${JSON.stringify(Array.from(resultData).slice(0, 5))}</p>`;
        
        // 메모리 정리
        inputTensor.dispose();
        result.dispose();
      } catch (error) {
        outputDiv.innerHTML = `<p>오류 발생: ${error.message}</p>`;
      }
    }
    
    // 페이지 로드 시 모델 로드 실행
    loadAndRunModel();
  </script>
</body>
</html>
```

### Node.js 환경

```javascript
// 파일명: test_model.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function loadAndTestModel() {
  try {
    // 모델 로드
    console.log('모델 로드 중...');
    const modelPath = 'file://./tfjs_models/jax_mnist/model.json';
    const model = await tf.loadLayersModel(modelPath);
    
    // 모델 구조 확인
    console.log('모델 로드 성공!');
    model.summary();
    
    // 추론 테스트
    console.log('\n추론 테스트 실행 중...');
    const inputTensor = tf.ones([1, 28, 28, 1]);
    const result = model.predict(inputTensor);
    
    // 결과 확인
    const resultData = await result.data();
    console.log('추론 결과 (처음 5개):', Array.from(resultData).slice(0, 5));
    
    // 메모리 정리
    inputTensor.dispose();
    result.dispose();
  } catch (error) {
    console.error('모델 로드 또는 추론 실패:', error);
  }
}

// 테스트 실행
loadAndTestModel();
```

```bash
# 실행 방법
npm install @tensorflow/tfjs-node
node test_model.js
```

## 모델 포맷 비교

TensorFlow.js는 두 가지 주요 포맷을 지원합니다:

1. **Layers 모델 (loadLayersModel)**
   - Keras 모델 구조를 유지
   - `model.json` 파일에 모델 토폴로지와 가중치 정보가 포함됨
   - `model.layers`, `model.summary()` 등의 Keras 유사 API 사용 가능
   - 예: `const model = await tf.loadLayersModel('path/to/model.json');`

2. **Graph 모델 (loadGraphModel)**
   - TensorFlow SavedModel에서 변환된 모델 포맷
   - TensorFlow 그래프 구조 유지
   - 텐서 이름과 연산 그래프로 접근
   - 예: `const model = await tf.loadGraphModel('path/to/model.json');`

변환된 모델의 `model.json` 파일을 살펴보면 `"format": "layers-model"` 또는 `"format": "graph-model"` 속성을 통해 어떤 포맷인지 알 수 있습니다.

## 로딩 시 주의사항

1. **경로 지정 방식**
   - 브라우저: 상대 또는 절대 URL 경로 사용
     - `'./model.json'` (같은 디렉토리)
     - `'https://example.com/models/model.json'` (원격 URL)
   - Node.js: `'file://'` 프로토콜 사용
     - `'file:///absolute/path/to/model.json'` (절대 경로)
     - `'file://./model.json'` (상대 경로)

2. **메모리 관리**
   - TensorFlow.js는 수동 메모리 관리 필요
   - 사용이 끝난 텐서는 반드시 `dispose()` 호출로 메모리 해제
   - 예: `tensor.dispose()` 또는 `tf.tidy(() => { ... })`

3. **배치 크기 처리**
   - 모델 입력의 첫 번째 차원은 보통 배치 크기
   - `null` 또는 `undefined` 배치 크기는 동적 배치 크기 의미
   - 단일 샘플 추론 시: `[1, 28, 28, 1]`과 같이 명시적 배치 차원 필요

4. **비동기 처리**
   - TensorFlow.js 모델 로딩과 데이터 접근은 비동기 작업
   - `async/await` 또는 Promise 체인 사용 필요
   - 예: `const model = await tf.loadLayersModel(...)`

5. **양자화된 모델 사용**
   - 양자화된 모델은 크기가 작지만 정확도 손실 가능성 있음
   - 웹 환경에서 로딩 시간과 네트워크 전송 시간 단축
   - 모바일 기기나 저사양 환경에서 유용 