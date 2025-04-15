<script>
  import { onMount } from 'svelte';
  import * as tf from '@tensorflow/tfjs';
  
  // 상태 변수
  let model = null;
  let canvas = null;
  let ctx = null;
  let isDrawing = false;
  let prediction = null;
  let loading = true;
  let error = null;
  let debugInfo = [];
  
  // 모델 로드
  async function loadModel() {
    try {
      const startTime = performance.now();
      const modelUrl = 'tfjs_models/jax_mnist/model.json';
      console.log('모델 로드 시도:', modelUrl);
      
      // 모델 파일 존재 여부 확인
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`모델 파일을 찾을 수 없습니다 (${response.status}): ${modelUrl}\n모델을 먼저 변환해야 합니다.`);
      }
      
      model = await tf.loadLayersModel(modelUrl);
      const endTime = performance.now();
      
      // 모델 정보 로깅
      console.log('모델 로드 완료:', {
        loadTime: `${(endTime - startTime).toFixed(2)}ms`,
        inputShape: model.inputs[0].shape,
        outputShape: model.outputs[0].shape
      });
      
      // 웜업 실행
      const dummyInput = tf.zeros([1, 28, 28, 1]);
      const warmupResult = model.predict(dummyInput);
      warmupResult.dispose();
      dummyInput.dispose();
      
      loading = false;
    } catch (e) {
      error = `모델 로드 중 오류가 발생했습니다: ${e.message}\n모델을 먼저 변환해야 합니다.`;
      console.error('모델 로드 실패:', e);
    }
  }
  
  // 캔버스 초기화
  function initCanvas() {
    if (!canvas) return;
    
    ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
  }
  
  // 그림 그리기 시작
  function startDrawing(e) {
    isDrawing = true;
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }
  
  // 그림 그리기 중
  function draw(e) {
    if (!isDrawing) return;
    
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }
  
  // 그림 그리기 종료
  function stopDrawing() {
    isDrawing = false;
    predict();
  }
  
  // 마우스 위치 가져오기
  function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }
  
  // 캔버스 지우기
  function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    prediction = null;
  }
  
  // 예측 수행
  async function predict() {
    if (!model) return;
    
    try {
      const startTime = performance.now();
      
      // 캔버스 이미지를 28x28 크기로 변환
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(255.0)
        .expandDims();
      
      // 예측 수행
      const predictions = await model.predict(tensor).data();
      const maxIndex = predictions.indexOf(Math.max(...predictions));
      
      const endTime = performance.now();
      console.log('예측 완료:', {
        predictionTime: `${(endTime - startTime).toFixed(2)}ms`,
        digit: maxIndex,
        confidence: predictions[maxIndex] * 100
      });
      
      prediction = {
        digit: maxIndex,
        confidence: predictions[maxIndex] * 100,
        probabilities: Array.from(predictions)
      };
      
      // 메모리 정리
      tensor.dispose();
    } catch (e) {
      console.error('예측 중 오류 발생:', e);
      error = '예측 중 오류가 발생했습니다.';
    }
  }
  
  onMount(() => {
    loadModel();
    initCanvas();
  });
</script>

<main>
  <h1>MNIST 손글씨 인식 데모</h1>
  
  {#if loading}
    <p>모델을 로드하는 중...</p>
  {:else if error}
    <p class="error">{error}</p>
  {:else}
    <div class="container">
      <div class="canvas-container">
        <canvas
          width={280}
          height={280}
          bind:this={canvas}
          on:mousedown={startDrawing}
          on:mousemove={draw}
          on:mouseup={stopDrawing}
          on:mouseleave={stopDrawing}
        />
        <button on:click={clearCanvas}>지우기</button>
      </div>
      
      {#if prediction}
        <div class="prediction">
          <h2>예측 결과</h2>
          <p>숫자: {prediction.digit}</p>
          <p>신뢰도: {prediction.confidence.toFixed(2)}%</p>
          
          <div class="probability-bars">
            {#each prediction.probabilities as prob, i}
              <div class="bar-container">
                <div class="bar" style="height: {prob * 100}%"></div>
                <span class="digit">{i}</span>
                <span class="probability">{prob.toFixed(2)}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</main>

<style>
  main {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    text-align: center;
  }
  
  h1 {
    color: #333;
    margin-bottom: 2rem;
  }
  
  .container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 2rem;
  }
  
  .canvas-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }
  
  canvas {
    border: 2px solid #333;
    border-radius: 4px;
    cursor: crosshair;
    background-color: white;
  }
  
  button {
    padding: 0.5rem 1rem;
    background-color: #333;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
  }
  
  button:hover {
    background-color: #444;
  }
  
  .prediction {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f9f9f9;
    min-width: 300px;
  }
  
  .probability-bars {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    height: 200px;
    margin-top: 1rem;
    padding: 1rem;
    background-color: #f0f0f0;
    border-radius: 4px;
  }
  
  .bar-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 30px;
  }
  
  .bar {
    width: 100%;
    background-color: #4CAF50;
    transition: height 0.3s;
    border-radius: 4px 4px 0 0;
  }
  
  .digit {
    margin-top: 0.5rem;
    font-size: 0.8rem;
  }
  
  .probability {
    font-size: 0.7rem;
    color: #666;
  }
  
  .error {
    color: red;
    font-weight: bold;
  }
</style> 