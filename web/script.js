const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");
const predictText = document.getElementById("predictText");

const CLASSES = [
  'an eye',
  'a bicycle',
  'a tree',
  'an alarm clock',
  'a book',
  'an airplane',
  'a cell phone',
  'a smiley face',
  'an apple',
  'a car'
];

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// load model
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("../models/onnx_model.onnx");

async function updatePredictions() {
  // get predictions
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
  const outputMap = await sess.run([input]);
  const predictions = outputMap.values().next().value.data;
  const maxPrediction = Math.max(...predictions);
  const maxPredIndex = predictions.indexOf(maxPrediction)

  // change text
  predictText.innerText = `${Math.round(maxPrediction * 100)}% sure this is ${CLASSES[maxPredIndex]}`
}

ctx.lineWidth = 28;
ctx.lineJoin = "round";

ctx.font = "25px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";

ctx.fillStyle = "#212121";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

ctx.strokeStyle = "#212121";

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  predictText.innerText = "Ready to predict!"
}

function drawLine(fromX, fromY, toX, toY) {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
}

function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }

  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

function canvasMouseUp(event) {
  predictText.innerText = "Thinking...";
  updatePredictions();
}

loadingModelPromise.then(() => {
  canvas.addEventListener("mouseup", canvasMouseUp);
  canvas.addEventListener("mousedown", canvasMouseDown);
  canvas.addEventListener("mousemove", canvasMouseMove);
  document.body.addEventListener("mouseup", bodyMouseUp);
  document.body.addEventListener("mouseout", bodyMouseOut);
  clearButton.addEventListener("mousedown", clearCanvas);

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillText("Draw something", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})