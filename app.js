/* Main application logic: UI wiring, model training, evaluation & preview */

let trainXs, trainYs, valXs, valYs, testXs, testYs; // dataset tensors
let model;                                          // tf.Model instance
const dataStatusEl  = document.getElementById('data-status');
const trainLogsEl   = document.getElementById('train-logs');
const modelInfoEl   = document.getElementById('model-info');
const previewEl     = document.getElementById('random-preview');
const visorBtn      = document.getElementById('visor-toggle');

// === Utility helpers ===
function log(el, msg) { el.textContent += msg + '\n'; }
function clear(el)     { el.textContent = ''; }
function disableUI(disabled=true){
  document.querySelectorAll('button').forEach(btn=>btn.disabled = disabled);
}

// === UI Event Listeners ===
document.getElementById('load-data').onclick      = loadDataHandler;
document.getElementById('train-btn').onclick       = trainHandler;
document.getElementById('evaluate-btn').onclick    = evaluateHandler;
document.getElementById('test-five-btn').onclick   = testFiveHandler;
document.getElementById('save-model-btn').onclick  = saveModelHandler;
document.getElementById('load-model-btn').onclick  = loadModelHandler;
document.getElementById('reset-btn').onclick       = resetHandler;
visorBtn.onclick = () => tfvis.visor().toggle();

async function loadDataHandler() {
  try {
    const trainFile = document.getElementById('train-csv').files[0];
    const testFile  = document.getElementById('test-csv').files[0];
    if (!trainFile || !testFile) {
      alert('Please select both training and test CSV files.');
      return;
    }
    disableUI(true);
    clear(dataStatusEl); clear(previewEl);
    log(dataStatusEl, 'Loading training data…');
    ({ xs: trainXs, ys: trainYs } = await loadTrainFromFiles(trainFile));
    log(dataStatusEl, `Training samples: ${trainXs.shape[0]}`);

    ({ xs: testXs, ys: testYs }   = await loadTestFromFiles(testFile));
    log(dataStatusEl, `Test samples:    ${testXs.shape[0]}`);

    // split train/val
    ({ trainXs, trainYs, valXs, valYs } =
      splitTrainVal(trainXs, trainYs, 0.1));
    log(dataStatusEl, `Validation split: ${valXs.shape[0]}`);

    log(dataStatusEl, 'Data ready ✔');
  } catch (e) {
    console.error(e);
    alert('Error loading data: ' + e.message);
  } finally { disableUI(false); }
}

// === Model Definition ===
function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({
    filters:32, kernelSize:3, activation:'relu', padding:'same',
    inputShape:[28,28,1]
  }));
  m.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu', padding:'same'}));
  m.add(tf.layers.maxPooling2d({poolSize:2}));
  m.add(tf.layers.dropout({rate:0.25}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:128, activation:'relu'}));
  m.add(tf.layers.dropout({rate:0.5}));
  m.add(tf.layers.dense({units:10, activation:'softmax'}));

  m.compile({
    optimizer:'adam',
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  });
  return m;
}

// === Training ===
async function trainHandler() {
  if (!trainXs) return alert('Load data first.');
  if (model) model.dispose();
  model = createModel();
  model.summary();
  modelInfoEl.textContent = '';
  model.summary(line => modelInfoEl.textContent += line + '\n');

  const BATCH = 128, EPOCHS = 6;
  clear(trainLogsEl);
  const surface = { name:'Training', tab:'Charts' };
  const cbks = tfvis.show.fitCallbacks(surface, ['loss','val_loss','acc','val_acc']);

  await model.fit(trainXs, trainYs, {
    epochs:EPOCHS, batchSize:BATCH, shuffle:true,
    validationData:[valXs, valYs],
    callbacks:{
      onEpochEnd: async (epoch, logs) => {
        log(trainLogsEl, `Epoch ${epoch+1}/${EPOCHS} – loss:${logs.loss.toFixed(3)} acc:${(logs.acc*100).toFixed(1)}%`);
        await tf.nextFrame(); // keep UI responsive
      },
      ...cbks
    }
  });
  log(trainLogsEl, 'Training complete ✔');
}

// === Evaluation ===
async function evaluateHandler() {
  if (!model || !testXs) return alert('Need trained & test data.');
  disableUI(true);
  tfvis.visor().surface({name:'Metrics', tab:'Charts'});
  const preds = model.predict(testXs).argMax(-1);
  const labels = testYs.argMax(-1);

  const accuracy = await preds.equal(labels).sum().data();
  const acc = accuracy[0] / testXs.shape[0];
  log(trainLogsEl, `Test accuracy: ${(acc*100).toFixed(2)}%`);

  // Confusion matrix
  const cm = await tf.math.confusionMatrix(labels, preds, 10).array();
  tfvis.render.confusionMatrix(
    {name:'Confusion Matrix', tab:'Charts'},
    {values:cm, tickLabels:[0,1,2,3,4,5,6,7,8,9]}
  );

  // Per-class accuracy
  const perClassAcc = cm.map((row,i)=>
    row[i]/ row.reduce((a,b)=>a+b,0) || 0 );
  tfvis.render.barchart(
    {name:'Per-Class Accuracy', tab:'Charts'},
    perClassAcc.map((v,i)=>({index:i, value:v}))
  );

  preds.dispose(); labels.dispose();
  disableUI(false);
}

// === Random 5 Preview ===
function testFiveHandler() {
  if (!model || !testXs) return alert('Need model & test data.');
  previewEl.innerHTML = '';
  const { batchXs, batchYs } = getRandomTestBatch(testXs, testYs, 5);
  const preds = model.predict(batchXs).argMax(-1).dataSync();
  const labels= batchYs.argMax(-1).dataSync();

  tf.tidy(()=>{
    for (let i=0;i<5;i++){
      const canvas = document.createElement('canvas');
      draw28x28ToCanvas(batchXs.slice([i,0,0,0],[1,28,28,1])
                                  .reshape([28,28]), canvas, 4);
      previewEl.appendChild(canvas);
      const span   = document.createElement('span');
      span.className='label';
      span.style.color = preds[i]===labels[i]?'green':'red';
      span.textContent = preds[i];
      previewEl.appendChild(span);
    }
  });
  batchXs.dispose(); batchYs.dispose();
}

// === Model Save / Load ===
async function saveModelHandler() {
  if (!model) return alert('Train or load a model first.');
  await model.save('downloads://mnist-cnn');
}

async function loadModelHandler() {
  const jsonFile = document.getElementById('upload-json').files[0];
  const binFile  = document.getElementById('upload-weights').files[0];
  if (!jsonFile || !binFile) return alert('Select model.json and weights.bin');
  if (model) model.dispose();
  model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
  modelInfoEl.textContent = '';
  model.summary(line => modelInfoEl.textContent += line + '\n');
  alert('Model loaded ✔');
}

// === Reset ===
function resetHandler(){
  [trainXs,trainYs,valXs,valYs,testXs,testYs].forEach(t=>t && t.dispose());
  if (model) model.dispose();
  trainXs=trainYs=valXs=valYs=testXs=testYs=model=null;
  clear(dataStatusEl); clear(trainLogsEl); clear(modelInfoEl); previewEl.innerHTML='';
  tfvis.visor().close();
  alert('State cleared.');
}
