/*  Application logic – trains CNN autoencoder & previews noisy→clean  */

let dl, model;
let trainXs, valXs, noisyTrain, noisyVal;
let testXs,  noisyTest;

const q = id => document.getElementById(id);
const log = msg => { q('logs').textContent += msg + '\n'; };

function updateStatus() {
  q('dataStatus').innerHTML  = `<strong>Data:</strong> ${trainXs ? 'loaded' : 'not loaded'}`;
  q('modelStatus').innerHTML = `<strong>Model:</strong> ${model ? 'ready' : 'none'}`;
}

/* ---------- data loading ---------- */
q('loadData').onclick = async () => {
  const fTrain = q('trainFile').files[0];
  const fTest  = q('testFile').files[0];
  if(!fTrain||!fTest) { alert('Select both train & test CSVs'); return; }

  if(dl) dl.train?.xs.dispose(), dl.test?.xs.dispose();
  dl = new MNISTDataLoader();

  log('Parsing train CSV …');
  const train = await dl.loadTrain(fTrain);
  log('Parsing test CSV …');
  const test  = await dl.loadTest (fTest);

  ({ trainXs, valXs } = dl.split(train.xs, train.ys, 0.1));

  noisyTrain = dl.addNoise(trainXs, 0.5);
  noisyVal   = dl.addNoise(valXs,   0.5);
  testXs     = test.xs;
  noisyTest  = dl.addNoise(testXs, 0.5);

  log(`Loaded: train ${trainXs.shape[0]}, val ${valXs.shape[0]}, test ${testXs.shape[0]}`);
  updateStatus();
};

/* ---------- model ---------- */
function buildAutoencoder(){
  const inp = tf.input({shape:[28,28,1]});
  // encoder
  let x = tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same'}).apply(inp);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x = tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}).apply(x);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  // decoder
  x = tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,activation:'relu',padding:'same'}).apply(x);
  x = tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,activation:'relu',padding:'same'}).apply(x);
  const out = tf.layers.conv2d({filters:1,kernelSize:3,activation:'sigmoid',padding:'same'}).apply(x);

  const m = tf.model({inputs:inp,outputs:out});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

/* ---------- training ---------- */
q('trainBtn').onclick = async () => {
  if(!trainXs) { alert('Load data first'); return; }
  if(model) model.dispose();
  model = buildAutoencoder();
  updateStatus();

  const BATCH=128, EPOCHS=10;
  log('Training …');
  await model.fit(noisyTrain, trainXs, {
    epochs:EPOCHS, batchSize:BATCH, shuffle:true,
    validationData:[noisyVal, valXs],
    callbacks: tfvis.show.fitCallbacks(
      { name:'Autoencoder Training', tab:'Charts' },
      ['loss','val_loss'], { callbacks:['onEpochEnd'] }
    )
  });
  log('Training complete ✔');
};

/* ---------- evaluation ---------- */
q('evalBtn').onclick = async () => {
  if(!model||!noisyTest) return alert('need model & data');
  const loss = (await model.evaluate(noisyTest, testXs).data())[0];
  log(`Test MSE: ${loss.toFixed(4)}`);
};

/* ---------- preview 5 random ---------- */
q('testFiveBtn').onclick = () => {
  if(!model||!noisyTest) { alert('need model & data'); return; }

  /* clear previous canvases */
  for(const el of ['previewNoisy','previewDenoised']) q(el).innerHTML='';

  /* choose 5 random indices then gather both noisy & clean tensors */
  const idx = tf.util.createShuffledIndices(testXs.shape[0]).slice(0,5);
  const idxT = tf.tensor1d(idx,'int32');
  const batchNoisy = tf.gather(noisyTest, idxT);
  const denoised   = model.predict(batchNoisy);

  const noisyArr    = batchNoisy.unstack();
  const denoisedArr = denoised.unstack();

  /* render canvases */
  noisyArr.forEach(t => {
    const c = document.createElement('canvas');
    dl.drawTensorToCanvas(t,c,4);
    q('previewNoisy').appendChild(c);
  });

  denoisedArr.forEach(t => {
    const c = document.createElement('canvas');
    dl.drawTensorToCanvas(t,c,4);
    q('previewDenoised').appendChild(c);
  });

  /* tidy */
  noisyArr.forEach(t=>t.dispose());
  denoisedArr.forEach(t=>t.dispose());
  batchNoisy.dispose(); denoised.dispose(); idxT.dispose();
};

/* ---------- save / load ---------- */
q('saveBtn').onclick = () => {
  if(!model) return alert('train first');
  model.save('downloads://mnist-dae');
};

q('loadBtn').onclick = async () => {
  const j = q('modelJson').files[0];
  const b = q('modelWeights').files[0];
  if(!j||!b) return alert('select model files first');
  if(model) model.dispose();
  model = await tf.loadLayersModel(tf.io.browserFiles([j,b]));
  log('Model loaded ✔');
  updateStatus();
};

/* ---------- utils ---------- */
q('resetBtn').onclick = () => { location.reload(); };
q('visorBtn').onclick = () => tfvis.visor().toggle();
