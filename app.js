/* =========================================================================
   app.js  –  UI wiring, CNN autoencoder training & noisy→clean preview
   ========================================================================= */

const q = id => document.getElementById(id);
const log = msg => { q('logs').textContent += msg+'\n'; };

let dl, model;
let trainXs, valXs, noisyTrain, noisyVal;
let testXs,  noisyTest;

/* ----- misc helpers ----- */
function updateStatus(){
  q('dataStatus').innerHTML  = `<strong>Data:</strong> ${trainXs?'loaded':'not loaded'}`;
  q('modelStatus').innerHTML = `<strong>Model:</strong> ${model?'ready':'none'}`;
}

/* ========================================================================
   1. LOAD DATA
   ====================================================================== */
q('loadData').onclick = async ()=>{
  const fTrain = q('trainFile').files[0];
  const fTest  = q('testFile').files[0];
  if(!fTrain||!fTest){ alert('Select BOTH train & test CSV files'); return; }

  try{
    log('📥 Parsing CSV files …');
    if(dl){ dl.train?.xs.dispose(); dl.test?.xs.dispose(); }
    dl = new MNISTDataLoader();

    const train = await dl.loadTrain(fTrain);   // {xs,ys}
    const test  = await dl.loadTest (fTest);

    ({ trainXs, valXs } = dl.split(train.xs, train.ys, 0.1));
    noisyTrain = dl.addNoise(trainXs,0.5);
    noisyVal   = dl.addNoise(valXs,0.5);
    testXs     = test.xs;
    noisyTest  = dl.addNoise(testXs,0.5);

    log(`✔ Data ready – train ${trainXs.shape[0]}, val ${valXs.shape[0]}, test ${testXs.shape[0]}`);
    updateStatus();
  }catch(err){
    console.error(err); alert('Failed to load data: '+err.message);
  }
};

/* ========================================================================
   2. BUILD MODEL
   ====================================================================== */
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

/* ========================================================================
   3. TRAIN
   ====================================================================== */
q('trainBtn').onclick = async ()=>{
  if(!trainXs){ alert('Load data first'); return; }
  if(model) model.dispose();
  model = buildAutoencoder(); updateStatus();

  log('🏋️ Training …');
  await model.fit(noisyTrain, trainXs,{
    epochs:10,batchSize:128,shuffle:true,
    validationData:[noisyVal,valXs],
    callbacks: tfvis.show.fitCallbacks(
      {name:'Autoencoder Training',tab:'Charts'},
      ['loss','val_loss'],{callbacks:['onEpochEnd']}
    )
  });
  log('✔ Training complete');
};

/* ========================================================================
   4. EVALUATE
   ====================================================================== */
q('evalBtn').onclick = async ()=>{
  if(!model||!noisyTest){ alert('Need model & data'); return; }
  const mse=(await model.evaluate(noisyTest,testXs).data())[0];
  log(`📊 Test MSE: ${mse.toFixed(4)}`);
};

/* ========================================================================
   5. PREVIEW 5 RANDOM
   ====================================================================== */
q('testFiveBtn').onclick = ()=>{
  if(!model||!noisyTest){ alert('Need model & data'); return; }
  ['previewNoisy','previewDenoised'].forEach(el=>q(el).innerHTML='');

  const idx = tf.util.createShuffledIndices(testXs.shape[0]).slice(0,5);
  const idxT = tf.tensor1d(idx,'int32');
  const batchNoisy = tf.gather(noisyTest,idxT);
  const denoised   = model.predict(batchNoisy);

  const noisyArr    = batchNoisy.unstack();
  const denoisedArr = denoised.unstack();

  noisyArr.forEach(t=>{
    const c=document.createElement('canvas');
    dl.drawTensorToCanvas(t,c,4);
    q('previewNoisy').appendChild(c);
  });
  denoisedArr.forEach(t=>{
    const c=document.createElement('canvas');
    dl.drawTensorToCanvas(t,c,4);
    q('previewDenoised').appendChild(c);
  });

  noisyArr.forEach(t=>t.dispose()); denoisedArr.forEach(t=>t.dispose());
  batchNoisy.dispose(); denoised.dispose(); idxT.dispose();
};

/* ========================================================================
   6. SAVE / LOAD
   ====================================================================== */
q('saveBtn').onclick = ()=>{ if(model) model.save('downloads://mnist-dae'); else alert('No model'); };

q('loadBtn').onclick = async ()=>{
  const j=q('modelJson').files[0], b=q('modelWeights').files[0];
  if(!j||!b){ alert('Select model files'); return; }
  if(model) model.dispose();
  model = await tf.loadLayersModel(tf.io.browserFiles([j,b]));
  log('✔ Model loaded'); updateStatus();
};

/* ========================================================================
   7. UTIL
   ====================================================================== */
q('resetBtn').onclick = ()=> location.reload();
q('visorBtn').onclick = ()=> tfvis.visor().toggle();
