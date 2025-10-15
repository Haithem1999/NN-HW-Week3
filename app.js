/* =========================================================================
   app.js – full workflow (noise, train, preview, save / reload)
   ========================================================================= */

const $ = id => document.getElementById(id);
const log = m => { $('logs').textContent += m + '\n'; };

let dl, model;
let trainXs, valXs, noisyTrain, noisyVal, testXs;

function status(){
  $('dataStatus').innerHTML  = `<strong>Data:</strong> ${trainXs?'loaded':'not loaded'}`;
  $('modelStatus').innerHTML = `<strong>Model:</strong> ${model?'ready':'none'}`;
}

/* --------------------------------------------------------------------- */
/* 1️⃣  LOAD DATA + build noisy train/val                                */
/* --------------------------------------------------------------------- */
$('loadData').onclick = async ()=>{
  const fTr = $('trainFile').files[0];
  const fTe = $('testFile').files[0];
  if(!fTr||!fTe) return alert('Select BOTH CSV files');

  try{
    log('📥 Parsing CSVs …');
    if(dl){ dl.train?.xs.dispose(); dl.test?.xs.dispose(); }
    dl = new MNISTDataLoader();

    const tr = await dl.loadTrain(fTr);
    const te = await dl.loadTest (fTe);

    ({ trainXs, valXs } = dl.split(tr.xs, tr.ys, 0.1));
    noisyTrain = dl.addNoise(trainXs,0.5);
    noisyVal   = dl.addNoise(valXs,0.5);
    testXs     = te.xs;

    log(`✔ Data ready: train ${trainXs.shape[0]}, val ${valXs.shape[0]}, test ${testXs.shape[0]}`);
    status();
  }catch(e){ console.error(e); alert('Load failed: '+e.message); }
};

/* --------------------------------------------------------------------- */
/* 2️⃣  Build CNN autoencoder                                            */
/* --------------------------------------------------------------------- */
function buildAE(){
  const inp = tf.input({shape:[28,28,1]});
  let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(inp);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x = tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x = tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
  x = tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
  const out = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
  const m = tf.model({inputs:inp,outputs:out});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

/*  TRAIN */
$('trainBtn').onclick = async ()=>{
  if(!trainXs) return alert('Load data first');
  if(model) model.dispose();
  model = buildAE(); status();

  log('🏋️ Training …');
  await model.fit(noisyTrain, trainXs,{
    epochs:10,batchSize:128,shuffle:true,validationData:[noisyVal,valXs],
    callbacks: tfvis.show.fitCallbacks({name:'Train',tab:'Charts'},['loss','val_loss'])
  });
  log('✔ Training complete');
};

/* --------------------------------------------------------------------- */
/* 3️⃣  Evaluate & Preview 5 random (on-the-fly noisy test)              */
/* --------------------------------------------------------------------- */
$('evalBtn').onclick = async ()=>{
  if(!model) return alert('Train or load a model first');
  const noisyEval = dl.addNoise(testXs,0.5);      // new noise each eval
  const mse = (await model.evaluate(noisyEval,testXs).data())[0];
  noisyEval.dispose();
  log(`📊 Test MSE: ${mse.toFixed(4)}`);
};

$('testFiveBtn').onclick = ()=>{
  if(!model) return alert('Need trained model');
  ['previewNoisy','previewDenoised'].forEach(id=>$(id).innerHTML='');

  /* fresh noise & random 5 indices */
  const noisyFull = dl.addNoise(testXs,0.5);
  const idx = tf.util.createShuffledIndices(testXs.shape[0]).slice(0,5);
  const idxT = tf.tensor1d(idx,'int32');
  const noisyBatch = tf.gather(noisyFull,idxT);

  const cleanPred = model.predict(noisyBatch);

  const noisyArr = noisyBatch.unstack();
  const denArr   = cleanPred.unstack();
  noisyArr.forEach(t=>{ const c=document.createElement('canvas'); dl.draw(t,c); $('previewNoisy').appendChild(c); });
  denArr  .forEach(t=>{ const c=document.createElement('canvas'); dl.draw(t,c); $('previewDenoised').appendChild(c); });

  [noisyArr,denArr].flat().forEach(t=>t.dispose());
  noisyBatch.dispose(); cleanPred.dispose(); noisyFull.dispose(); idxT.dispose();
};

/* --------------------------------------------------------------------- */
/* 4️⃣  Save & Reload                                                    */
/* --------------------------------------------------------------------- */
$('saveBtn').onclick = ()=> model ? model.save('downloads://mnist-dae') : alert('No model');

$('loadBtn').onclick = async ()=>{
  const j=$('modelJson').files[0], b=$('modelWeights').files[0];
  if(!j||!b) return alert('Pick JSON & BIN');
  if(model) model.dispose();
  model = await tf.loadLayersModel(tf.io.browserFiles([j,b]));
  log('✔ Model loaded'); status();
};

/* --------------------------------------------------------------------- */
/* util buttons                                                          */
/* --------------------------------------------------------------------- */
$('resetBtn').onclick = ()=> location.reload();
$('visorBtn').onclick = ()=> tfvis.visor().toggle();
