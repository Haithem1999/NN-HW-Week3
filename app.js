/* =========================================================================
   app.js – workflow: load → train → preview (fresh noisy test) → save/load
   ========================================================================= */

const $ = id=>document.getElementById(id);
const log = m=>{$('logs').textContent+=m+'\n';};

let dl, model;
let trainXs,valXs,noisyTrain,noisyVal,testXs;

function status(){
  $('dataStatus').innerHTML  = `<strong>Data:</strong> ${trainXs?'loaded':'not loaded'}`;
  $('modelStatus').innerHTML = `<strong>Model:</strong> ${model?'ready':'none'}`;
}

/* 1. LOAD DATA --------------------------------------------------------- */
$('loadData').onclick = async ()=>{
  const fTr=$('trainFile').files[0], fTe=$('testFile').files[0];
  if(!fTr||!fTe) return alert('Select BOTH CSV files');
  try{
    log('📥 Parsing CSV …');
    if(dl){ dl.train?.xs.dispose(); dl.test?.xs.dispose(); }
    dl=new MNISTDataLoader();
    const tr=await dl.loadTrain(fTr), te=await dl.loadTest(fTe);
    ({trainXs,valXs}=dl.split(tr.xs,tr.ys,0.1));
    noisyTrain=dl.addNoise(trainXs,0.5);
    noisyVal  =dl.addNoise(valXs,0.5);
    testXs    =te.xs;
    log(`✔ Data ready – train ${trainXs.shape[0]} / val ${valXs.shape[0]} / test ${testXs.shape[0]}`);
    status();
  }catch(e){alert(e.message);}
};

/* 2. BUILD MODEL ------------------------------------------------------- */
function buildAE(){
  const i=tf.input({shape:[28,28,1]});
  let x=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(i);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x=tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x=tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
  x=tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
  const o=tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);
  const m=tf.model({inputs:i,outputs:o});
  m.compile({optimizer:'adam',loss:'meanSquaredError'});
  return m;
}

/* 3. TRAIN ------------------------------------------------------------- */
$('trainBtn').onclick = async ()=>{
  if(!trainXs) return alert('Load data first');
  if(model) model.dispose();
  model=buildAE(); status();
  log('🏋️ Training …');
  await model.fit(noisyTrain,trainXs,{
    epochs:10,batchSize:128,shuffle:true,
    validationData:[noisyVal,valXs],
    callbacks:tfvis.show.fitCallbacks({name:'Training',tab:'Charts'},['loss','val_loss'])
  });
  log('✔ Training complete');
};

/* 4. EVALUATE ---------------------------------------------------------- */
$('evalBtn').onclick = async ()=>{
  if(!model) return alert('Train / load model first');
  const noisyEval=dl.addNoise(testXs,0.5);
  const mse=(await model.evaluate(noisyEval,testXs).data())[0];
  log(`📊 MSE on fresh noisy test: ${mse.toFixed(4)}`);
  noisyEval.dispose();
};

/* 5. PREVIEW (fresh noise each click) ---------------------------------- */
$('testFiveBtn').onclick = async ()=>{
  if(!model) return alert('Need trained model');
  ['previewNoisy','previewDenoised'].forEach(id=>$(id).innerHTML='');

  const idx=tf.util.createShuffledIndices(testXs.shape[0]).slice(0,5);
  const idxT=tf.tensor1d(idx,'int32');
  const cleanBatch=tf.gather(testXs,idxT);
  const noisyBatch=dl.addNoise(cleanBatch,0.5);
  const denoised=model.predict(noisyBatch);

  const noisyArr=noisyBatch.unstack(), denArr=denoised.unstack();
  for(let i=0;i<5;i++){
    const cnvN=document.createElement('canvas');
    await dl.draw(noisyArr[i],cnvN); $('previewNoisy').appendChild(cnvN);
    const cnvD=document.createElement('canvas');
    await dl.draw(denArr[i],cnvD);  $('previewDenoised').appendChild(cnvD);
  }
  [noisyArr,denArr].flat().forEach(t=>t.dispose());
  cleanBatch.dispose(); noisyBatch.dispose(); denoised.dispose(); idxT.dispose();
};

/* 6. SAVE / LOAD ------------------------------------------------------- */
$('saveBtn').onclick = ()=> model ? model.save('downloads://mnist-dae') : alert('No model');
$('loadBtn').onclick = async ()=>{
  const j=$('modelJson').files[0], b=$('modelWeights').files[0];
  if(!j||!b) return alert('Pick both files'); if(model) model.dispose();
  model=await tf.loadLayersModel(tf.io.browserFiles([j,b])); log('✔ Model loaded'); status();
};

/* util ----------------------------------------------------------------- */
$('resetBtn').onclick = ()=> location.reload();
$('visorBtn').onclick = ()=> tfvis.visor().toggle();
