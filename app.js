/* =========================================================================
   app.js – workflow: load → train → preview (persistent noisy test) → save/load
   ========================================================================= */

const $ = id=>document.getElementById(id);
const log = m=>{$('logs').textContent+=m+'\n';};

let dl, model;
let trainXs,valXs,noisyTrain,noisyVal,testXs;
let persistentNoisyTest = null; // Store noisy test data for consistent visualization

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
    if(dl){ 
      dl.train?.xs.dispose(); 
      dl.test?.xs.dispose(); 
    }
    if(persistentNoisyTest) persistentNoisyTest.dispose();
    persistentNoisyTest = null;
    
    dl=new MNISTDataLoader();
    const tr=await dl.loadTrain(fTr), te=await dl.loadTest(fTe);
    ({trainXs,valXs}=dl.split(tr.xs,tr.ys,0.1));
    
    // Step 1: Add random noise to training and validation data
    noisyTrain=dl.addNoise(trainXs,0.5);
    noisyVal  =dl.addNoise(valXs,0.5);
    testXs    =te.xs;
    
    // Generate persistent noisy test set for consistent visualization
    persistentNoisyTest = dl.addNoise(testXs, 0.5);
    
    log(`✔ Data ready – train ${trainXs.shape[0]} / val ${valXs.shape[0]} / test ${testXs.shape[0]}`);
    log('✔ Noise added to train/val/test data (noise factor: 0.5)');
    status();
  }catch(e){alert(e.message);}
};

/* 2. BUILD MODEL ------------------------------------------------------- */
function buildAE(){
  // Step 2: Build CNN Autoencoder architecture for denoising
  const i=tf.input({shape:[28,28,1]});
  
  // Encoder
  let x=tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(i);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  x=tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
  x=tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
  
  // Decoder
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
  
  // Step 2: Train the CNN autoencoder
  model=buildAE(); 
  status();
  log('🏋️ Training CNN Autoencoder for denoising…');
  log('   Input: Noisy images → Output: Clean images');
  
  await model.fit(noisyTrain,trainXs,{
    epochs:10,
    batchSize:128,
    shuffle:true,
    validationData:[noisyVal,valXs],
    callbacks:tfvis.show.fitCallbacks({name:'Training',tab:'Charts'},['loss','val_loss'])
  });
  
  log('✔ Training complete');
};

/* 4. EVALUATE ---------------------------------------------------------- */
$('evalBtn').onclick = async ()=>{
  if(!model) return alert('Train / load model first');
  if(!persistentNoisyTest) return alert('Load data first');
  
  const mse=(await model.evaluate(persistentNoisyTest,testXs).data())[0];
  log(`📊 MSE on noisy test set: ${mse.toFixed(4)}`);
};

/* 5. PREVIEW - Step 3: Display denoising results for 5 random images -- */
$('testFiveBtn').onclick = async ()=>{
  if(!model) return alert('Need trained model');
  if(!persistentNoisyTest) return alert('Load data first');
  
  // Clear previous previews
  ['previewNoisy','previewDenoised'].forEach(id=>$(id).innerHTML='');

  // Step 3: Select 5 random images from test set
  const idx=tf.util.createShuffledIndices(testXs.shape[0]).slice(0,5);
  const idxT=tf.tensor1d(idx,'int32');
  
  // Get the persistent noisy versions (same noise each time)
  const noisyBatch=tf.gather(persistentNoisyTest,idxT);
  
  // Denoise using trained model
  const denoised=model.predict(noisyBatch);

  // Display results
  const noisyArr=noisyBatch.unstack(), denArr=denoised.unstack();
  for(let i=0;i<5;i++){
    const cnvN=document.createElement('canvas');
    await dl.draw(noisyArr[i],cnvN); 
    $('previewNoisy').appendChild(cnvN);
    
    const cnvD=document.createElement('canvas');
    await dl.draw(denArr[i],cnvD);  
    $('previewDenoised').appendChild(cnvD);
  }
  
  [noisyArr,denArr].flat().forEach(t=>t.dispose());
  noisyBatch.dispose(); 
  denoised.dispose(); 
  idxT.dispose();
  
  log('✔ Displayed denoising results for 5 random test images');
};

/* 6. SAVE / LOAD - Step 4: Save and reload model ---------------------- */
$('saveBtn').onclick = ()=> {
  if(!model) return alert('No model to save');
  // Step 4: Save the trained model
  model.save('downloads://mnist-dae');
  log('✔ Model saved as mnist-dae.json and mnist-dae.weights.bin');
};

$('loadBtn').onclick = async ()=>{
  const j=$('modelJson').files[0], b=$('modelWeights').files[0];
  if(!j||!b) return alert('Pick both files (JSON + weights)'); 
  
  if(model) model.dispose();
  
  // Step 4: Load the saved model to verify results
  model=await tf.loadLayersModel(tf.io.browserFiles([j,b])); 
  log('✔ Model loaded successfully - ready for verification');
  status();
};

/* util ----------------------------------------------------------------- */
$('resetBtn').onclick = ()=> location.reload();
$('visorBtn').onclick = ()=> tfvis.visor().toggle();
