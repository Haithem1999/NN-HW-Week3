/* =========================================================================
   app.js – workflow: load → train → preview (persistent noisy test) → save/load
   ========================================================================= */

const $ = id=>document.getElementById(id);
const log = m=>{$('logs').textContent+=m+'\n'; console.log(m);};

let dl, model;
let trainXs,valXs,noisyTrain,noisyVal,testXs;
let persistentNoisyTest = null;

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
  }catch(e){
    log('❌ ERROR: '+e.message);
    alert(e.message);
    console.error(e);
  }
};

/* 2. BUILD MODEL - Stable sequential architecture -------------------- */
function buildAE(){
  const model = tf.sequential();
  
  // Encoder
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    padding: 'same'
  }));
  
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    padding: 'same'
  }));
  
  // Bottleneck
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  
  // Decoder
  model.add(tf.layers.upSampling2d({
    size: [2, 2]
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  
  model.add(tf.layers.upSampling2d({
    size: [2, 2]
  }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu'
  }));
  
  // Output
  model.add(tf.layers.conv2d({
    filters: 1,
    kernelSize: 3,
    padding: 'same',
    activation: 'sigmoid'
  }));
  
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });
  
  return model;
}

/* 3. TRAIN ------------------------------------------------------------- */
$('trainBtn').onclick = async ()=>{
  if(!trainXs) {
    alert('Load data first');
    log('❌ Cannot train: No data loaded');
    return;
  }
  
  try{
    if(model) model.dispose();
    model=buildAE(); 
    status();
    log('🏋️ Training CNN Autoencoder for denoising…');
    log('   Architecture: Sequential model with stable layers');
    log('   Input: Noisy images → Output: Clean images');
    
    await model.fit(noisyTrain,trainXs,{
      epochs:25,
      batchSize:128,
      shuffle:true,
      validationData:[noisyVal,valXs],
      callbacks:{
        onEpochEnd: (epoch, logs) => {
          if((epoch+1)%5===0){
            log(`   Epoch ${epoch+1}/25 - loss: ${logs.loss.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}`);
          }
        }
      }
    });
    
    log('✔ Training complete');
  }catch(e){
    log('❌ Training ERROR: '+e.message);
    alert('Training error: '+e.message);
    console.error(e);
  }
};

/* 4. EVALUATE ---------------------------------------------------------- */
$('evalBtn').onclick = async ()=>{
  if(!model) {
    alert('Train / load model first');
    log('❌ Cannot evaluate: No model');
    return;
  }
  if(!persistentNoisyTest) {
    alert('Load data first');
    log('❌ Cannot evaluate: No test data');
    return;
  }
  
  try{
    const result = await model.evaluate(persistentNoisyTest,testXs);
    const mse = Array.isArray(result) ? (await result[0].data())[0] : (await result.data())[0];
    if(Array.isArray(result)) result.forEach(t => t.dispose());
    else result.dispose();
    log(`📊 MSE on noisy test set: ${mse.toFixed(4)}`);
  }catch(e){
    log('❌ Evaluation ERROR: '+e.message);
    alert('Evaluation error: '+e.message);
    console.error(e);
  }
};

/* 5. PREVIEW - Step 3: Display denoising results for 5 random images -- */
$('testFiveBtn').onclick = async ()=>{
  log('🎲 Test 5 Random button clicked...');
  
  if(!model) {
    alert('Need trained model. Please train or load a model first.');
    log('❌ Cannot preview: No model loaded/trained');
    return;
  }
  if(!persistentNoisyTest || !testXs) {
    alert('Load data first. Please load CSV files.');
    log('❌ Cannot preview: No test data loaded');
    return;
  }
  
  try{
    log('✔ Starting preview generation...');
    
    ['previewNoisy','previewDenoised'].forEach(id=>$(id).innerHTML='');

    const totalImages = testXs.shape[0];
    log(`   Total test images available: ${totalImages}`);
    
    const shuffled = tf.util.createShuffledIndices(totalImages);
    const selectedIndices = [];
    for(let i=0; i<5; i++){
      selectedIndices.push(shuffled[i]);
    }
    log(`   Selected indices: ${selectedIndices.join(', ')}`);
    
    const idxT = tf.tensor1d(selectedIndices, 'int32');
    const noisyBatch = tf.gather(persistentNoisyTest, idxT);
    log('   ✔ Gathered noisy batch');
    
    const denoised = model.predict(noisyBatch);
    log('   ✔ Model prediction complete');
    
    // Debug: Check output values
    const denoisedMin = denoised.min().dataSync()[0];
    const denoisedMax = denoised.max().dataSync()[0];
    const denoisedMean = denoised.mean().dataSync()[0];
    log(`   📊 Denoised stats - min: ${denoisedMin.toFixed(4)}, max: ${denoisedMax.toFixed(4)}, mean: ${denoisedMean.toFixed(4)}`);

    const noisyArr = noisyBatch.unstack();
    const denArr = denoised.unstack();
    log('   ✔ Unstacked tensors, rendering to canvas...');
    
    for(let i=0; i<5; i++){
      const cnvN = document.createElement('canvas');
      await dl.draw(noisyArr[i], cnvN); 
      $('previewNoisy').appendChild(cnvN);
      
      const cnvD = document.createElement('canvas');
      await dl.draw(denArr[i], cnvD);  
      $('previewDenoised').appendChild(cnvD);
    }
    
    [noisyArr, denArr].flat().forEach(t=>t.dispose());
    noisyBatch.dispose(); 
    denoised.dispose(); 
    idxT.dispose();
    
    log('✔ Successfully displayed denoising results for 5 random test images');
  }catch(e){
    log('❌ Preview ERROR: '+e.message);
    alert('Error generating preview: '+e.message);
    console.error('Full error:', e);
  }
};

/* 6. SAVE / LOAD ------------------------------------------------------- */
$('saveBtn').onclick = ()=> {
  if(!model) {
    alert('No model to save');
    log('❌ Cannot save: No model');
    return;
  }
  try{
    model.save('downloads://mnist-dae');
    log('✔ Model saved as mnist-dae.json and mnist-dae.weights.bin');
  }catch(e){
    log('❌ Save ERROR: '+e.message);
    alert('Save error: '+e.message);
    console.error(e);
  }
};

$('loadBtn').onclick = async ()=>{
  const j=$('modelJson').files[0], b=$('modelWeights').files[0];
  if(!j||!b) {
    alert('Pick both files (JSON + weights)');
    log('❌ Cannot load: Need both JSON and weights files');
    return;
  }
  
  try{
    if(model) model.dispose();
    log('📥 Loading model from files...');
    model = await tf.loadLayersModel(tf.io.browserFiles([j, b])); 
    log('✔ Model loaded successfully - ready for verification');
    status();
  }catch(e){
    log('❌ Load ERROR: '+e.message);
    alert('Load error: '+e.message);
    console.error(e);
  }
};

/* util ----------------------------------------------------------------- */
$('resetBtn').onclick = ()=> location.reload();
$('visorBtn').onclick = ()=> tfvis.visor().toggle();

status();
log('🚀 Application ready. Load CSV files to begin.');
