// MNIST Denoiser Autoencoder Application
let dataLoader;
let model;
let trainXs, valXs, trainNoisyXs, valNoisyXs;
let testXs, testNoisyXs;
const logsEl      = document.getElementById('trainingLogs');
const dataStatus  = document.getElementById('dataStatus');
const modelInfoEl = document.getElementById('modelInfo');
const previewNoisyEl    = document.getElementById('previewNoisy');
const previewDenoisedEl = document.getElementById('previewDenoised');

function log(msg){ const div=document.createElement('div'); div.textContent=msg; logsEl.appendChild(div); }

function clearLogs(){ logsEl.innerHTML=''; }
function clearPreview(){ previewNoisyEl.innerHTML=''; previewDenoisedEl.innerHTML=''; }

// === Data Loading ===
document.getElementById('loadDataBtn').onclick = async ()=>{
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile  = document.getElementById('testFile').files[0];
    if(!trainFile||!testFile){ alert('Select both train and test CSV files'); return;}
    if(dataLoader) dataLoader.dispose();
    dataLoader = new MNISTDataLoader();
    log('Loading training data…');
    const train = await dataLoader.loadTrainFromFiles(trainFile);
    log('Loading test data…');
    const test  = await dataLoader.loadTestFromFiles(testFile);

    // Split
    const split = dataLoader.splitTrainVal(train.xs, train.ys, 0.1);
    trainXs = split.trainXs; valXs = split.valXs;

    // Create noisy versions
    trainNoisyXs = dataLoader.addNoise(trainXs, 0.5);
    valNoisyXs   = dataLoader.addNoise(valXs, 0.5);
    testXs       = test.xs;
    testNoisyXs  = dataLoader.addNoise(testXs, 0.5);

    dataStatus.innerHTML = `<h3>Data Status</h3>
        <p>Train: ${trainXs.shape[0]} (noisy pairs)</p>
        <p>Val:   ${valXs.shape[0]}</p>
        <p>Test:  ${testXs.shape[0]}</p>`;
    log('Data ready ✔');
};

// === Model Creation ===
function createAutoencoder(){
    const input = tf.input({shape:[28,28,1]});

    // Encoder
    let x = tf.layers.conv2d({filters:32,kernelSize:3,padding:'same',activation:'relu'}).apply(input);
    x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:64,kernelSize:3,padding:'same',activation:'relu'}).apply(x);
    x = tf.layers.maxPooling2d({poolSize:2,padding:'same'}).apply(x);

    // Decoder
    x = tf.layers.conv2dTranspose({filters:64,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
    x = tf.layers.conv2dTranspose({filters:32,kernelSize:3,strides:2,padding:'same',activation:'relu'}).apply(x);
    const output = tf.layers.conv2d({filters:1,kernelSize:3,padding:'same',activation:'sigmoid'}).apply(x);

    const auto = tf.model({inputs:input, outputs:output});
    auto.compile({optimizer:'adam', loss:'meanSquaredError'});
    return auto;
}

// === Training ===
document.getElementById('trainBtn').onclick = async ()=>{
    if(!trainXs){ alert('Load data first'); return; }
    if(model) model.dispose();
    model = createAutoencoder();
    modelInfoEl.textContent='';
    model.summary(line=>modelInfoEl.textContent+=line+'\\n');

    const BATCH=128,EPOCHS=10;
    clearLogs();
    const history = await model.fit(trainNoisyXs, trainXs, {
        epochs:EPOCHS,
        batchSize:BATCH,
        validationData:[valNoisyXs, valXs],
        shuffle:true,
        callbacks: tfvis.show.fitCallbacks(
            { name:'Autoencoder Training', tab:'Charts'},
            ['loss','val_loss'],
            { callbacks:['onEpochEnd'] }
        )
    });
    log('Training complete ✔');
};

// === Evaluation ===
document.getElementById('evaluateBtn').onclick = async ()=>{
    if(!model||!testNoisyXs){ alert('Need trained model and data'); return; }
    const evalLoss = await model.evaluate(testNoisyXs, testXs).data();
    log(`Test MSE: ${evalLoss[0].toFixed(4)}`);
};

// === Test 5 Random Denoise ===
document.getElementById('testFiveBtn').onclick = ()=>{
    if(!model||!testNoisyXs){ alert('Need model and data'); return; }
    clearPreview();
    const { batchXs } = dataLoader.getRandomTestBatch(testNoisyXs, testXs, 5);
    const denoised = model.predict(batchXs);

    const noisyArr    = batchXs.unstack();
    const denoisedArr = denoised.unstack();

    for(let i=0;i<noisyArr.length;i++){
        const cnvNoisy = document.createElement('canvas');
        dataLoader.draw28x28ToCanvas(noisyArr[i], cnvNoisy, 4);
        previewNoisyEl.appendChild(cnvNoisy);

        const cnvDen  = document.createElement('canvas');
        dataLoader.draw28x28ToCanvas(denoisedArr[i], cnvDen, 4);
        previewDenoisedEl.appendChild(cnvDen);
    }

    noisyArr.forEach(t=>t.dispose());
    denoisedArr.forEach(t=>t.dispose());
    batchXs.dispose();
    denoised.dispose();
};

// === Save / Load ===
document.getElementById('saveModelBtn').onclick = async ()=>{
    if(!model){ alert('No model'); return;}
    await model.save('downloads://mnist-denoiser');
    log('Model saved');
};

document.getElementById('loadModelBtn').onclick = async ()=>{
    const jsonFile = document.getElementById('modelJsonFile').files[0];
    const binFile  = document.getElementById('modelWeightsFile').files[0];
    if(!jsonFile||!binFile){ alert('Select model files'); return;}
    if(model) model.dispose();
    model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
    modelInfoEl.textContent='';
    model.summary(line=>modelInfoEl.textContent+=line+'\\n');
    log('Model loaded');
};

// === Reset ===
document.getElementById('resetBtn').onclick = ()=>{
    if(model){ model.dispose(); model=null; }
    if(dataLoader){ dataLoader.dispose(); }
    [trainXs,valXs,trainNoisyXs,valNoisyXs,testXs,testNoisyXs] = Array(6).fill(null);
    clearLogs(); clearPreview();
    dataStatus.innerHTML='<h3>Data Status</h3><p>No data loaded</p>';
    modelInfoEl.innerHTML='<h3>Model Info</h3><p>No model loaded</p>';
    tfvis.visor().close();
    log('Reset done');
};

// === Visor Toggle ===
document.getElementById('toggleVisorBtn').onclick = ()=> tfvis.visor().toggle();
