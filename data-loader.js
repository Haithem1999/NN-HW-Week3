// Extended MNISTDataLoader with noise utility for autoencoder training
class MNISTDataLoader {
    constructor() {
        this.trainData = null;
        this.testData  = null;
    }

    /* Reads CSV (label, 784 pixels) and returns {xs, ys, count} */
    async loadCSVFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.onload = e => {
                try {
                    const text   = e.target.result.trim();
                    const lines  = text.split(/\r?\n/).filter(l => l.trim() !== '');
                    const labels = [];
                    const pixels = [];

                    for (const line of lines) {
                        const values = line.split(',').map(Number);
                        if (values.length !== 785) continue;
                        labels.push(values[0]);
                        pixels.push(values.slice(1));
                    }
                    if (!labels.length) throw new Error('No valid rows found');

                    const xs = tf.tidy(() => tf.tensor2d(pixels).div(255).reshape([labels.length, 28, 28, 1]));
                    const ys = tf.tidy(() => tf.oneHot(labels, 10));
                    resolve({ xs, ys, count: labels.length });
                } catch (err) { reject(err); }
            };
            reader.readAsText(file);
        });
    }

    async loadTrainFromFiles(file) { this.trainData = await this.loadCSVFile(file); return this.trainData; }
    async loadTestFromFiles (file) { this.testData  = await this.loadCSVFile(file); return this.testData; }

    /* Splits xs/ys into train/val */
    splitTrainVal(xs, ys, valRatio = 0.1) {
        const total = xs.shape[0];
        const val   = Math.floor(total * valRatio);
        return tf.tidy(() => {
            const [trainXs, valXs] = tf.split(xs, [total - val, val]);
            const [trainYs, valYs] = tf.split(ys, [total - val, val]);
            return { trainXs, trainYs, valXs, valYs };
        });
    }

    /* Returns noisy tensor: xs + N(0,1)*factor clipped to [0,1] */
    addNoise(xs, noiseFactor = 0.5) {
        return tf.tidy(() => {
            const noise = tf.randomNormal(xs.shape, 0, 1).mul(noiseFactor);
            return xs.add(noise).clipByValue(0,1);
        });
    }

    /* k random samples from xs/ys */
    getRandomTestBatch(xs, ys, k = 5) {
        return tf.tidy(() => {
            const idxs = tf.util.createShuffledIndices(xs.shape[0]).slice(0, k);
            const tensorIdxs = tf.tensor1d(idxs, 'int32');
            const batchXs = tf.gather(xs, tensorIdxs);
            const batchYs = tf.gather(ys, tensorIdxs);
            tensorIdxs.dispose();
            return { batchXs, batchYs, indices: idxs };
        });
    }

    /* Draw a single 28x28 image to a canvas (grayscale) */
    draw28x28ToCanvas(imgTensor, canvas, scale=4){
        const ctx = canvas.getContext('2d');
        const [w,h] = [28,28];
        canvas.width  = w*scale;
        canvas.height = h*scale;
        const data = imgTensor.reshape([h,w]).mul(255).dataSync();
        const imageData = ctx.createImageData(w,h);
        for(let i=0;i<data.length;i++){
            const v = data[i];
            imageData.data[i*4+0]=v;
            imageData.data[i*4+1]=v;
            imageData.data[i*4+2]=v;
            imageData.data[i*4+3]=255;
        }
        const tmp = document.createElement('canvas');
        tmp.width=w; tmp.height=h;
        tmp.getContext('2d').putImageData(imageData,0,0);
        ctx.imageSmoothingEnabled=false;
        ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
        tmp.remove();
    }

    dispose(){
        const disposeData = d => { if(d){d.xs.dispose(); d.ys.dispose();} };
        disposeData(this.trainData);
        disposeData(this.testData);
        this.trainData=this.testData=null;
    }
}
