/*  MNISTDataLoader  –  parses CSV, adds noise, utility drawing helpers */

class MNISTDataLoader {
  constructor() {
    this.train = null;
    this.test  = null;
  }

  /* Parse a single CSV file → { xs:4-D, ys:2-D, count } */
  async #parseCsv(file) {
    const txt = await file.text();
    const lines = txt.trim().split(/\r?\n/).filter(l => l);
    const n = lines.length;

    const pix = new Float32Array(n * 28 * 28);
    const lbl = new Uint8Array(n);

    let offset = 0;
    lines.forEach((row,i) => {
      const vals = row.split(',').map(Number);
      lbl[i] = vals[0];
      for (let p=1; p<=784; ++p) pix[offset++] = vals[p] / 255;   // normalize
    });

    const xs = tf.tensor4d(pix, [n,28,28,1]);
    const ys = tf.oneHot(tf.tensor1d(lbl,'int32'), 10);
    return { xs, ys, count:n };
  }

  async loadTrain(file){ this.train = await this.#parseCsv(file); return this.train; }
  async loadTest (file){ this.test  = await this.#parseCsv(file); return this.test;  }

  /** Split into train/val tensors */
  split(xs, ys, ratio=0.1){
    const val = Math.floor(xs.shape[0]*ratio);
    const [trainXs, valXs] = tf.split(xs, [xs.shape[0]-val, val]);
    const [trainYs, valYs] = tf.split(ys, [ys.shape[0]-val, val]);
    return { trainXs, valXs, trainYs, valYs };
  }

  /** Add Gaussian noise and clip to [0,1] */
  addNoise(x, factor=0.5){
    return tf.tidy(()=> x.add(tf.randomNormal(x.shape,0,1).mul(factor)).clipByValue(0,1));
  }

  /** Draw a single 28×28 tensor to a canvas (scale *4 for visibility) */
  drawTensorToCanvas(t, canvas, scale=4){
    const [h,w] = [28,28];
    canvas.width  = w*scale;
    canvas.height = h*scale;

    const ctx = canvas.getContext('2d');
    const data = t.reshape([h,w]).mul(255).dataSync();
    const img  = ctx.createImageData(w,h);
    for(let i=0;i<data.length;i++){
      const v = data[i];
      img.data[i*4+0]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255;
    }
    // disable smoothing when scaling
    const tmp = document.createElement('canvas');
    tmp.width=w; tmp.height=h;
    tmp.getContext('2d').putImageData(img,0,0);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
    tmp.remove();
  }
}
