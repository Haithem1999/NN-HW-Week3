/* =========================================================================
   data-loader.js – CSV → tensors, noise injection, reliable draw()
   ========================================================================= */

class MNISTDataLoader {
  constructor(){ this.train=null; this.test=null; }

  /* robust CSV reader --------------------------------------------------- */
  async #parseCsv(file){
    const txt = await file.text();
    let rows  = txt.split(/\r?\n/).filter(Boolean);
    if(isNaN(+rows[0].split(',')[0])) rows.shift();                 // drop header
    const pix=[], lab=[];
    rows.forEach((line,i)=>{
      line=line.replace('\uFEFF','');
      const v=line.split(','); if(v.length!==785) return;
      const y=+v[0]; if(Number.isNaN(y)) return;
      lab.push(y); for(let p=1;p<=784;p++) pix.push(+v[p]/255);
    });
    if(!lab.length) throw new Error('No valid rows in CSV');
    return {
      xs: tf.tensor4d(pix,[lab.length,28,28,1]),
      ys: tf.oneHot(tf.tensor1d(lab,'int32'),10),
      count:lab.length
    };
  }
  async loadTrain(f){ this.train=await this.#parseCsv(f); return this.train; }
  async loadTest (f){ this.test =await this.#parseCsv(f); return this.test;  }

  split(xs,ys,r=0.1){
    const v=Math.floor(xs.shape[0]*r);
    const[tX,vX]=tf.split(xs,[xs.shape[0]-v,v]);
    const[tY,vY]=tf.split(ys,[ys.shape[0]-v,v]);
    return{trainXs:tX,valXs:vX,trainYs:tY,valYs:vY};
  }

  /**
   * Step 1: Add salt-and-pepper noise to images
   * @param {tf.Tensor} x - Clean images tensor [batch, 28, 28, 1]
   * @param {number} p - Noise probability (0.5 = 50% of pixels affected)
   * @returns {tf.Tensor} Noisy images tensor
   */
  addNoise(x,p=0.5){
    return tf.tidy(()=>{
      const r=tf.randomUniform(x.shape);
      // Black noise: r < p/2 → set pixel to 0
      const black=r.less(p/2).cast('float32');
      // White noise: p/2 <= r < p → set pixel to 1
      const white=r.greaterEqual(p/2).logicalAnd(r.less(p)).cast('float32');
      // Keep original pixels where r >= p
      return x.mul(r.greaterEqual(p).cast('float32')).add(white).clipByValue(0,1);
    });
  }

  /**
   * Reliable canvas render via tf.browser.toPixels (nearest-neighbour scaled)
   * @param {tf.Tensor} t - Image tensor [28, 28, 1]
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {number} s - Scale factor (default 4x for 112×112 display)
   */
  async draw(t,canvas,s=4){
    const img=t.reshape([28,28,1]);                            // [28,28,1]
    const big=tf.image.resizeNearestNeighbor(img,[28*s,28*s]); // upscale
    await tf.browser.toPixels(big.squeeze(),canvas);
    img.dispose(); big.dispose();
  }
}
