/* =========================================================================
   data-loader.js  –  robust CSV → tensors, noise injection, drawing utils
   ========================================================================= */

class MNISTDataLoader {
  constructor(){ this.train=null; this.test=null; }

  /* ---------- private CSV parser (skips headers & bad rows) ------------ */
  async #parseCsv(file){
    const txt = await file.text();
    let rows  = txt.split(/\r?\n/).filter(Boolean);

    /* Remove header row if first cell isn’t numeric */
    if (isNaN(parseInt(rows[0].split(',')[0]))) rows.shift();

    const pix = [];       // flattened, normalised pixels
    const lbl = [];

    rows.forEach((line,i)=>{
      line = line.replace('\uFEFF','');
      const v = line.split(',');
      if (v.length !== 785) return console.warn(`⚠️  bad row ${i+1}`);
      const lab = +v[0];
      if (Number.isNaN(lab)) return console.warn(`⚠️  NaN at row ${i+1}`);
      lbl.push(lab);
      for (let p=1; p<=784; p++) pix.push(+v[p] / 255);
    });

    if(!lbl.length) throw new Error('No valid data rows in CSV.');

    const xs = tf.tensor4d(pix, [lbl.length,28,28,1]);
    const ys = tf.oneHot(tf.tensor1d(lbl,'int32'),10);
    return { xs, ys, count:lbl.length };
  }

  async loadTrain(f){ this.train = await this.#parseCsv(f); return this.train; }
  async loadTest (f){ this.test  = await this.#parseCsv(f); return this.test;  }

  /* ---------- helpers -------------------------------------------------- */
  split(xs,ys,r=0.1){
    const v = Math.floor(xs.shape[0]*r);
    const [tX,vX] = tf.split(xs,[xs.shape[0]-v,v]);
    const [tY,vY] = tf.split(ys,[ys.shape[0]-v,v]);
    return { trainXs:tX, valXs:vX, trainYs:tY, valYs:vY };
  }

  /** Salt-and-pepper style noise (strong visual corruption) */
  addNoise(x,prob=0.5){
    return tf.tidy(()=>{
      const r = tf.randomUniform(x.shape);                 // 0-1
      const black = r.less(prob/2).cast('float32');        // set to 0
      const white = r.greaterEqual(prob/2).logicalAnd(r.less(prob)).cast('float32');
      return x.mul(r.greaterEqual(prob).cast('float32'))   // keep originals
              .add(white)                                  // add white pixels
              .clipByValue(0,1);
    });
  }

  /** Draw 28×28 tensor on canvas (scale*4) */
  draw(t, c, s=4){
    const [h,w]=[28,28]; c.width=w*s; c.height=h*s;
    const ctx=c.getContext('2d');
    const d=t.reshape([h,w]).mul(255).dataSync();
    const img=ctx.createImageData(w,h);
    for(let i=0;i<d.length;i++){ const v=d[i]; img.data.set([v,v,v,255],i*4); }
    const tmp=document.createElement('canvas'); tmp.width=w; tmp.height=h;
    tmp.getContext('2d').putImageData(img,0,0);
    ctx.imageSmoothingEnabled=false; ctx.drawImage(tmp,0,0,c.width,c.height);
    tmp.remove();
  }
}
