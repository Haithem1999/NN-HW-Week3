/* =========================================================================
   data-loader.js  –  robust CSV → tensors + noise & drawing helpers
   ========================================================================= */

class MNISTDataLoader {
  constructor(){ this.train=null; this.test=null; }

  /* ---------- private robust CSV parser -------------------------------- */
  async #parseCsv(file){
    const text = await file.text();
    let rows = text.split(/\r?\n/).filter(Boolean);

    /* Remove header if first cell isn’t numeric */
    if (isNaN(parseInt(rows[0].split(',')[0]))) rows.shift();

    const images = [];   // flat pixel values (normalized 0-1)
    const labels = [];

    rows.forEach((line,i)=>{
      line = line.replace('\uFEFF','');                  // strip UTF-8 BOM
      const vals = line.split(',');
      if (vals.length !== 785){
        console.warn(`⚠️ Skipping line ${i+1} (length ${vals.length})`);
        return;
      }
      const label = Number(vals[0]);
      if (Number.isNaN(label)){
        console.warn(`⚠️ Skipping line ${i+1} (NaN label)`);
        return;
      }
      labels.push(label);
      for(let p=1;p<=784;p++) images.push(Number(vals[p])/255);
    });

    if(!labels.length) throw new Error('No valid rows found in CSV.');

    const xs = tf.tensor4d(images, [labels.length,28,28,1]);
    const ys = tf.oneHot(tf.tensor1d(labels,'int32'),10);
    return { xs, ys, count:labels.length };
  }

  /* public loaders */
  async loadTrain(file){ this.train = await this.#parseCsv(file); return this.train; }
  async loadTest (file){ this.test  = await this.#parseCsv(file); return this.test;  }

  /* helpers -------------------------------------------------------------- */
  split(xs,ys,ratio=0.1){
    const val = Math.floor(xs.shape[0]*ratio);
    const [tX,vX] = tf.split(xs,[xs.shape[0]-val,val]);
    const [tY,vY] = tf.split(ys,[ys.shape[0]-val,val]);
    return { trainXs:tX, valXs:vX, trainYs:tY, valYs:vY };
  }

  addNoise(x,f=0.5){
    return tf.tidy(()=> x.add(tf.randomNormal(x.shape,0,1).mul(f)).clipByValue(0,1));
  }

  drawTensorToCanvas(t,canvas,scale=4){
    const [h,w]=[28,28];
    canvas.width=w*scale; canvas.height=h*scale;
    const ctx=canvas.getContext('2d');
    const data=t.reshape([h,w]).mul(255).dataSync();
    const img=ctx.createImageData(w,h);
    for(let i=0;i<data.length;i++){
      const v=data[i]; img.data.set([v,v,v,255],i*4);
    }
    const tmp=document.createElement('canvas');
    tmp.width=w; tmp.height=h;
    tmp.getContext('2d').putImageData(img,0,0);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
    tmp.remove();
  }
}
