/* Helper functions for loading & manipulating MNIST CSV data.
   CSV row: <label>,<p0>,<p1>,...,<p783>  (784 pixels, 0-255)            */

const PIXELS = 28 * 28;
const LABELS = 10;

/**
 * Reads a CSV file from an <input type="file"> element and converts it
 * into image & label tensors suitable for CNN input.
 * Returns: { xs: tf.Tensor4D, ys: tf.Tensor2D }
 */
async function loadCsvAsTensors(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = err => reject(err);
    reader.onload = () => {
      try {
        const text = reader.result.trim();
        const lines = text.split(/\r?\n/);
        const numSamples = lines.length;

        // Typed arrays for efficiency
        const images = new Float32Array(numSamples * PIXELS);
        const labels = new Uint8Array(numSamples);

        let imgOffset = 0;
        lines.forEach((line, idx) => {
          if (!line) return; // skip empty
          const parts = line.split(',').map(Number);
          labels[idx] = parts[0];
          for (let i = 1; i <= PIXELS; i++) {
            images[imgOffset++] = parts[i] / 255; // normalize 0-1
          }
        });

        /* Build tensors:
           xs -> [N,28,28,1]   ys -> one-hot [N,10]                     */
        const xs = tf.tensor4d(images, [numSamples, 28, 28, 1]);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), LABELS);
        resolve({ xs, ys });
      } catch (e) { reject(e); }
    };
    reader.readAsText(file);
  });
}

// Public API --------------------------------------------------------------
async function loadTrainFromFiles(file) { return loadCsvAsTensors(file); }
async function loadTestFromFiles (file) { return loadCsvAsTensors(file); }

/** Splits tensors into train/validation sets. */
function splitTrainVal(xs, ys, valRatio = 0.1) {
  const total   = xs.shape[0];
  const valSize = Math.floor(total * valRatio);
  const trainSize = total - valSize;

  const [trainXs, valXs] = tf.split(xs, [trainSize, valSize]);
  const [trainYs, valYs] = tf.split(ys, [trainSize, valSize]);
  return { trainXs, trainYs, valXs, valYs };
}

/** Returns k random samples from test set for preview. */
function getRandomTestBatch(xs, ys, k = 5) {
  const total = xs.shape[0];
  const idxs = tf.util.createShuffledIndices(total).slice(0, k);
  const indices = tf.tensor1d(idxs, 'int32');
  const batchXs = tf.gather(xs, indices);
  const batchYs = tf.gather(ys, indices);
  indices.dispose();
  return { batchXs, batchYs };
}

/** Draws a single 28x28 tensor to an HTMLCanvasElement. */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
  const [h, w] = [28, 28];
  canvas.width  = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);
  const data = tensor.dataSync();

  for (let i = 0; i < data.length; i++) {
    const val = data[i] * 255;
    imageData.data[i * 4 + 0] = val; // R
    imageData.data[i * 4 + 1] = val; // G
    imageData.data[i * 4 + 2] = val; // B
    imageData.data[i * 4 + 3] = 255; // A
  }
  // scale the image for visibility
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);
  tmpCanvas.remove();
}
