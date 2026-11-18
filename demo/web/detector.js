// Small CNN forward pass in plain JavaScript. Batch normalisation is already
// folded into the convolution weights (see ../export_weights.py), so the
// network is: [conv3x3 + ReLU] x2, max-pool, three times; global average
// pooling; dense + ReLU; dense + sigmoid. Input: 32x32 RGB, values 0-255.

function conv3x3Relu(x, h, w, cin, weights, offset, cout) {
  const k = weights.subarray(offset, offset + 9 * cin * cout);   // HWIO
  const b = weights.subarray(offset + 9 * cin * cout, offset + 9 * cin * cout + cout);
  const out = new Float32Array(h * w * cout);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      const o = (y * w + xx) * cout;
      for (let c = 0; c < cout; c++) out[o + c] = b[c];
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xs = xx + dx;
          if (xs < 0 || xs >= w) continue;
          const i = (yy * w + xs) * cin;
          const kb = ((dy + 1) * 3 + (dx + 1)) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const v = x[i + ci];
            if (v === 0) continue;
            const kr = kb + ci * cout;
            for (let c = 0; c < cout; c++) out[o + c] += v * k[kr + c];
          }
        }
      }
      for (let c = 0; c < cout; c++) if (out[o + c] < 0) out[o + c] = 0;
    }
  }
  return out;
}

function maxPool2(x, h, w, c) {
  const oh = h >> 1, ow = w >> 1, out = new Float32Array(oh * ow * c);
  for (let y = 0; y < oh; y++)
    for (let xx = 0; xx < ow; xx++)
      for (let ch = 0; ch < c; ch++) {
        const a = x[((2 * y) * w + 2 * xx) * c + ch], b = x[((2 * y) * w + 2 * xx + 1) * c + ch];
        const d = x[((2 * y + 1) * w + 2 * xx) * c + ch], e = x[((2 * y + 1) * w + 2 * xx + 1) * c + ch];
        out[(y * ow + xx) * c + ch] = Math.max(a, b, d, e);
      }
  return out;
}

function predict(model, rgb) {
  // rgb: Uint8ClampedArray or array of 32*32*3 values in [0, 255], HWC order.
  const W = model.weights;
  let x = new Float32Array(32 * 32 * 3);
  for (let i = 0; i < x.length; i++) x[i] = rgb[i] / 255;
  let h = 32, w = 32, c = 3, off = 0, vec = null;
  for (const layer of model.layout) {
    if (layer.type === "conv") {
      x = conv3x3Relu(x, h, w, layer.cin, W, off, layer.cout);
      off += 9 * layer.cin * layer.cout + layer.cout;
      c = layer.cout;
    } else if (layer.type === "pool") {
      x = maxPool2(x, h, w, c); h >>= 1; w >>= 1;
    } else if (layer.type === "dense") {
      if (vec === null) {            // global average pooling before the first dense layer
        vec = new Float32Array(c);
        for (let i = 0; i < h * w; i++) for (let ch = 0; ch < c; ch++) vec[ch] += x[i * c + ch];
        for (let ch = 0; ch < c; ch++) vec[ch] /= h * w;
      }
      const k = W.subarray(off, off + layer.nin * layer.nout);
      const b = W.subarray(off + layer.nin * layer.nout, off + layer.nin * layer.nout + layer.nout);
      off += layer.nin * layer.nout + layer.nout;
      const out = new Float32Array(layer.nout);
      for (let o = 0; o < layer.nout; o++) {
        let s = b[o];
        for (let i = 0; i < layer.nin; i++) s += vec[i] * k[i * layer.nout + o];
        out[o] = layer.activation === "relu" ? Math.max(0, s) : 1 / (1 + Math.exp(-s));
      }
      vec = out;
    }
  }
  return vec[0];   // probability that the image is AI-generated
}

if (typeof module !== "undefined") module.exports = { predict };
