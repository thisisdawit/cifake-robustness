const $ = (id) => document.getElementById(id);
const state = { models: {}, ref: null, source: null };

async function loadModels() {
  const manifest = await (await fetch("model.json")).json();
  for (const [key, m] of Object.entries(manifest)) {
    const buf = await (await fetch(m.file)).arrayBuffer();
    state.models[key] = { layout: m.layout, weights: new Float32Array(buf) };
  }
  state.ref = await (await fetch("spectrum_reference.json")).json();
  drawSpectrum(null);
}

function canvas(n) { const c = document.createElement("canvas"); c.width = c.height = n; return c; }

function jpeg(c, quality) {
  return new Promise((resolve) => c.toBlob(async (blob) => {
    const bmp = await createImageBitmap(blob);
    const out = canvas(c.width); out.getContext("2d").drawImage(bmp, 0, 0); resolve(out);
  }, "image/jpeg", quality));
}

async function toBenchmark(img) {
  // Centre crop, downscale to 32x32, JPEG quality 75: the format CIFAKE is stored in.
  const s = Math.min(img.width, img.height), c = canvas(32), g = c.getContext("2d");
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, 32, 32);
  return jpeg(c, 0.75);
}

function blurred(src, sigma) {
  // Mirror the image on all sides first, so the blur has real pixels at the border
  // and the result stays fully opaque.
  const big = canvas(96), b = big.getContext("2d");
  for (const [sx, tx] of [[-1, 32], [1, 32], [-1, 96]])
    for (const [sy, ty] of [[-1, 32], [1, 32], [-1, 96]]) { b.setTransform(sx, 0, 0, sy, tx, ty); b.drawImage(src, 0, 0); }
  const c = canvas(32), g = c.getContext("2d");
  g.filter = `blur(${sigma}px)`; g.drawImage(big, -32, -32); return c;
}

async function degrade(base, scale, sigma, sharpen, quality) {
  let c = base;
  if (sigma > 0) c = blurred(c, sigma);
  if (scale < 1) {
    const n = Math.max(1, Math.round(32 * scale)), small = canvas(n), sg = small.getContext("2d");
    sg.imageSmoothingQuality = "high"; sg.drawImage(c, 0, 0, 32, 32, 0, 0, n, n);
    const up = canvas(32), ug = up.getContext("2d");
    ug.imageSmoothingQuality = "high"; ug.drawImage(small, 0, 0, n, n, 0, 0, 32, 32); c = up;
  }
  if (sharpen) {
    const a = c.getContext("2d").getImageData(0, 0, 32, 32), b = blurred(c, 1).getContext("2d").getImageData(0, 0, 32, 32);
    for (let i = 0; i < a.data.length; i += 4)
      for (let k = 0; k < 3; k++) a.data[i + k] = a.data[i + k] + 1.5 * (a.data[i + k] - b.data[i + k]);
    const out = canvas(32); out.getContext("2d").putImageData(a, 0, 0); c = out;
  }
  if (quality < 100) c = await jpeg(c, quality / 100);
  return c;
}

function rgbOf(c) {
  const d = c.getContext("2d").getImageData(0, 0, 32, 32).data, rgb = new Uint8ClampedArray(32 * 32 * 3);
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
  return rgb;
}

function radialSpectrum(rgb) {
  const N = 32, y = new Float64Array(N * N);
  let mean = 0;
  for (let i = 0; i < N * N; i++) { y[i] = 0.299 * rgb[3 * i] + 0.587 * rgb[3 * i + 1] + 0.114 * rgb[3 * i + 2]; mean += y[i]; }
  mean /= N * N;
  const han = (n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) y[r * N + c] = (y[r * N + c] - mean) * han(r) * han(c);
  const sum = new Float64Array(17), cnt = new Float64Array(17);
  for (let u = 0; u < N; u++) for (let v = 0; v < N; v++) {
    let re = 0, im = 0;
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const a = (-2 * Math.PI * (u * r + v * c)) / N; re += y[r * N + c] * Math.cos(a); im += y[r * N + c] * Math.sin(a);
    }
    const su = (u + N / 2) % N, sv = (v + N / 2) % N;
    const rad = Math.round(Math.hypot(su - N / 2, sv - N / 2));
    if (rad >= 1 && rad <= 16) { sum[rad] += re * re + im * im; cnt[rad]++; }
  }
  return Array.from({ length: 16 }, (_, k) => Math.log10(sum[k + 1] / cnt[k + 1] + 1e-6));
}

function drawSpectrum(values) {
  const svg = $("spec"), W = 440, H = 220, L = 38, R = 12, T = 10, B = 34;
  const all = state.ref.real.concat(state.ref.generated, values || []);
  const lo = Math.floor(Math.min(...all)), hi = Math.ceil(Math.max(...all));
  const X = (k) => L + (k / 15) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const path = (arr) => arr.map((v, k) => `${k ? "L" : "M"}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  let s = "";
  for (let v = lo; v <= hi; v++) s += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eeece8"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">spatial frequency → (low to fine detail)</text>`;
  s += `<path d="${path(state.ref.real)}" fill="none" stroke="#2a78d6" stroke-width="2"/>`;
  s += `<path d="${path(state.ref.generated)}" fill="none" stroke="#eb6834" stroke-width="2"/>`;
  if (values) s += `<path d="${path(values)}" fill="none" stroke="#141414" stroke-width="2.2" stroke-dasharray="5 3"/>`;
  s += `<text x="${W - R}" y="${T + 12}" text-anchor="end" style="fill:#2a78d6">average photograph</text>`;
  s += `<text x="${W - R}" y="${T + 27}" text-anchor="end" style="fill:#eb6834">average generated image</text>`;
  if (values) s += `<text x="${W - R}" y="${T + 42}" text-anchor="end" style="fill:#141414">this image</text>`;
  svg.innerHTML = s;
}

function show(key, p) {
  const ai = p >= 0.5, conf = ai ? p : 1 - p;
  $(`c-${key}`).textContent = `${ai ? "AI-generated" : "Photograph"} · model score ${(conf * 100).toFixed(1)}%`;
  $(`b-${key}`).style.width = `${(p * 100).toFixed(1)}%`;
}

let busy = false, again = false;
async function update() {
  const scale = +$("scale").value, sigma = +$("blur").value, q = +$("q").value, sharpen = $("sharpen").checked;
  $("o-scale").textContent = `${scale.toFixed(2)}×`;
  $("o-blur").textContent = sigma > 0 ? `σ = ${sigma.toFixed(1)}` : "off";
  $("o-q").textContent = q < 100 ? q : "off";
  if (!state.source || !state.models.clean) return;
  if (busy) { again = true; return; }
  busy = true;
  const c = await degrade(state.source, scale, sigma, sharpen, q);
  $("preview").getContext("2d").drawImage(c, 0, 0);
  const rgb = rgbOf(c);
  show("clean", predict(state.models.clean, rgb));
  show("hard", predict(state.models.hardened, rgb));
  drawSpectrum(radialSpectrum(rgb));
  busy = false;
  if (again) { again = false; update(); }
}

async function useFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = await createImageBitmap(file);
  state.source = await toBenchmark(img);
  update();
}

const presets = { clean: [1, 0, 100, false], upload: [0.5, 0, 50, false], blur: [1, 1, 100, false], sharpen: [1, 0, 100, true] };
document.querySelectorAll(".presets button").forEach((b) => b.addEventListener("click", () => {
  const [s, bl, q, sh] = presets[b.dataset.p];
  $("scale").value = s; $("blur").value = bl; $("q").value = q; $("sharpen").checked = sh; update();
}));
["scale", "blur", "q", "sharpen"].forEach((id) => $(id).addEventListener("input", update));
$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => useFile(e.target.files[0]));
$("drop").addEventListener("dragover", (e) => { e.preventDefault(); $("drop").classList.add("over"); });
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", (e) => { e.preventDefault(); $("drop").classList.remove("over"); useFile(e.dataTransfer.files[0]); });
window.addEventListener("paste", (e) => { const f = [...e.clipboardData.files][0]; if (f) useFile(f); });
loadModels().then(update);
