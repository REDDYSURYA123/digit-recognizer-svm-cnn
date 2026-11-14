export const MNIST_IMAGES_SPRITE_URL =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
export const MNIST_LABELS_URL =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

export const MNIST_IMAGE_SIZE = 784; // 28 * 28
export const TRAIN_COUNT = 10000;
export const TEST_COUNT = 1000;
export const TEST_OFFSET = 55000;

export interface MnistBrowserDataset {
  trainImages: Float32Array; // 10,000 * 784 in [0, 1]
  trainLabels: Uint8Array; // 10,000 in 0..9
  testImages: Float32Array; // 1,000 * 784 in [0, 1]
  testLabels: Uint8Array; // 1,000 in 0..9
}

function loadCorsImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error('Failed to load MNIST sprite image from Google Cloud Storage CDN.'));
    img.src = url;
  });
}

function extractSpriteRows(
  img: HTMLImageElement,
  startRow: number,
  rowCount: number,
  chunkRows = 1000
): Float32Array {
  const result = new Float32Array(rowCount * MNIST_IMAGE_SIZE);
  const canvas = document.createElement('canvas');
  canvas.width = MNIST_IMAGE_SIZE;
  canvas.height = chunkRows;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Could not acquire 2D canvas context for MNIST decoding.');

  for (let offset = 0; offset < rowCount; offset += chunkRows) {
    const count = Math.min(chunkRows, rowCount - offset);
    if (canvas.height !== count) canvas.height = count;
    ctx.clearRect(0, 0, MNIST_IMAGE_SIZE, count);
    ctx.drawImage(
      img,
      0,
      startRow + offset,
      MNIST_IMAGE_SIZE,
      count,
      0,
      0,
      MNIST_IMAGE_SIZE,
      count
    );
    const { data } = ctx.getImageData(0, 0, MNIST_IMAGE_SIZE, count);
    const baseIdx = offset * MNIST_IMAGE_SIZE;
    for (let i = 0; i < count * MNIST_IMAGE_SIZE; i++) {
      result[baseIdx + i] = data[i * 4] / 255.0;
    }
  }
  return result;
}

function decodeOneHotLabels(oneHotBytes: Uint8Array, startRow: number, count: number): Uint8Array {
  const labels = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const base = (startRow + i) * 10;
    let bestClass = 0;
    let bestVal = -1;
    for (let c = 0; c < 10; c++) {
      const v = oneHotBytes[base + c];
      if (v > bestVal) {
        bestVal = v;
        bestClass = c;
      }
    }
    labels[i] = bestClass;
  }
  return labels;
}

export async function fetchMnistSubsetFromCdn(
  onStatus?: (msg: string, progress: number) => void
): Promise<MnistBrowserDataset> {
  onStatus?.('Fetching MNIST sprite & labels from public CDN...', 10);

  const [img, labelsResp] = await Promise.all([
    loadCorsImage(MNIST_IMAGES_SPRITE_URL),
    fetch(MNIST_LABELS_URL, { mode: 'cors' }),
  ]);

  if (!labelsResp.ok) {
    throw new Error(`Failed to fetch MNIST labels (HTTP ${labelsResp.status}).`);
  }

  onStatus?.('Decoding 65,000 one-hot labels...', 35);
  const labelsBuffer = await labelsResp.arrayBuffer();
  const labelsBytes = new Uint8Array(labelsBuffer);

  onStatus?.('Extracting 10,000 training digits (5k SVM / 10k CNN)...', 68);
  const trainImages = extractSpriteRows(img, 0, TRAIN_COUNT);
  const trainLabels = decodeOneHotLabels(labelsBytes, 0, TRAIN_COUNT);

  onStatus?.('Extracting 1,000 held-out test digits (28×28)...', 90);
  const testImages = extractSpriteRows(img, TEST_OFFSET, TEST_COUNT);
  const testLabels = decodeOneHotLabels(labelsBytes, TEST_OFFSET, TEST_COUNT);

  onStatus?.('MNIST ready (10,000 train / 1,000 test).', 100);
  return { trainImages, trainLabels, testImages, testLabels };
}

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface MnistProcessedDigit {
  pixels28: Float32Array;
  isEmpty: boolean;
  bbox: BoundingBox | null;
  centerOfMass: { x: number; y: number } | null;
}

export interface SegmentedDigitRegion {
  index: number;
  bbox: BoundingBox;
  pixels28: Float32Array;
  centerOfMass: { x: number; y: number } | null;
}

/**
 * Applies LeCun MNIST preprocessing to a bounded sub-region of a raw grid:
 * 1. Scale tight bounding box longest side to 20px preserving aspect ratio
 * 2. Center by center-of-mass inside a 28x28 frame, normalized to [0, 1]
 */
export function normalizeBoxToMnist28(
  rawGrid: Float32Array,
  gridWidth: number,
  bbox: BoundingBox
): MnistProcessedDigit {
  const { minX, minY, maxX, maxY } = bbox;
  if (maxX < minX || maxY < minY) {
    return {
      pixels28: new Float32Array(MNIST_IMAGE_SIZE),
      isEmpty: true,
      bbox: null,
      centerOfMass: null,
    };
  }

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20.0 / Math.max(bw, bh);
  const targetW = Math.max(1, Math.round(bw * scale));
  const targetH = Math.max(1, Math.round(bh * scale));

  const fitted28 = new Float32Array(MNIST_IMAGE_SIZE);
  const offsetX = Math.floor((28 - targetW) / 2);
  const offsetY = Math.floor((28 - targetH) / 2);

  for (let ty = 0; ty < targetH; ty++) {
    const sy = minY + ((ty + 0.5) / targetH) * bh - 0.5;
    const y0 = Math.max(minY, Math.min(maxY, Math.floor(sy)));
    const y1 = Math.max(minY, Math.min(maxY, y0 + 1));
    const wy = Math.max(0, Math.min(1, sy - y0));

    for (let tx = 0; tx < targetW; tx++) {
      const sx = minX + ((tx + 0.5) / targetW) * bw - 0.5;
      const x0 = Math.max(minX, Math.min(maxX, Math.floor(sx)));
      const x1 = Math.max(minX, Math.min(maxX, x0 + 1));
      const wx = Math.max(0, Math.min(1, sx - x0));

      const v00 = rawGrid[y0 * gridWidth + x0];
      const v10 = rawGrid[y0 * gridWidth + x1];
      const v01 = rawGrid[y1 * gridWidth + x0];
      const v11 = rawGrid[y1 * gridWidth + x1];
      const val = (v00 * (1 - wx) + v10 * wx) * (1 - wy) + (v01 * (1 - wx) + v11 * wx) * wy;

      const gx = offsetX + tx;
      const gy = offsetY + ty;
      if (gx >= 0 && gx < 28 && gy >= 0 && gy < 28) {
        fitted28[gy * 28 + gx] = val;
      }
    }
  }

  let mass = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = fitted28[y * 28 + x];
      mass += v;
      sumX += x * v;
      sumY += y * v;
    }
  }

  const pixels28 = new Float32Array(MNIST_IMAGE_SIZE);
  let comX = 14;
  let comY = 14;
  if (mass > 1e-6) {
    comX = sumX / mass;
    comY = sumY / mass;
    const shiftX = Math.round(13.5 - comX);
    const shiftY = Math.round(13.5 - comY);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const sx = x - shiftX;
        const sy = y - shiftY;
        if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
          pixels28[y * 28 + x] = fitted28[sy * 28 + sx];
        }
      }
    }
  }

  let maxVal = 0;
  for (let i = 0; i < MNIST_IMAGE_SIZE; i++) {
    if (pixels28[i] > maxVal) maxVal = pixels28[i];
  }
  if (maxVal > 0.15 && maxVal < 0.92) {
    const inv = 1.0 / maxVal;
    for (let i = 0; i < MNIST_IMAGE_SIZE; i++) {
      pixels28[i] = Math.min(1, pixels28[i] * inv);
    }
  }

  return {
    pixels28,
    isEmpty: false,
    bbox,
    centerOfMass: { x: comX, y: comY },
  };
}

/**
 * Segments a wide drawing grid into separate digits ordered left-to-right:
 * 1. Finds 8-connected components of active foreground pixels (> 0.08).
 * 2. Filters negligible specks (< 8 pixels).
 * 3. Sorts left-to-right and merges vertically stacked strokes (such as two-stroke
 *    '4', '5', or '7') whose horizontal x-intervals overlap >= 42%.
 * 4. Isolates each component mask and applies LeCun MNIST 28x28 normalization.
 */
export function segmentGridToDigits(
  rawGrid: Float32Array,
  width: number,
  height: number
): SegmentedDigitRegion[] {
  const total = width * height;
  const compMap = new Int32Array(total);
  const queue = new Int32Array(total);
  const rawBoxes: { compId: number; bbox: BoundingBox }[] = [];
  let nextCompId = 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const startIdx = y * width + x;
      if (compMap[startIdx] !== 0 || rawGrid[startIdx] <= 0.08) continue;

      const compId = nextCompId++;
      let head = 0;
      let tail = 0;
      queue[tail++] = startIdx;
      compMap[startIdx] = compId;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;

      while (head < tail) {
        const idx = queue[head++];
        const cx = idx % width;
        const cy = (idx - cx) / width;
        count++;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= width) continue;
            const nIdx = ny * width + nx;
            if (compMap[nIdx] === 0 && rawGrid[nIdx] > 0.08) {
              compMap[nIdx] = compId;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      if (count >= 8) {
        rawBoxes.push({ compId, bbox: { minX, minY, maxX, maxY } });
      }
    }
  }

  if (rawBoxes.length === 0) return [];

  rawBoxes.sort((a, b) => a.bbox.minX - b.bbox.minX);

  const merged: { bbox: BoundingBox; compIds: Set<number> }[] = [
    { bbox: rawBoxes[0].bbox, compIds: new Set([rawBoxes[0].compId]) },
  ];

  for (let i = 1; i < rawBoxes.length; i++) {
    const cur = rawBoxes[i];
    let mergedIntoExisting = false;

    for (let m = merged.length - 1; m >= 0; m--) {
      const prev = merged[m].bbox;
      const overlap = Math.min(prev.maxX, cur.bbox.maxX) - Math.max(prev.minX, cur.bbox.minX) + 1;
      const wPrev = prev.maxX - prev.minX + 1;
      const wCur = cur.bbox.maxX - cur.bbox.minX + 1;
      const minW = Math.max(1, Math.min(wPrev, wCur));

      if (overlap / minW >= 0.42) {
        merged[m].bbox = {
          minX: Math.min(prev.minX, cur.bbox.minX),
          minY: Math.min(prev.minY, cur.bbox.minY),
          maxX: Math.max(prev.maxX, cur.bbox.maxX),
          maxY: Math.max(prev.maxY, cur.bbox.maxY),
        };
        merged[m].compIds.add(cur.compId);
        mergedIntoExisting = true;
        break;
      }
    }

    if (!mergedIntoExisting) {
      merged.push({ bbox: cur.bbox, compIds: new Set([cur.compId]) });
    }
  }

  merged.sort((a, b) => a.bbox.minX - b.bbox.minX);

  return merged.map((group, idx) => {
    const { minX, minY, maxX, maxY } = group.bbox;
    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const isolated = new Float32Array(bw * bh);
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const gIdx = y * width + x;
        if (group.compIds.has(compMap[gIdx])) {
          isolated[(y - minY) * bw + (x - minX)] = rawGrid[gIdx];
        }
      }
    }
    const normalized = normalizeBoxToMnist28(isolated, bw, {
      minX: 0,
      minY: 0,
      maxX: bw - 1,
      maxY: bh - 1,
    });
    return {
      index: idx,
      bbox: group.bbox,
      pixels28: normalized.pixels28,
      centerOfMass: normalized.centerOfMass,
    };
  });
}

/**
 * Decodes an uploaded PNG/JPG image (single or multi-digit), auto-detects light vs dark
 * background from border pixels, inverts if light-background, and segments into digits.
 */
export async function segmentUploadedFileToDigits(
  file: File,
  targetWidth = 560,
  targetHeight = 280
): Promise<{ regions: SegmentedDigitRegion[]; previewDataUrl: string }> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Unsupported or corrupted image file.'));
      el.src = objectUrl;
    });

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Canvas 2D context unavailable.');

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, targetWidth, targetHeight);

    const scale = Math.min(targetWidth / img.width, targetHeight / img.height);
    const drawW = img.width * scale;
    const drawH = img.height * scale;
    ctx.drawImage(
      img,
      (targetWidth - drawW) / 2,
      (targetHeight - drawH) / 2,
      drawW,
      drawH
    );

    const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const { data } = imgData;
    const lum = new Float32Array(targetWidth * targetHeight);
    for (let i = 0; i < lum.length; i++) {
      const p = i * 4;
      lum[i] = (0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]) / 255.0;
    }

    let borderSum = 0;
    let borderCount = 0;
    for (let x = 0; x < targetWidth; x++) {
      borderSum += lum[x] + lum[(targetHeight - 1) * targetWidth + x];
      borderCount += 2;
    }
    for (let y = 0; y < targetHeight; y++) {
      borderSum += lum[y * targetWidth] + lum[y * targetWidth + targetWidth - 1];
      borderCount += 2;
    }
    const shouldInvert = borderSum / borderCount > 0.5;

    for (let i = 0; i < lum.length; i++) {
      const v = shouldInvert ? 1.0 - lum[i] : lum[i];
      const cleaned = v < 0.16 ? 0 : Math.min(1, (v - 0.16) / 0.84);
      lum[i] = cleaned;
      const gray = Math.round(cleaned * 255);
      data[i * 4] = gray;
      data[i * 4 + 1] = gray;
      data[i * 4 + 2] = gray;
      data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    const regions = segmentGridToDigits(lum, targetWidth, targetHeight);
    return { regions, previewDataUrl: canvas.toDataURL('image/png') };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}
