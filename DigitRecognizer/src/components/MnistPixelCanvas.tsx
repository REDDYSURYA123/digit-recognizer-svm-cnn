import { useEffect, useRef } from 'react';

interface MnistPixelCanvasProps {
  pixels: Float32Array;
  offset?: number;
  displaySize?: number;
  className?: string;
}

export function MnistPixelCanvas({
  pixels,
  offset = 0,
  displaySize = 84,
  className = '',
}: MnistPixelCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(1, pixels[offset + i] ?? 0));
      const lum = Math.round(v * 255);
      const p = i * 4;
      img.data[p] = lum;
      img.data[p + 1] = lum;
      img.data[p + 2] = lum;
      img.data[p + 3] = 255;
    }

    const offscreen = document.createElement('canvas');
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;
    offCtx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, displaySize, displaySize);
    ctx.drawImage(offscreen, 0, 0, displaySize, displaySize);
  }, [pixels, offset, displaySize]);

  return (
    <canvas
      ref={canvasRef}
      width={displaySize}
      height={displaySize}
      className={`block rounded border border-slate-200 dark:border-slate-700 bg-black ${className}`}
      style={{ width: displaySize, height: displaySize }}
    />
  );
}
