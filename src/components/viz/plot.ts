/** Shared helpers for the small hand-rolled SVG plots. */

export interface Frame {
  w: number;
  h: number;
  pad: { t: number; r: number; b: number; l: number };
}

export function makeScales(f: Frame, xDomain: [number, number], yDomain: [number, number]) {
  const x0 = f.pad.l;
  const x1 = f.w - f.pad.r;
  const y0 = f.h - f.pad.b;
  const y1 = f.pad.t;
  const sx = (v: number) =>
    x0 + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * (x1 - x0);
  const sy = (v: number) =>
    y0 + ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * (y1 - y0);
  return { sx, sy, x0, x1, y0, y1 };
}

export function linePath(
  pts: [number, number][],
  sx: (v: number) => number,
  sy: (v: number) => number,
): string {
  return pts
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`)
    .join(" ");
}

export function areaPath(
  pts: [number, number][],
  sx: (v: number) => number,
  sy: (v: number) => number,
  baseline: number,
): string {
  if (pts.length === 0) return "";
  const top = linePath(pts, sx, sy);
  const last = pts[pts.length - 1];
  const first = pts[0];
  return `${top} L${sx(last[0]).toFixed(2)},${baseline.toFixed(2)} L${sx(first[0]).toFixed(2)},${baseline.toFixed(2)} Z`;
}

export function range(n: number, from = 0, to = 1): number[] {
  return Array.from({ length: n }, (_, i) => from + ((to - from) * i) / (n - 1));
}

/** Deterministic PRNG so every visitor sees the same "random" data. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller on a seeded uniform source. */
export function gaussian(rnd: () => number, mean = 0, sd = 1): number {
  const u = Math.max(rnd(), 1e-9);
  const v = rnd();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const PLOT_COLORS = {
  bias: "#f59e0b",
  variance: "#38bdf8",
  total: "#818cf8",
  train: "#10b981",
  val: "#f43f5e",
  grid: "currentColor",
};
