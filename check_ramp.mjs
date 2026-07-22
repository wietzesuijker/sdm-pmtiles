// Regression gate for the GPU color-relief ramp. The failure this guards against is
// SILENT: tiles render, no console/map errors, every other gate passes — but the
// colour-ramp stops get mis-packed (e.g. greenFactor/blueFactor 0 -> divide-by-zero in
// MapLibre's stop packer) and the whole layer saturates to its top colour. A screenshot
// colour histogram is the only thing that catches it: a real suitability surface must
// show the LOW viridis bands (purple/blue/teal), not just yellow.
import { chromium } from "playwright";
const port = process.argv[2] || "8991";
const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("pageerror", (e) => errs.push(String(e)));

// prairie_specialist is a Gaussian: mostly low-mid suitability, a small high core.
await page.goto(`http://127.0.0.1:${port}/index.html?species=prairie_specialist`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__currentId === "prairie_specialist", { timeout: 15000 });
await page.waitForTimeout(2500);
await page.screenshot({ path: "out/ramp_gate.png" });
await browser.close();

// Classify saturated pixels to nearest viridis anchor (no WebGL readback: the composited
// screenshot is the source of truth for what the user actually sees).
const { execFileSync } = await import("node:child_process");
const py = `
import numpy as np, sys
from PIL import Image
im = np.asarray(Image.open("out/ramp_gate.png").convert("RGB")).astype(int)
r,g,b = im[...,0],im[...,1],im[...,2]
sat = (np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)) > 40
pix = im[sat]
anchors = {"purple":(68,1,84),"blue":(59,82,139),"teal":(33,145,140),"green":(94,201,98),"yellow":(253,231,37)}
names = list(anchors); ac = np.array([anchors[k] for k in names])
idx = ((pix[:,None,:]-ac[None])**2).sum(2).argmin(1)
tot = max(pix.shape[0], 1)
frac = {k: float((idx==i).sum())/tot for i,k in enumerate(names)}
low = frac["purple"]+frac["blue"]+frac["teal"]
print("colored_px", tot)
for k in names: print(k, round(frac[k],4))
print("low_bands", round(low,4))
# A correct gradient shows substantial low bands and is NOT dominated by yellow.
ok = tot > 5000 and low > 0.30 and frac["yellow"] < 0.50
print("VERDICT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
`;
let code = 0;
try {
  const out = execFileSync(".venv/bin/python", ["-c", py], { encoding: "utf8" });
  console.log(out.trim());
} catch (e) {
  console.log((e.stdout || "").trim());
  console.log((e.stderr || "").trim());
  code = 1;
}
if (errs.length) { console.log(`pageerrors: ${errs.length}`); code = 1; }
console.log(code === 0 ? "PASS: color-relief renders a real viridis gradient (not saturated)" : "FAIL: ramp saturated or errored");
process.exit(code);
