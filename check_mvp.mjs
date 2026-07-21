// Gate for the species-picker MVP: (1) first layer renders over Range; (2) click
// reads the TRUE value for the active layer from its COG; (3) switching species
// swaps the served layer and click then reads the NEW layer's values. Values are
// cross-checked at pixel centres against each layer's float source raster.
//   node check_mvp.mjs [port]
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";

const port = process.argv[2] || "8991";
const url = `http://127.0.0.1:${port}/index.html`;

// (id, source tif, bounds, a pixel (col,row) inside a high-suitability core)
const SIZE = 2000;
const CASES = [
  ["boreal_generalist", "out/boreal_generalist.tif", [-100, 44, -55, 62], 600, 900],
  ["atlantic_coastal", "out/atlantic_coastal.tif", [-70, 43, -52, 52], 1100, 900],
];
const centre = ([w, s, e, n], c, r) =>
  [w + (c + 0.5) * (e - w) / SIZE, n - (r + 0.5) * (n - s) / SIZE];
const truth = (tif, lng, lat) => parseFloat(execFileSync("gdallocationinfo",
  ["-valonly", "-geoloc", tif, String(lng), String(lat)], { encoding: "utf8" }).trim());

const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
let r206 = 0;
page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
page.on("pageerror", (e) => errs.push(String(e)));
page.on("response", (res) => { if (res.url().includes(".pmtiles") && res.status() === 206) r206++; });

await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__manifestLoaded === true, { timeout: 15000 });
await page.waitForFunction(() => window.__currentId, { timeout: 15000 });

// Combobox UX: clicking the field opens the FULL list; typing filters; clicking
// an option selects it (the datalist trap the user hit was "only shows current").
await page.click("#species");
await page.waitForSelector("#species-list li[data-id]", { timeout: 5000 });
const nAll = await page.$$eval("#species-list li[data-id]", (e) => e.length);
await page.fill("#species", "prair");
await page.dispatchEvent("#species", "input");
await page.waitForTimeout(200);
const nFilt = await page.$$eval("#species-list li[data-id]", (e) => e.length);
await page.click('#species-list li[data-id="prairie_specialist"]');
await page.waitForFunction(() => window.__currentId === "prairie_specialist", { timeout: 5000 });
const comboOk = nAll >= 4 && nFilt === 1;
console.log(`combobox: click shows ${nAll}, "prair" -> ${nFilt}, pick selects prairie ${comboOk ? "ok" : "FAIL"}`);

let allOk = comboOk;
for (const [id, tif, bounds, c, r] of CASES) {
  await page.evaluate((i) => window.__selectById(i), id);
  await page.waitForFunction((i) => window.__currentId === i, id, { timeout: 8000 });
  // let the new COG load
  await page.waitForTimeout(1500);
  const [lng, lat] = centre(bounds, c, r);
  let web, ok = false;
  for (let t = 0; t < 8; t++) {
    const res = await page.evaluate(([a, b]) => window.valueAt(a, b), [lng, lat]);
    if (!res.loading) { web = res.value ?? 0; break; }
    await page.waitForTimeout(400);
  }
  const src = truth(tif, lng, lat);
  ok = Math.abs((web ?? -1) - src) <= 1e-4 + 1e-6;
  allOk &&= ok;
  console.log(`${id}: web=${(web ?? -1).toFixed(4)} source=${src.toFixed(4)} ${ok ? "ok" : "MISMATCH"}`);
}
const r206after = r206;
await browser.close();
console.log(`pmtiles 206: ${r206after}, console errors: ${errs.length}`);
errs.slice(0, 4).forEach((e) => console.log("  ! " + e));
const pass = allOk && r206after > 0 && errs.length === 0;
console.log(pass ? "PASS: picker renders + per-layer click-inspect correct across a switch" : "FAIL");
process.exit(pass ? 0 : 1);
