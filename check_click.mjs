// Gate for click-to-inspect: the value read client-side from the COG at a given
// lon/lat must match the source raster's value there. Proves the map surfaces
// TRUE suitability values, not colours decoded from the display tiles.
//   node check_click.mjs [port]
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";

const port = process.argv[2] || "8991";
const url = `http://127.0.0.1:${port}/index.html`;

// Ground truth straight from the source GeoTIFF via gdallocationinfo.
function truth(lng, lat) {
  const out = execFileSync("gdallocationinfo",
    ["-valonly", "-geoloc", "out/sdm_synth.tif", String(lng), String(lat)],
    { encoding: "utf8" });
  return parseFloat(out.trim());
}

// Sample at exact pixel centres (from the fixture's extent/size) so geotiff.js,
// GDAL and rasterio all resolve to the same pixel — no boundary-rounding ambiguity.
const W = 3000, H = 3000, WEST = -95, NORTH = 60, PX = 40 / W, PY = 18 / H;
const center = (c, r) => [WEST + (c + 0.5) * PX, NORTH - (r + 0.5) * PY];
const points = [center(900, 1050), center(1000, 1200), center(1300, 1000), center(2200, 2000)];

const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
page.on("pageerror", (e) => errs.push(String(e)));
await page.goto(url, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => typeof window.valueAt === "function", { timeout: 15000 });

let allOk = true;
for (const [lng, lat] of points) {
  const r = await page.evaluate(([a, b]) => window.valueAt(a, b), [lng, lat]);
  const web = r.value ?? 0;             // nodata/absence -> 0
  const src = truth(lng, lat);
  const ok = Math.abs(web - src) <= 1e-4 + 1e-6;  // uint16 scale quantum
  allOk &&= ok;
  console.log(`(${lat}, ${lng})  web=${web.toFixed(4)}  source=${src.toFixed(4)}  ` +
    `${ok ? "ok" : "MISMATCH"}`);
}
await browser.close();
console.log(`console errors: ${errs.length}`);
const pass = allOk && errs.length === 0;
console.log(pass ? "PASS: clicked values match the source SDM" : "FAIL");
process.exit(pass ? 0 : 1);
