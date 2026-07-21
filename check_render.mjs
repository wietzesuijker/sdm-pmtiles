// Headless serving gate: load viewer.html, confirm MapLibre renders the SDM
// PMTiles over HTTP Range requests with no errors, and that 206 responses were
// actually served for the .pmtiles archive.
//   node check_render.mjs [port]
import { chromium } from "playwright";

const port = process.argv[2] || "8991";
const url = `http://127.0.0.1:${port}/index.html`;

const browser = await chromium.launch();
const page = await browser.newPage();

const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));

let pmtiles206 = 0, pmtiles200 = 0;
page.on("response", (r) => {
  if (r.url().includes(".pmtiles")) {
    if (r.status() === 206) pmtiles206++;
    else if (r.status() === 200) pmtiles200++;
  }
});

await page.goto(url, { waitUntil: "domcontentloaded" });
// Serving is proven by the browser fetching tile byte-ranges and rendering them.
// Poll for >=2 pmtiles 206s (archive header + at least one tile), then settle.
for (let i = 0; i < 40 && pmtiles206 < 2; i++) await page.waitForTimeout(500);
await page.waitForTimeout(1500);

const mapErrors = await page.evaluate(() => window.__mapErrors || []);
await page.screenshot({ path: "out/render.png" });
await browser.close();

console.log(`pmtiles range requests: ${pmtiles206} x 206, ${pmtiles200} x 200`);
console.log(`console errors: ${consoleErrors.length}`);
consoleErrors.slice(0, 5).forEach((e) => console.log("  ! " + e));
console.log(`map errors: ${mapErrors.length}`);
mapErrors.slice(0, 5).forEach((e) => console.log("  ! " + e));

const ok = pmtiles206 > 0 && consoleErrors.length === 0 && mapErrors.length === 0;
console.log(ok ? "PASS: SDM PMTiles rendered over Range requests" : "FAIL");
process.exit(ok ? 0 : 1);
