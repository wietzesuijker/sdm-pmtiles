// Browser gate: at 1 km cells the page's readout equals the Python reference
// (evidence_decay.apply_curve on decay/out/ursus-arctos_*.tif), for 1980-2010 and 2071-2100.
//   node decay/check_decay.mjs [port] [outdir]  (serve with: python serve.py 8992)
import { chromium } from "playwright";

const port = process.argv[2] || "8992";
// [column, row, stretch, period, expected readout prefix]
const cells = [
  [2600, 1800, 0.5, "1980-2010", "ensemble 23, 0 km from evidence, shown 23"],
  [3600, 3500, 0.5, "1980-2010", "ensemble 8, 110 km from evidence, shown 4"],
  [3600, 3500, 0, "1980-2010", "ensemble 8, 110 km from evidence, shown 0"],
  [3300, 3100, 0.5, "2071-2100", "ensemble 4, 0 km from evidence, shown 4"],
];
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1200, height: 1010 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/decay/index.html`);
await page.waitForFunction(() => document.getElementById("total").textContent !== "", null, { timeout: 60000 });
await page.selectOption("#sp", "ursus-arctos");
const m = await page.locator("#map").boundingBox();
let failed = 0;
for (const [c, r, s, period, want] of cells) {
  await page.selectOption("#period", period);
  await page.$eval("#scale", (el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }, String(s));
  // centre the cell (origin -4410000, 4310000, 1 km) at 250 m per screen pixel
  await page.evaluate(([x, y]) => { map.getView().setResolution(250); map.getView().setCenter([x, y]); },
    [-4410000 + (c + 0.5) * 1000, 4310000 - (r + 0.5) * 1000]);
  await page.waitForTimeout(5000);
  for (const dx of [1, 0]) { await page.mouse.move(m.x + m.width / 2 + dx, m.y + m.height / 2); await page.waitForTimeout(300); }
  const got = await page.textContent("#readout");
  const ok = got.startsWith(want);
  failed += !ok;
  console.log(ok ? "ok  " : "FAIL", `c${c} r${r} s${s} ${period}: ${got}`);
}
await page.screenshot({ path: `${process.argv[3] || "."}/decay_check.png` });
await browser.close();
if (errors.length) console.log("page errors:", errors);
if (failed || errors.length) process.exit(1);
