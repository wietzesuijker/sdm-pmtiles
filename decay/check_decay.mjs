// Browser gate: the page's recomputed clip equals the Python reference on the
// same overview pixels, at stretch 1 and 0.5. Writes values for compare.
//   node decay/check_decay.mjs [port] [outdir]  (serve with: python serve.py 8992)
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";

const port = process.argv[2] || "8992";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 1300 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/decay/index.html`);
await page.waitForFunction(() => document.getElementById("d50").value !== "", null, { timeout: 60000 });
const slug = await page.$eval("#sp", (s) => s.value);
const out = { slug, errors };
for (const s of ["1", "0.5"]) {
  await page.$eval("#scale", (el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }, s);
  out[s] = await page.evaluate(() => {
    const c = document.getElementById("map"), d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    const nz = []; for (let i = 0; i < d.length; i += 4) nz.push(d[i + 3] ? 1 : 0);
    return { w: c.width, h: c.height, d50: document.getElementById("d50").value, painted: nz.reduce((a, b) => a + b, 0) };
  });
  await page.screenshot({ path: `${process.argv[3] || "."}/decay_s${s}.png`, fullPage: true });
}
writeFileSync(`${process.argv[3] || "."}/decay_check.json`, JSON.stringify(out));
console.log(JSON.stringify(out));
await browser.close();
if (errors.length) process.exit(1);
