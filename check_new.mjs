// One-off check of this restructure's new behaviours: deep-link ?species=<id>,
// panel collapse (map standalone), and that species info renders OUTSIDE the map.
import { chromium } from "playwright";
const port = process.argv[2] || "8991";
const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
page.on("pageerror", (e) => errs.push(String(e)));

// Deep link straight to a non-default species.
await page.goto(`http://127.0.0.1:${port}/index.html?species=great_lakes`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__currentId, { timeout: 15000 });
const deep = await page.evaluate(() => window.__currentId);
const spName = await page.textContent("#sp-name");
const spGroup = await page.textContent("#sp-group");
const urlKept = page.url().includes("species=great_lakes");

// Panel collapse -> map stands alone.
await page.click("#panelbtn");
await page.waitForTimeout(300);
const collapsed = await page.$eval("#stage", (el) => el.classList.contains("collapsed"));
const btnLabel = await page.textContent("#panelbtn");

// species info is a sibling of the map, not inside #mapwrap.
const infoOutside = await page.$eval("#speciesinfo", (el) => !el.closest("#mapwrap"));

await page.screenshot({ path: "out/embed.png", fullPage: true });
await browser.close();

const ok = deep === "great_lakes" && spName === "Great Lakes basin" && spGroup === "plants"
  && urlKept && collapsed && btnLabel.includes("▶") && infoOutside && errs.length === 0;
console.log(`deep-link: __currentId=${deep}, name="${spName}", group="${spGroup}", url-kept=${urlKept}`);
console.log(`panel collapse: collapsed=${collapsed}, btn="${btnLabel.trim()}"`);
console.log(`species info outside map: ${infoOutside}`);
console.log(`console errors: ${errs.length}`); errs.slice(0,4).forEach(e=>console.log("  ! "+e));
console.log(ok ? "PASS: deep-link + standalone map + info-outside-map" : "FAIL");
process.exit(ok ? 0 : 1);
