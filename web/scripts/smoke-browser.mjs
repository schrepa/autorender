// @ts-check
import { setTimeout as delay } from "node:timers/promises";
import { launchContext } from "./browser-launch.mjs";

const [url, limit = "600000"] = process.argv.slice(2);
const { page, close } = await launchContext({ viewport: { width: 1000, height: 700 }, deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", error => { if (errors.length < 20) errors.push(String(error).slice(0, 2000)); });
page.on("console", entry => { if (entry.type() === "error" && errors.length < 20) errors.push(entry.text().slice(0, 2000)); });
let printed = 0;
try {
  await page.goto(url, { waitUntil: "domcontentloaded" });
  const until = Date.now() + Number(limit);
  for (;;) {
    const state = await page.evaluate(() => /** @type {any} */ (window).__engineSmokes ?? null);
    for (const record of state?.records.slice(printed) ?? []) {
      console.log(JSON.stringify(record));
      printed++;
    }
    if (state?.done) {
      if (state.error) {
        console.error(JSON.stringify({ error: state.error, page: errors }));
        process.exitCode = 1;
      }
      break;
    }
    if (Date.now() > until) {
      console.error(JSON.stringify({ error: { code: "command_timeout", path: "$" }, page: errors }));
      process.exitCode = 1;
      break;
    }
    await delay(250);
  }
} finally {
  await close();
}
