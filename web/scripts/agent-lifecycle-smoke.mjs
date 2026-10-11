/** Browser-backed session lifetime checks; no game or GPU rendering is required. */
import "./exit-codes.mjs";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";

const script = fileURLToPath(new URL("./agent.mjs", import.meta.url));
const fixture = createServer((_, response) => {
  response.writeHead(200, { "Content-Type": "text/html" });
  response.end(`<script>
    window.__engineDev = {
      instance: crypto.randomUUID(),
      async request(command) {
        if (command.op === "replace") window.__engineDev.instance = crypto.randomUUID();
        if (command.op === "navigate") setTimeout(() => location.href = "/next", 25);
        return {ok:true,observation:{frame:"0",tick:"0",scene:null},result:{paused:true}};
      }
    };
  </script>`);
});
await new Promise(resolve => fixture.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${fixture.address().port}`;

try {
  for (const operation of ["replace", "navigate"]) {
    const reserve = createServer();
    await new Promise(resolve => reserve.listen(0, "127.0.0.1", resolve));
    const port = reserve.address().port;
    await new Promise(resolve => reserve.close(resolve));
    const child = spawn(process.execPath, [script, "serve", "--url", url, "--port", String(port)], { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", bytes => { stderr = (stderr + bytes.toString()).slice(-8192); });
    const lines = createInterface({ input: child.stdout });
    let session;
    async function request(op) {
      const response = await fetch(`http://127.0.0.1:${port}/request`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(5000),
        body: JSON.stringify({ version: 1, session, id: randomUUID(), command: { op } }),
      });
      return response.json();
    }
    try {
      const ready = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error(`Startup timed out: ${stderr}`)), 30_000);
        child.once("error", error => { clearTimeout(timeout); reject(error); });
        child.once("exit", code => { clearTimeout(timeout); reject(new Error(`Bridge exited ${code}: ${stderr}`)); });
        lines.once("line", line => { clearTimeout(timeout); resolve(JSON.parse(line)); });
      });
      assert.equal(ready.ok, true, JSON.stringify(ready));
      session = ready.session;
      assert.equal((await request("status")).ok, true);
      assert.equal((await request(operation)).ok, true);
      if (operation === "replace") assert.equal((await request("status")).error.code, "runtime_replaced");
      let state;
      for (let i = 0; i < 40; i++) {
        state = (await (await fetch(`http://127.0.0.1:${port}/sessions`)).json()).sessions[0].state;
        if (state === "unavailable") break;
        await delay(50);
      }
      assert.equal(state, "unavailable");
      assert.equal((await request("status")).error.code, "session_unavailable");
    } finally {
      lines.close();
      if (child.exitCode === null) {
        const exited = new Promise(resolve => child.once("exit", resolve));
        if (session) assert.equal((await request("close")).ok, true);
        else child.kill("SIGTERM");
        await exited;
      }
    }
  }
  console.log(JSON.stringify({ ok: true, checks: ["runtime_replacement", "main_frame_navigation", "close_unavailable_session"] }));
} finally {
  fixture.closeAllConnections();
  await new Promise(resolve => fixture.close(resolve));
}
