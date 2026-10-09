/** Own one identified development runtime, independent of its HTTP transport. */
import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { AgentError, loopbackUrl } from "./agent-protocol.mjs";
import { launchOptions } from "./browser-launch.mjs";

const workspace = fileURLToPath(new URL("../../", import.meta.url));

/** A replacement is ready only after its identity, build, pause, and render checks pass. */
export async function launchBrowserSession({ url: value, browserPath, session = randomUUID(), expectedBuild, signal }) {
  const url = loopbackUrl(String(value));
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(session)) throw new AgentError("invalid_session", "Session identity must be a bounded identifier.", "session");
  const { chromium } = await import("@playwright/test");
  const browser = await chromium.launch(launchOptions({ executablePath: browserPath }));
  let page;
  let instance;
  let build;
  let closing;
  let invalidated = false;
  let invalidationCallback;
  let initialNavigation = true;
  let captureNumber = 0;
  const listeners = [];
  const diagnostics = [];

  function listen(target, event, handler) {
    target.on(event, handler);
    listeners.push(() => target.off(event, handler));
  }
  function diagnostic(kind, message) {
    diagnostics.push({ kind, message: message.slice(0, 2000), at: new Date().toISOString() });
    if (diagnostics.length > 100) diagnostics.shift();
  }
  function close() {
    if (!closing) {
      for (const remove of listeners.splice(0)) remove();
      closing = browser.close();
    }
    return closing;
  }
  function invalidate() {
    if (invalidated || closing) return;
    invalidated = true;
    try { invalidationCallback?.(); } finally { void close().catch(() => {}); }
  }
  function setInvalidated(callback) {
    invalidationCallback = callback;
    if (invalidated) callback();
  }
  const aborted = () => { void close().catch(() => {}); };
  if (signal?.aborted) { await close(); throw new AgentError("session_cancelled", "Browser startup was cancelled."); }
  signal?.addEventListener("abort", aborted, { once: true });
  listeners.push(() => signal?.removeEventListener("abort", aborted));
  async function requestBrowser(command) {
    if (invalidated || closing) throw new AgentError("session_unavailable", "The owned browser runtime is no longer available.");
    const response = await page.evaluate(({ expected, expectedBuild, command }) => {
      const api = window.__engineDev;
      if (!api || api.instance !== expected || (expectedBuild !== null && api.build !== expectedBuild)) {
        return { ok: false, error: { code: "runtime_replaced", path: "$", message: "The browser runtime changed. Start a new development session." } };
      }
      return api.request(command);
    }, { expected: instance, expectedBuild: build, command });
    if (response?.error?.code === "runtime_replaced") invalidate();
    return response;
  }
  async function readyRequest(command) {
    const deadline = Date.now() + 5000;
    while (true) {
      let timer;
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new AgentError("runtime_busy", "The development runtime did not finish initialization.");
      let response;
      try {
        response = await Promise.race([
          requestBrowser(command),
          new Promise((_, reject) => { timer = setTimeout(() => reject(new AgentError("runtime_unavailable", "The initialization command timed out.")), remaining); }),
        ]);
      } finally { clearTimeout(timer); }
      // Busy means the command was rejected before dispatch, so retry is safe.
      if (response?.error?.code === "command_busy") { await delay(25); continue; }
      if (!response?.ok) throw new AgentError(response?.error?.code ?? "runtime_unavailable", response?.error?.message ?? "The development runtime rejected initialization.", response?.error?.path);
      if (response.result?.paused !== true) throw new AgentError("runtime_not_paused", "A new development runtime must start paused.");
      return response;
    }
  }

  try {
    listen(browser, "disconnected", invalidate);
    page = await browser.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
    listen(page, "pageerror", error => diagnostic("pageerror", String(error)));
    listen(page, "console", message => { if (["error", "warning"].includes(message.type())) diagnostic(message.type(), message.text()); });
    listen(page, "framenavigated", frame => { if (!initialNavigation && frame === page.mainFrame()) invalidate(); });
    listen(page, "close", invalidate);
    listen(page, "crash", invalidate);
    await page.route("**/*", async route => {
      const request = route.request();
      if (request.isNavigationRequest() && request.frame() === page.mainFrame()) {
        try { loopbackUrl(request.url()); } catch { await route.abort("blockedbyclient"); return; }
      }
      await route.continue();
    });
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 30_000 });
    initialNavigation = false;
    try {
      await page.waitForFunction(() => typeof window.__engineDev?.request === "function" || document.querySelector("#status")?.getAttribute("data-state") === "error", null, { timeout: 60_000 });
      const identity = await page.evaluate(() => ({
        instance: window.__engineDev?.instance,
        build: window.__engineDev?.build ?? null,
        ready: typeof window.__engineDev?.request === "function",
      }));
      if (!identity.ready) throw new AgentError("runtime_unavailable", "The game reported an initialization error.");
      instance = identity.instance;
      build = identity.build;
      if (typeof instance !== "string" || !instance) throw new AgentError("missing_runtime_identity", "The development browser must expose a runtime instance identity.");
      if (expectedBuild !== undefined && build !== expectedBuild) throw new AgentError("build_mismatch", `Expected build ${expectedBuild}, received ${String(build)}.`, "build");
      await readyRequest({ op: "render" });
      await readyRequest({ op: "status" });
    } catch (error) {
      const status = await page.locator("#status").textContent({ timeout: 1000 }).catch(() => "unavailable");
      throw new AgentError(error?.code ?? "runtime_unavailable", `${error?.message ?? error} Status: ${status}. Browser diagnostics: ${JSON.stringify(diagnostics.slice(-5))}`, error?.path);
    }
    if (invalidated || closing) throw new AgentError("session_unavailable", "The browser closed during initialization.");
    const artifacts = resolve(workspace, "artifacts", "agent", session);
    return {
      session, url: url.href, instance, build, close, setInvalidated,
      async execute(command) {
        if (command.op === "describe" && command.name === "ProfileCapture") {
          if (Object.keys(command).some(key => !["op", "name"].includes(key))) throw new AgentError("invalid_request", "describe accepts only name.");
          const response = await requestBrowser({ op: "status" });
          if (!response.ok) return response;
          const { profileDescriptor } = await import("./agent-profile.mjs");
          const status = await page.evaluate(() => window.__engineProfile?.status() ?? { available: false });
          response.result = { ...profileDescriptor, status };
          return response;
        }
        if (command.op === "profile") {
          const { captureProfile } = await import("./agent-profile.mjs");
          return captureProfile({ page, requestBrowser, command, artifacts, workspace });
        }
        if (command.op === "diagnostics") {
          const response = await requestBrowser({ op: "status" });
          if (response.ok) response.result = { observation: response.observation ?? response.result?.observation, entries: [...diagnostics] };
          return response;
        }
        const response = await requestBrowser(command);
        if (command.op === "describe" && command.name === undefined && response.ok && Array.isArray(response.result?.types)) {
          response.result.types.push({ name: "ProfileCapture", kind: "tooling", description: "Optional detailed profiling in the isolated profiling build" });
        }
        if (command.op === "capture" && response.ok) {
          if (response.result?.paused !== true) throw new AgentError("capture_requires_pause", "The browser must confirm it is paused before capture.");
          try {
            await mkdir(artifacts, { recursive: true });
            const artifact = resolve(artifacts, `capture-${++captureNumber}.png`);
            await page.locator("canvas").first().screenshot({ path: artifact, timeout: 10_000, animations: "disabled" });
            response.result.artifact = { path: relative(workspace, artifact).split(sep).join("/"), mime_type: "image/png" };
          } catch (error) {
            return { ok: false, observation: response.observation ?? response.result.observation, error: { code: "capture_failed", path: "$", message: error.message } };
          }
        }
        return response;
      },
    };
  } catch (error) { await close(); throw error; }
}
