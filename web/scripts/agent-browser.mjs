/** Own one identified development runtime, independent of its HTTP transport. */
// @ts-check
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rm, stat } from "node:fs/promises";
import { resolve, relative, sep } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { AgentError, loopbackUrl } from "./agent-protocol.mjs";
import { launchContext } from "./browser-launch.mjs";
import { recordSent } from "./exercised.mjs";
import { workspace } from "./workspace.mjs";

/** How many pixels of two captures under artifacts/agent differ by more
 * than `tolerance` (0 to 255) in a color channel: `{a, b, tolerance}`. */
async function compareCaptures(command, requestBrowser, page) {
  if (Object.keys(command).some(key => !["op", "a", "b", "tolerance"].includes(key))) {
    throw new AgentError("unknown_field", "compare takes a, b and tolerance.", "$");
  }
  const tolerance = command.tolerance ?? 0;
  if (!Number.isInteger(tolerance) || tolerance < 0 || tolerance > 255) throw new AgentError("out_of_range", "tolerance is 0 to 255.", "tolerance");
  const root = resolve(workspace, "artifacts", "agent");
  const images = [];
  for (const name of ["a", "b"]) {
    const path = typeof command[name] === "string" ? resolve(workspace, command[name]) : "";
    if (!path.startsWith(root + sep) || !path.endsWith(".png")) throw new AgentError("invalid_path", `${name} must name a capture under artifacts/agent.`, name);
    const bytes = await readFile(path).catch(() => { throw new AgentError("invalid_path", `${command[name]} cannot be read.`, name); });
    images.push(bytes.toString("base64"));
  }
  const response = await requestBrowser({ op: "status" });
  if (!response.ok) return response;
  const result = await page.evaluate(async ({ images, tolerance }) => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (pixels[0].width !== pixels[1].width || pixels[0].height !== pixels[1].height) {
      return { same_size: false, sizes: pixels.map(({ width, height }) => [width, height]) };
    }
    let different = 0, largest = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      let pixel = 0;
      for (let channel = 0; channel < 3; channel++) pixel = Math.max(pixel, Math.abs(pixels[0].data[offset + channel] - pixels[1].data[offset + channel]));
      if (pixel > tolerance) different++;
      largest = Math.max(largest, pixel);
    }
    return { same_size: true, width: pixels[0].width, height: pixels[0].height, different, total: pixels[0].width * pixels[0].height, largest_difference: largest };
  }, { images, tolerance });
  return { ...response, result: { compare: result, a: command.a, b: command.b, tolerance } };
}

/** A replacement is ready only after its identity, build, pause, and render checks pass. */
// Sessions' captures stay under artifacts/agent/SESSION; past the most
// recent SESSIONS_KEPT, the oldest go.
const SESSIONS_KEPT = 10;

async function pruneSessions(root) {
  const entries = await readdir(root, { withFileTypes: true }).catch(() => []);
  const sessions = await Promise.all(entries.filter(entry => entry.isDirectory()).map(async entry => {
    const path = resolve(root, entry.name);
    return { path, modified: (await stat(path).catch(() => ({ mtimeMs: 0 }))).mtimeMs };
  }));
  sessions.sort((a, b) => b.modified - a.modified);
  await Promise.all(sessions.slice(SESSIONS_KEPT).map(({ path }) => rm(path, { recursive: true, force: true })));
}

/**
 * A development session of the page `url` (a loopback address) in Chrome,
 * `browserPath`'s where given, named `session`, refusing a page of another
 * build than `expectedBuild` where given, which `signal` ends.
 * @param {{ url: string | URL, browserPath?: string, session?: string, expectedBuild?: string | null, signal?: AbortSignal }} options
 */
export async function launchBrowserSession({ url: value, browserPath, session = randomUUID(), expectedBuild, signal }) {
  const url = loopbackUrl(String(value));
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(session)) throw new AgentError("invalid_session", "Session identity must be a bounded identifier.", "session");
  // The shaders earlier sessions compiled stay compiled (launchContext).
  const browser = await launchContext({ executablePath: browserPath, viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
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
    recordSent(command);
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
    listen(browser.context, "close", invalidate);
    page = browser.page;
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
    try {
      // Navigations before the runtime is ready (the dev server reloading
      // after a rebuild) still belong to loading it.
      await page.waitForFunction(() => typeof window.__engineDev?.request === "function" || document.querySelector("#status")?.getAttribute("data-state") === "error", null, { timeout: 60_000 });
      const identity = await page.evaluate(() => ({
        instance: window.__engineDev?.instance,
        build: window.__engineDev?.build ?? null,
        ready: typeof window.__engineDev?.request === "function",
      }));
      if (!identity.ready) throw new AgentError("runtime_unavailable", "The game reported an initialization error.");
      initialNavigation = false;
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
    await pruneSessions(resolve(workspace, "artifacts", "agent"));
    return {
      session, url: url.href, instance, build, close, setInvalidated,
      /** The page the runtime runs in, for checks that read it. */
      page,
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
        if (command.op === "compare") {
          return compareCaptures(command, requestBrowser, page);
        }
        if (command.op === "diagnostics") {
          const response = await requestBrowser({ op: "status" });
          if (response.ok) response.result = { observation: response.observation ?? response.result?.observation, entries: [...diagnostics] };
          return response;
        }
        const response = await requestBrowser(command);
        if (command.op === "describe" && command.name === undefined && response.ok && Array.isArray(response.result?.types)) {
          response.result.types.push({ name: "ProfileCapture", kind: "tooling", description: "Optional detailed profiling in the isolated profiling build" });
          const { controls, parameters } = await page.evaluate(() => ({ controls: window.__engineDev?.controls ?? {}, parameters: window.__engineDev?.parameters ?? {} }));
          response.result.types.push({ name: "GameControls", kind: "tooling", description: "The game's own controls, which the game control runs by name with args", controls });
          response.result.types.push({ name: "PageParameters", kind: "tooling", description: "The page URL's parameters the host and game read", parameters });
        }
        // Captures, and frames that keep their last: the canvas as shown.
        if ((command.op === "capture" || (command.op === "frames" && command.capture === true)) && response.ok) {
          if (response.result?.paused !== true) throw new AgentError("capture_requires_pause", "The browser must confirm it is paused before capture.");
          try {
            await mkdir(artifacts, { recursive: true });
            const artifact = resolve(artifacts, `capture-${++captureNumber}.png`);
            await page.locator("canvas").first().screenshot({ path: artifact, timeout: 10_000, animations: "disabled" });
            const bytes = await readFile(artifact);
            const observation = response.observation ?? response.result.observation;
            response.result.artifact = {
              path: relative(workspace, artifact).split(sep).join("/"), mime_type: "image/png", bytes: bytes.length,
              sha256: createHash("sha256").update(bytes).digest("hex"), frame: observation?.frame, tick: observation?.tick,
            };
          } catch (error) {
            return { ok: false, observation: response.observation ?? response.result.observation, error: { code: "capture_failed", path: "$", message: error.message } };
          }
        }
        return response;
      },
    };
  } catch (error) { await close(); throw error; }
}
