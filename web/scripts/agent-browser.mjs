// @ts-check
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { resolve, relative, sep } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { AgentError, atMost, detailOf, loopbackUrl, range, refusal, variants } from "./agent-protocol.mjs";
import { launchContext } from "./browser-launch.mjs";
import { workspace } from "./workspace.mjs";

const FIRST_RENDER_MS = 300_000;

const status = { verb: "query", name: "status" };

async function compareCaptures(compare, requestBrowser, page) {
  if (!compare || typeof compare !== "object" || Object.keys(compare).some(key => !["a", "b", "tolerance"].includes(key))) {
    throw new AgentError("unknown_field", "$.compare", { allowed: variants(["a", "b", "tolerance"]) });
  }
  const tolerance = compare.tolerance ?? 0;
  if (!Number.isInteger(tolerance) || tolerance < 0 || tolerance > 255) throw new AgentError("out_of_range", "$.compare.tolerance", { allowed: range(0, 255), got: tolerance });
  const root = resolve(workspace, "artifacts", "agent");
  const images = [];
  for (const name of ["a", "b"]) {
    const path = typeof compare[name] === "string" ? resolve(workspace, compare[name]) : "";
    if (!path.startsWith(root + sep) || !path.endsWith(".png")) throw new AgentError("invalid_path", `$.compare.${name}`, { expected: "path", got: compare[name] });
    const bytes = await readFile(path).catch(() => { throw new AgentError("invalid_path", `$.compare.${name}`, { expected: "readable", got: compare[name] }); });
    images.push(bytes.toString("base64"));
  }
  const response = await requestBrowser(status);
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
  return { ...response, result: { compare: result, a: compare.a, b: compare.b, tolerance } };
}

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

function captures(command) {
  return command.verb === "capture" || (command.verb === "control" && command.name === "frames" && command.args?.capture === true);
}

const CANVAS_ONLY = "body *{visibility:hidden!important}canvas{visibility:visible!important}";

/** @param {{ browserPath?: string }} [options] */
async function openBrowser({ browserPath } = {}) {
  const browser = await launchContext({ executablePath: browserPath, viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const { page } = browser;
  try {
    await page.addInitScript(() => {
      const device = /** @type {any} */ (globalThis).GPUDevice;
      if (device === undefined) return;
      const create = device.prototype.createShaderModule;
      device.prototype.createShaderModule = /** @this {any} @param {any} descriptor */ function (descriptor) {
        const module = create.call(this, descriptor);
        void module.getCompilationInfo().then(info => {
          for (const message of info.messages) {
            if (message.type === "error") console.error(`createShaderModule ${descriptor.label ?? ""}:${message.lineNum}:${message.linePos} ${message.message}`);
          }
        });
        return module;
      };
      const pop = device.prototype.popErrorScope;
      device.prototype.popErrorScope = /** @this {any} */ function () {
        return pop.call(this).then(/** @param {any} error */ error => {
          if (error) console.error(`popErrorScope ${error.constructor?.name ?? "GPUError"}: ${error.message}`);
          return error;
        });
      };
    });
    await page.route("**/*", async route => {
      const request = route.request();
      if (request.isNavigationRequest() && request.frame() === page.mainFrame()) {
        try { loopbackUrl(request.url()); } catch { await route.abort("blockedbyclient"); return; }
      }
      await route.continue();
    });
  } catch (error) { await browser.close(); throw error; }
  return browser;
}

/** @param {{ url: string | URL, browserPath?: string }} options */
export async function launchBrowserSession({ url: value, browserPath }) {
  const url = loopbackUrl(String(value));
  const session = randomUUID();
  const browser = await openBrowser({ browserPath });
  let page;
  let instance;
  let ready;
  let closing;
  let invalidated = false;
  let invalidationCallback;
  let initialNavigation = true;
  let captureNumber = 0;
  let refuseLoad;
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
  async function evaluateRequest(command) {
    if (invalidated || closing) throw new AgentError("session_unavailable", "$", { expected: "live" });
    return page.evaluate(({ expected, command }) => {
      const api = window.__engineDev;
      if (!api || api.instance !== expected) {
        return { ok: false, error: { code: "runtime_replaced", path: "$", expected: "matching", got: api?.instance ?? null } };
      }
      return api.request(command);
    }, { expected: instance, command });
  }
  async function attached() {
    while (true) {
      if (invalidated || closing) throw new AgentError("session_unavailable", "$", { expected: "live" });
      const current = ready;
      try { await current; return current; } catch (error) { if (ready === current) throw error; }
    }
  }
  async function requestBrowser(command) {
    while (true) {
      const current = await attached();
      const asked = instance;
      try {
        const response = await evaluateRequest(command);
        if (response?.error?.code !== "runtime_replaced") return response;
      } catch (error) {
        if (await page.evaluate(() => window.__engineDev?.instance).catch(() => undefined) === asked) throw error;
      }
      if (ready === current) reattach();
    }
  }
  async function readyRequest(command, limit = 5000) {
    const deadline = Date.now() + limit;
    while (true) {
      let timer;
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new AgentError("runtime_busy", "$", { allowed: atMost(limit) });
      let response;
      try {
        response = await Promise.race([
          evaluateRequest(command),
          new Promise((_, reject) => { timer = setTimeout(() => reject(new AgentError("runtime_unavailable", "$", { allowed: atMost(remaining) })), remaining); }),
        ]);
      } finally { clearTimeout(timer); }
      if (response?.error?.code === "command_busy") { await delay(25); continue; }
      if (!response?.ok) throw new AgentError(response?.error?.code ?? "runtime_unavailable", response?.error?.path, detailOf(response?.error));
      if (response.result?.paused !== true) throw new AgentError("runtime_not_paused", "$", { expected: "paused" });
      return response;
    }
  }
  async function keepSaved(response, artifacts) {
    const artifact = response?.result?.artifact;
    if (!response?.ok || typeof artifact?.path !== "string" || !artifact.path.startsWith("page/")) return response;
    const text = await page.evaluate(path => window.__engineDev?.saved(path), artifact.path);
    if (typeof text !== "string") {
      return { ok: false, observation: response.observation, error: { code: "capture_failed", path: "$", expected: "readable", got: artifact.path } };
    }
    await mkdir(artifacts, { recursive: true });
    const path = resolve(artifacts, artifact.path.slice("page/".length));
    await writeFile(path, text);
    const bytes = await readFile(path);
    if (createHash("sha256").update(bytes).digest("hex") !== artifact.sha256) {
      return { ok: false, observation: response.observation, error: { code: "capture_failed", path: "$", expected: "matching", got: artifact.path } };
    }
    artifact.path = relative(workspace, path).split(sep).join("/");
    return response;
  }
  function watchLoad() {
    const refused = new Promise((_, reject) => { refuseLoad = reject; });
    refused.catch(() => {});
    return refused;
  }
  async function attach(refused) {
    try {
      await Promise.race([
        page.waitForFunction(() => typeof window.__engineDev?.request === "function" || document.querySelector("#status")?.getAttribute("data-state") === "error", null, { timeout: 60_000, polling: 500 }),
        refused,
      ]);
      const identity = await page.evaluate(() => ({
        instance: window.__engineDev?.instance,
        ready: typeof window.__engineDev?.request === "function",
      }));
      if (!identity.ready) throw new AgentError("runtime_unavailable", "$", { expected: "live" });
      if (typeof identity.instance !== "string" || !identity.instance) throw new AgentError("missing_runtime_identity", "$", { expected: "id", got: identity.instance ?? null });
      instance = identity.instance;
      const compiled = Date.now() + FIRST_RENDER_MS;
      while ((await readyRequest({ verb: "control", name: "render" }, FIRST_RENDER_MS)).result.rendering?.readiness?.compiling > 0) {
        if (Date.now() > compiled) throw new AgentError("runtime_busy", "$", { allowed: atMost(FIRST_RENDER_MS), got: { compiling: true } });
        await delay(50);
      }
    } catch (error) {
      const shown = await page.locator("#status").textContent({ timeout: 1000 }).catch(() => "unavailable");
      throw new AgentError(error?.code ?? "runtime_unavailable", error?.path, { got: { error: refusal(error), status: shown, diagnostics: diagnostics.slice(-5) } });
    }
  }
  function reattach(refused = watchLoad()) {
    ready = attach(refused);
    ready.catch(() => {});
  }

  try {
    const refused = watchLoad();
    listen(browser.context, "close", invalidate);
    page = browser.page;
    listen(page, "pageerror", error => diagnostic("pageerror", String(error)));
    listen(page, "console", message => { if (["error", "warning"].includes(message.type())) diagnostic(message.type(), message.text()); });
    listen(page, "response", response => {
      if (response.status() < 400) return;
      const failed = `${response.status()} ${response.url()}`;
      diagnostic("http", failed);
      if (["document", "script"].includes(response.request().resourceType())) refuseLoad(new AgentError("runtime_unavailable", "$", { expected: "live", got: failed }));
    });
    listen(page, "framenavigated", frame => { if (!initialNavigation && frame === page.mainFrame()) reattach(); });
    listen(page, "close", invalidate);
    listen(page, "crash", invalidate);
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 30_000 });
    initialNavigation = false;
    reattach(refused);
    await attached();
    if (invalidated || closing) throw new AgentError("session_unavailable", "$", { expected: "live" });
    const artifacts = resolve(workspace, "artifacts", "agent", session);
    await pruneSessions(resolve(workspace, "artifacts", "agent"));
    const owned = {
      session, url: url.href, close, setInvalidated,
      page,
      async activity() {
        const waiting = await Promise.race([page.evaluate(() => window.__engineDev?.waitingOnGpu()).catch(() => undefined), delay(1000)]);
        return typeof waiting === "boolean" ? { waiting_on_gpu: waiting } : {};
      },
      async execute(command) {
        if (command.verb === "describe" && command.name === "ProfileCapture") {
          if (Object.keys(command).some(key => !["verb", "name"].includes(key))) throw new AgentError("unknown_field", "$", { allowed: variants(["verb", "name"]) });
          const response = await requestBrowser(status);
          if (!response.ok) return response;
          const { profileDescriptor } = await import("./agent-profile.mjs");
          const available = await page.evaluate(() => window.__engineProfile?.status() ?? { available: false });
          response.result = { ...profileDescriptor, status: available };
          return response;
        }
        if (command.verb === "profile" && command.deep !== undefined) {
          const { captureProfile } = await import("./agent-profile.mjs");
          try { return await captureProfile({ page, requestBrowser, command: command.deep, artifacts, workspace }); }
          catch (error) {
            if (!(error instanceof AgentError)) throw error;
            const { observation } = await requestBrowser(status);
            return { ok: false, ...(observation ? { observation } : {}), error: refusal(error) };
          }
        }
        if (command.verb === "capture" && command.compare !== undefined) {
          const { compare, ...capture } = command;
          if (typeof compare?.b === "string") return compareCaptures(compare, requestBrowser, page);
          const captured = await owned.execute(capture);
          if (!captured.ok) return captured;
          const compared = await compareCaptures({ ...compare, b: captured.result.artifact.path }, requestBrowser, page);
          if (compared.ok) captured.result.compare = compared.result.compare;
          return compared.ok ? captured : compared;
        }
        if (command.verb === "query" && command.name === "diagnostics") {
          const response = await requestBrowser(status);
          if (response.ok) response.result = { observation: response.observation ?? response.result?.observation, entries: [...diagnostics] };
          return response;
        }
        const response = await requestBrowser(command);
        if (command.verb === "describe" && command.name === undefined && response.ok && Array.isArray(response.result?.types)) {
          response.result.types.push({ name: "ProfileCapture", kind: "tooling" });
          const { controls, parameters } = await page.evaluate(() => ({ controls: window.__engineDev?.controls ?? {}, parameters: window.__engineDev?.parameters ?? {} }));
          response.result.types.push({ name: "GameControls", kind: "tooling", controls });
          response.result.types.push({ name: "PageParameters", kind: "tooling", parameters });
        }
        if (captures(command) && response.ok) {
          if (response.result?.paused !== true) throw new AgentError("capture_requires_pause", "$", { expected: "paused" });
          try {
            await mkdir(artifacts, { recursive: true });
            const artifact = resolve(artifacts, `capture-${++captureNumber}.png`);
            await page.locator("canvas").first().screenshot({ path: artifact, timeout: 10_000, animations: "disabled", style: CANVAS_ONLY });
            const bytes = await readFile(artifact);
            const observation = response.observation ?? response.result.observation;
            response.result.artifact = {
              path: relative(workspace, artifact).split(sep).join("/"), mime_type: "image/png", bytes: bytes.length,
              sha256: createHash("sha256").update(bytes).digest("hex"), frame: observation?.frame, tick: observation?.tick,
            };
          } catch (error) {
            return { ok: false, observation: response.observation ?? response.result.observation, error: { ...refusal(error), code: "capture_failed", path: "$" } };
          }
        }
        return keepSaved(response, artifacts);
      },
    };
    return owned;
  } catch (error) { await close(); throw error; }
}
