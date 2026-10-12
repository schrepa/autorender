/** Own the edit/build/browser loop; runtime frames never participate in watching. */
// @ts-check
import { randomUUID } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { access, mkdir, lstat, realpath, rm, stat, open } from "node:fs/promises";
import { resolve, dirname, relative } from "node:path";
import { AgentError, createBridge, failure } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";
import { fingerprintSources, watchSources } from "./iteration-watch.mjs";
import { IterationController } from "./iteration-controller.mjs";
import { runBuild } from "./iteration-process.mjs";
import { workspace } from "./workspace.mjs";

// The browser clients and their games' pages, generated from xtask
// (`cargo xtask schema`).
const clients = JSON.parse(readFileSync(new URL("../src/generated/clients.json", import.meta.url), "utf8"));
const gamePages = Object.fromEntries(clients.filter(client => client.game).map(client => [client.game, client.page]));
const wasmNames = clients.map(client => client.wasm);
const builtArtifact = new RegExp(`^/_engine/builds/([A-Za-z0-9_-]{1,100})/(${wasmNames.join("|")})(\\.js|_bg\\.wasm)$`);
const GENERATION = /^[A-Za-z0-9_-]{1,100}$/;
function port(value, fallback) {
  const text = value ?? String(fallback);
  if (!/^\d+$/.test(text) || Number(text) < 1 || Number(text) > 65535) throw new AgentError("invalid_arguments", "Ports must be integers from 1 to 65535.");
  return Number(text);
}

/** One foreground process owns Vite, a bridge, one build, and at most two browsers. */
export async function startIteration(values) {
  for (const key of Object.keys(values)) if (!["web-port", "port", "game", "browser"].includes(key)) throw new AgentError("invalid_arguments", `Unknown option --${key}.`);
  const webPort = port(values["web-port"], 5174);
  const bridgePort = port(values.port, 7331);
  if (webPort === bridgePort) throw new AgentError("invalid_arguments", "Web and bridge ports must differ.");
  const game = values.game ?? "stress";
  if (!(game in gamePages)) throw new AgentError("invalid_arguments", `Game must be one of ${Object.keys(gamePages).join(", ")}.`);
  const executable = process.env.ENGINE_XTASK_BIN ?? resolve(workspace, "target/debug", process.platform === "win32" ? "xtask.exe" : "xtask");
  await access(executable).catch(() => { throw new AgentError("build_worker_unavailable", "Start with cargo xtask agent watch to compile the native build worker.", executable); });
  const outputRoot = resolve(workspace, "web/pkg-iterations");
  await mkdir(outputRoot, { recursive: true });
  if (relative(outputRoot, await realpath(outputRoot)) !== "") throw new AgentError("unsafe_output_path", "Generated output must not resolve through a symlink.", outputRoot);
  // The dedicated Cargo target directory is shared. One supervisor per working copy
  // prevents cross-supervisor compiler/artifact races; other copies are independent.
  const lockPath = resolve(outputRoot, ".watch.lock");
  const watch = randomUUID();
  let lock;
  try { lock = await open(lockPath, "wx"); }
  catch (error) {
    if (error.code === "EEXIST") throw new AgentError("watch_locked", "This working copy already has a watcher. After a crash, confirm it stopped before removing web/pkg-iterations/.watch.lock.", lockPath);
    throw error;
  }
  await lock.writeFile(JSON.stringify({ pid: process.pid, watch, web_port: webPort, bridge_port: bridgePort }));
  const cancellation = new AbortController();
  const owned = new Set();
  const available = new Set();
  const retained = [];
  const browserStates = new WeakMap();
  const baseUrl = `http://127.0.0.1:${webPort}/${gamePages[game]}`;
  let active;
  let controller;
  let bridge;
  let vite;
  let stopWatching;
  let watcherError;
  let shutdownPromise;
  let finishStartup;
  const startupFinished = new Promise(resolveStartup => { finishStartup = resolveStartup; });
  const emit = event => { process.stdout.write(`${JSON.stringify({ version: 1, watch, ...event })}\n`); };

  async function removeGeneration(id) {
    if (!GENERATION.test(id) || !owned.has(id)) return;
    const path = resolve(outputRoot, id);
    try {
      if ((await lstat(path)).isSymbolicLink() || dirname(await realpath(path)) !== outputRoot) throw new AgentError("unsafe_output_path", "Refusing to remove a generation outside its owned directory.", path);
      await rm(path, { recursive: true, force: true });
    } catch (error) { if (error.code !== "ENOENT") throw error; }
    owned.delete(id);
    available.delete(id);
  }
  async function shutdown() {
    if (shutdownPromise) return shutdownPromise;
    shutdownPromise = (async () => {
      cancellation.abort();
      stopWatching?.();
      bridge?.markUnavailable();
      // Await acquisitions already in flight so no server can appear after cleanup.
      await startupFinished;
      const cleanupErrors = [];
      const clean = async operation => { try { await operation(); } catch (error) { cleanupErrors.push(error); } };
      await clean(() => controller?.stop());
      await clean(() => active?.close());
      await clean(() => vite?.close());
      await clean(async () => {
        if (bridge?.server.listening) {
          bridge.server.closeAllConnections();
          await new Promise(resolveClose => bridge.server.close(resolveClose));
        }
      });
      for (const id of [...owned]) await clean(() => removeGeneration(id));
      await clean(() => lock.close());
      await clean(() => rm(lockPath));
      process.off("SIGINT", stopSignal);
      process.off("SIGTERM", stopSignal);
      if (cleanupErrors.length) throw new AggregateError(cleanupErrors, "Some owned development resources could not be cleaned up.");
    })();
    return shutdownPromise;
  }
  const stopSignal = () => { void shutdown().catch(error => { process.stderr.write(`${error}\n`); process.exitCode = 1; }); };
  process.once("SIGINT", stopSignal);
  process.once("SIGTERM", stopSignal);

  try {
    controller = new IterationController({
      signal: cancellation.signal,
      fingerprint: () => fingerprintSources(workspace),
      build: async (attempt) => {
        owned.add(attempt.id);
        const result = await runBuild({ executable, root: workspace, output: resolve(outputRoot, attempt.id), signal: cancellation.signal });
        if (result.ok) available.add(attempt.id);
        return result;
      },
      prepare: async attempt => {
        const url = new URL(baseUrl);
        url.searchParams.set("engine_build", attempt.id);
        const candidate = await launchBrowserSession({ url, browserPath: values.browser, expectedBuild: attempt.id, signal: cancellation.signal });
        const state = { available: true };
        browserStates.set(candidate, state);
        candidate.setInvalidated(() => { state.available = false; if (active === candidate) bridge.markUnavailable(); });
        return candidate;
      },
      promote: async (candidate, snapshot, commit) => {
        const previous = active;
        await bridge.replaceSession({
          session: candidate.session, url: candidate.url, build: candidate.build, execute: candidate.execute,
          onTimeout: () => { browserStates.get(candidate).available = false; return candidate.close(); },
          validate: async () => {
            const matches = !cancellation.signal.aborted && (await fingerprintSources(workspace)).fingerprint === snapshot.fingerprint;
            if (!browserStates.get(candidate)?.available) throw new AgentError("activation_failed", "The candidate browser closed before activation.");
            return matches;
          },
          commit: () => {
            if (!browserStates.get(candidate)?.available) throw new AgentError("activation_failed", "The candidate browser closed before activation.");
            commit();
            active = candidate;
          },
        });
        retained.push(candidate.build);
        await previous?.close();
        while (retained.length > 3) await removeGeneration(retained.shift());
      },
      discard: async (attempt, candidate) => {
        await candidate?.close();
        await removeGeneration(attempt.id);
      },
      onEvent: status => emit({ event: "development", state: status.state, active: status.active, latest: status.latest && { build: status.latest.build, state: status.latest.state, error: status.latest.error } }),
    });
    bridge = createBridge({
      session: randomUUID(), url: baseUrl, initiallyUnavailable: true,
      execute: () => { throw new AgentError("session_unavailable", "No successful build is active."); },
      onClose: shutdown,
      development: () => ({ watch, game, web_url: baseUrl, bridge_url: `http://127.0.0.1:${bridgePort}`, runtime_state: active ? (browserStates.get(active)?.available ? "ready" : "unavailable") : "none", watcher_error: watcherError ?? null, ...controller.status() }),
    });
    await new Promise((resolveListen, reject) => { bridge.server.once("error", reject); bridge.server.listen(bridgePort, "127.0.0.1", resolveListen); });
    cancellation.signal.throwIfAborted();
    const { createServer } = await import("vite");
    vite = await createServer({
      root: resolve(workspace, "web"), configFile: resolve(workspace, "web/vite.config.ts"),
      logLevel: "silent",
      resolve: { alias: wasmNames.map(name => ({ find: `../pkg/${name}`, replacement: `virtual:iteration-${name}` })) },
      server: { host: "127.0.0.1", port: webPort, strictPort: true, hmr: false, watch: { ignored: ["**/pkg-iterations/**"] } },
      plugins: [{
        name: "immutable-development-builds",
        resolveId(id) { if (id.startsWith("virtual:iteration-")) return `\0${id}`; },
        load(id) { if (id.startsWith("\0virtual:iteration-")) return 'throw new Error("Open the base game URL to select an active Rust build."); export default undefined;'; },
        configureServer(server) {
          server.middlewares.use(async (request, response, next) => {
            const url = new URL(request.url ?? "/", baseUrl);
            const match = builtArtifact.exec(url.pathname);
            if (match) {
              if (request.method !== "GET" && request.method !== "HEAD") { response.statusCode = 405; response.end(); return; }
              if (!available.has(match[1])) { response.statusCode = 404; response.end("Build unavailable; open the game's base URL for the active build."); return; }
              try {
                const path = resolve(outputRoot, match[1], `${match[2]}${match[3]}`);
                const metadata = await stat(path);
                response.setHeader("Content-Type", match[3] === ".js" ? "text/javascript; charset=utf-8" : "application/wasm");
                response.setHeader("Content-Length", metadata.size);
                response.setHeader("Cache-Control", "no-store");
                if (request.method === "HEAD") { response.end(); return; }
                const stream = createReadStream(path);
                stream.on("error", () => response.destroy());
                response.on("close", () => stream.destroy());
                stream.pipe(response);
              } catch { response.statusCode = 404; response.end("Build artifact unavailable"); }
              return;
            }
            if (["/", ...Object.values(gamePages).map(page => `/${page}`)].includes(url.pathname) && !url.searchParams.has("engine_build")) {
              if (!active) { response.statusCode = 503; response.end("The first Rust build is not ready. Read agent development status for diagnostics."); return; }
              url.searchParams.set("engine_build", active.build);
              response.writeHead(302, { Location: `${url.pathname}${url.search}`, "Cache-Control": "no-store" }); response.end(); return;
            }
            next();
          });
        },
      }],
    });
    await vite.listen();
    cancellation.signal.throwIfAborted();
    emit({ event: "watch_ready", endpoint: `http://127.0.0.1:${bridgePort}`, url: baseUrl, state: "starting" });
    stopWatching = watchSources({
      root: workspace, signal: cancellation.signal,
      onChange: snapshot => { watcherError = undefined; controller.enqueue(snapshot); },
      onError: error => { watcherError = failure(error).error; emit({ event: "watch_error", error: watcherError }); },
    });
  } catch (error) {
    finishStartup();
    await shutdown();
    if (cancellation.signal.aborted && error.name === "AbortError") return;
    throw error;
  } finally { finishStartup(); }
}
