import { defineConfig } from "vite";
import { resolve, sep } from "node:path";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { json } from "node:stream/consumers";
import { createReadStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { promisify } from "node:util";
import { brotliCompress, constants, gzip } from "node:zlib";
import clients from "./src/generated/clients.json";
import { wasmImport } from "./scripts/wasm-alias.mjs";

const instrumented = new Map([["profiling", "profile"]]);
const selected = process.env.ENGINE_GAME;
const games = clients.flatMap(({ wasm, game, page }) => game && page && (!selected || game === selected) ? [{ wasm, game, page }] : []);
const publicFolder = resolve(import.meta.dirname, "public");
const packIndex = resolve(publicFolder, "cooked/index.json");
const workspace = resolve(import.meta.dirname, "..");
const gameFolders = clients.map(client => resolve(workspace, client.folder) + sep);

export default defineConfig(({ command, mode }) => ({
  define: { __PUBLIC_PACKS__: JSON.stringify(existsSync(packIndex)) },
  plugins: [{
    name: "precompressed-copies",
    apply: () => command === "build" && mode === "production",
    async writeBundle(options, bundle) {
      const brotli = promisify(brotliCompress);
      const zip = promisify(gzip);
      await Promise.all(Object.keys(bundle).filter(name => /\.(js|wasm)$/.test(name)).map(async name => {
        const file = resolve(options.dir ?? "", name);
        const bytes = await readFile(file);
        const [br, gz] = await Promise.all([
          brotli(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: constants.BROTLI_MAX_QUALITY, [constants.BROTLI_PARAM_SIZE_HINT]: bytes.length } }),
          zip(bytes, { level: constants.Z_BEST_COMPRESSION }),
        ]);
        await Promise.all([writeFile(`${file}.br`, br), writeFile(`${file}.gz`, gz)]);
      }));
    },
  }, {
    name: "public-lists-restart",
    configureServer(server) {
      server.watcher.on("all", (event, path) => {
        if ((event === "add" || event === "unlink") && resolve(path) === packIndex) void server.restart();
      });
    },
  }, {
    name: "dev-loop",
    hotUpdate({ file }) {
      return file.includes("/pkg-dev/") ? [] : undefined;
    },
    configureServer(server) {
      server.middlewares.use("/_engine/reload", (request, response) => {
        if (request.method !== "POST") { response.statusCode = 405; response.end(); return; }
        for (const environment of Object.values(server.environments)) environment.moduleGraph.invalidateAll();
        server.ws.send({ type: "full-reload", path: "*" });
        response.end();
      });
      server.middlewares.use("/_engine/scene", async (request, response) => {
        if (request.method !== "POST") { response.statusCode = 405; response.end(); return; }
        try {
          const { files } = await json(request) as { files: string[] };
          server.ws.send({ type: "custom", event: "engine:scene-changed", data: { urls: files.map(file => `/_engine/source/${file}`) } });
          response.end();
        } catch { response.statusCode = 400; response.end(); }
      });
      server.middlewares.use("/_engine/source", async (request, response) => {
        try {
          const file = resolve(workspace, `.${decodeURIComponent(request.url?.split("?")[0] ?? "")}`);
          if (request.method !== "GET" || !file.endsWith(".scene.json") || !gameFolders.some(folder => file.startsWith(folder))) throw new Error("not a game's scene file");
          const text = await readFile(file);
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.setHeader("Cache-Control", "no-store");
          response.end(text);
        } catch { response.statusCode = 404; response.end("Scene source not found"); }
      });
      server.middlewares.use("/_engine/runs", async (request, response) => {
        const plain = /^[A-Za-z0-9_.-]+$/;
        const safe = (name: string) => plain.test(name) && !name.startsWith(".");
        const runsFolder = resolve(workspace, "artifacts", "runs");
        const missing = () => { response.statusCode = 404; response.end("Run file not found"); };
        if (request.method !== "GET") { response.statusCode = 405; response.end(); return; }
        const [path = "", query = ""] = (request.url ?? "").split("?");
        response.setHeader("Cache-Control", "no-store");
        try {
          if (path === "/" || path === "") {
            const folders = existsSync(runsFolder) ? await readdir(runsFolder, { withFileTypes: true }) : [];
            const runs = await Promise.all(folders.filter(folder => folder.isDirectory() && safe(folder.name)).map(async folder => {
              const inside = await readdir(resolve(runsFolder, folder.name), { withFileTypes: true }).catch(() => null);
              if (!inside) return null;
              const files = await Promise.all(inside.filter(file => file.isFile() && safe(file.name)).map(file =>
                stat(resolve(runsFolder, folder.name, file.name)).then(facts => ({ name: file.name, bytes: facts.size, modified_ms: facts.mtimeMs }), () => null)));
              const kept = files.flatMap(file => file ? [file] : []);
              return { name: folder.name, open: kept.some(file => file.name === "session.json"), files: kept };
            }));
            response.setHeader("Content-Type", "application/json; charset=utf-8");
            response.end(JSON.stringify(runs.flatMap(run => run ? [run] : [])));
            return;
          }
          const parts = path.split("/");
          if (parts.length !== 3 || parts[0] !== "" || !safe(parts[1]) || !safe(parts[2])) { missing(); return; }
          const from = new URLSearchParams(query).get("from") ?? "0";
          if (!/^[0-9]+$/.test(from)) { missing(); return; }
          const file = resolve(runsFolder, parts[1], parts[2]);
          const facts = await stat(file);
          if (!facts.isFile()) { missing(); return; }
          const start = Number(from);
          const types: Record<string, string> = { png: "image/png", json: "application/json; charset=utf-8", jsonl: "application/jsonl; charset=utf-8", log: "text/plain; charset=utf-8" };
          response.setHeader("Content-Type", types[parts[2].split(".").pop() ?? ""] ?? "application/octet-stream");
          response.setHeader("X-Size", String(facts.size));
          if (start >= facts.size) { response.end(); return; }
          response.setHeader("Content-Length", String(facts.size - start));
          const stream = createReadStream(file, { start, end: facts.size - 1 });
          stream.on("error", () => response.destroy());
          stream.pipe(response);
        } catch { missing(); }
      });
    },
  }],
  resolve: {
    alias: instrumented.has(mode) ? games.map(({ wasm }) => ({
      find: wasmImport(wasm), replacement: resolve(import.meta.dirname, `pkg-${instrumented.get(mode)}/${wasm}.js`),
    })) : command === "serve" ? clients.map(client => client.wasm).flatMap(name => [{
      find: wasmImport(name), replacement: resolve(import.meta.dirname, `pkg-dev/${name}.js`),
    }, {
      find: wasmImport(name, "_bg\\.wasm\\?url"), replacement: `${resolve(import.meta.dirname, `pkg-dev/${name}_bg.wasm`)}?url`,
    }]) : [],
  },
  worker: { format: "es" },
  build: {
    outDir: instrumented.has(mode) ? `dist-${instrumented.get(mode)}` : "dist",
    rollupOptions: {
      input: Object.fromEntries(games.map(({ game, page }) => [game, resolve(import.meta.dirname, page)])),
    },
  },
}));
