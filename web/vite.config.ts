import { defineConfig } from "vite";
import { resolve, sep } from "node:path";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { json } from "node:stream/consumers";
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
