import { defineConfig } from "vite";
import { resolve, sep } from "node:path";
import { json } from "node:stream/consumers";
import { readFile, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { brotliCompress, constants, gzip } from "node:zlib";
import clients from "./src/generated/clients.json";

const instrumented = new Map([["profiling", "profile"]]);
const selected = process.env.ENGINE_GAME;
const games = clients.flatMap(({ game, page }) => game && page && (!selected || game === selected) ? [{ game, page }] : []);
const workspace = resolve(import.meta.dirname, "..");
const gameFolders = clients.map(client => resolve(workspace, client.folder) + sep);

const wasmImport = (name: string) => new RegExp(`^(?:\\.\\./)+pkg/${name}$`);

export default defineConfig(({ command, mode }) => ({
  plugins: [{
    name: "precompressed-copies",
    apply: () => command === "build" && mode === "production",
    async writeBundle(options, bundle) {
      const brotli = promisify(brotliCompress);
      const zip = promisify(gzip);
      const compressed = async (file: string) => {
        const bytes = await readFile(file);
        const [br, gz] = await Promise.all([
          brotli(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: constants.BROTLI_MAX_QUALITY, [constants.BROTLI_PARAM_SIZE_HINT]: bytes.length } }),
          zip(bytes, { level: constants.Z_BEST_COMPRESSION }),
        ]);
        if (br.length < bytes.length) await writeFile(`${file}.br`, br);
        if (gz.length < bytes.length) await writeFile(`${file}.gz`, gz);
      };
      await Promise.all(Object.keys(bundle).filter(name => /\.(js|wasm)$/.test(name)).map(name => compressed(resolve(options.dir ?? "", name))));
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
          if (request.method !== "GET" || !gameFolders.some(folder => file.startsWith(folder))) throw new Error("not a game's file");
          const bytes = await readFile(file);
          response.setHeader("Content-Type", file.endsWith(".json") ? "application/json; charset=utf-8" : "application/octet-stream");
          response.setHeader("Cache-Control", "no-store");
          response.end(bytes);
        } catch { response.statusCode = 404; response.end("Source file not found"); }
      });
    },
  }],
  resolve: {
    alias: clients.map(({ wasm }) => ({
      find: wasmImport(wasm), replacement: resolve(import.meta.dirname, `${instrumented.has(mode) ? `pkg-${instrumented.get(mode)}` : command === "serve" ? "pkg-dev" : "pkg"}/${wasm}.js`),
    })),
  },
  build: {
    outDir: instrumented.has(mode) ? `dist-${instrumented.get(mode)}` : "dist",
    rollupOptions: {
      input: Object.fromEntries(games.map(({ game, page }) => [game, resolve(import.meta.dirname, page)])),
    },
  },
}));
