import { defineConfig } from "vite";
import { relative, resolve, sep } from "node:path";
import { createReadStream, existsSync, readdirSync } from "node:fs";
import { readFile, stat, writeFile } from "node:fs/promises";
import { json } from "node:stream/consumers";
import { pathToFileURL } from "node:url";
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
const pagePlugins = clients.map(({ game }) => resolve(import.meta.dirname, "src", game, "vite.mjs")).filter(existsSync);
const ownFolders = games.map(({ game }) => resolve(publicFolder, game));
const ownFiles = ownFolders.filter(existsSync).flatMap(folder => readdirSync(folder, { recursive: true, withFileTypes: true })
  .filter(entry => entry.isFile())
  .map(entry => relative(publicFolder, resolve(entry.parentPath, entry.name)).split(sep).join("/")));

export default defineConfig(async ({ command, mode }) => ({
  define: { __PUBLIC_PACKS__: JSON.stringify(existsSync(packIndex)), __GAME_FILES__: JSON.stringify(ownFiles) },
  server: { watch: { ignored: ["**/public/scenes/**"] } },
  plugins: [...(await Promise.all(pagePlugins.map(async file => (await import(pathToFileURL(file).href)).default))).flat(), {
    name: "explicit-scene-reload",
    configureServer(server) {
      const types = new Map([["json", "application/json; charset=utf-8"], ["png", "image/png"], ["jpg", "image/jpeg"],
        ["jpeg", "image/jpeg"], ["webp", "image/webp"], ["ktx2", "image/ktx2"]]);
      server.middlewares.use(async (request, response, next) => {
        const path = request.url?.split("?")[0];
        const scene = !!path && /^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(path);
        const asset = !!path && /^\/scenes\/[A-Za-z0-9_-][A-Za-z0-9_\/-]*(\.[A-Za-z0-9_-]+)+$/.test(path)
          && !path.split("/").some(part => part === "." || part === "..");
        if (!path || (!scene && !asset)) return next();
        if (request.method !== "GET" && request.method !== "HEAD") { response.statusCode = 405; response.end(); return; }
        const limit = scene ? 64 * 1024 * 1024 : 1024 * 1024 * 1024;
        try {
          const file = resolve(import.meta.dirname, "public", path.slice(1));
          const metadata = await stat(file);
          if (!metadata.isFile()) { response.statusCode = 404; response.end(); return; }
          if (metadata.size > limit) { response.statusCode = 413; response.end("Scene file exceeds its size bound"); return; }
          response.setHeader("Content-Type", types.get(path.split(".").pop()!.toLowerCase()) ?? "application/octet-stream");
          response.setHeader("Cache-Control", "no-store");
          response.setHeader("Accept-Ranges", "bytes");
          const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? "");
          let [start, end] = [0, limit];
          if (range) {
            start = Number(range[1]);
            end = Math.min(range[2] ? Number(range[2]) : metadata.size - 1, metadata.size - 1);
            if (start > end) {
              response.statusCode = 416;
              response.setHeader("Content-Range", `bytes */${metadata.size}`);
              response.end();
              return;
            }
            response.statusCode = 206;
            response.setHeader("Content-Range", `bytes ${start}-${end}/${metadata.size}`);
            response.setHeader("Content-Length", String(end - start + 1));
          }
          if (request.method === "HEAD") { response.end(); return; }
          const stream = createReadStream(file, { start, end });
          stream.on("error", () => response.destroy());
          response.on("close", () => stream.destroy());
          stream.pipe(response);
        } catch { response.statusCode = 404; response.end("Scene source not found"); }
      });
    },
  }, {
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
        const file = resolve(path);
        const listed = file === packIndex || ownFolders.some(folder => file.startsWith(folder + sep));
        if ((event === "add" || event === "unlink") && listed) void server.restart();
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
