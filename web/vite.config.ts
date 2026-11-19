import { defineConfig } from "vite";
import { createHash } from "node:crypto";
import { dirname, extname, relative, resolve, sep } from "node:path";
import { json } from "node:stream/consumers";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { brotliCompress, constants, gzip } from "node:zlib";
import clients from "./src/generated/clients.json";

const instrumented = new Map([["profiling", "profile"]]);
const selected = process.env.ENGINE_GAME;
const games = clients.flatMap(({ wasm, game, page, folder }) => game && page && (!selected || game === selected) ? [{ wasm, game, page, folder }] : []);
const workspace = resolve(import.meta.dirname, "..");
const gameFolders = clients.map(client => resolve(workspace, client.folder) + sep);

const jsonStrings = /"(?:[^"\\]|\\.)*"/g;
const wasmImport = (name: string) => new RegExp(`^(?:\\.\\./)+pkg/${name}$`);

async function shipScenes(out: string, folder: string): Promise<string[]> {
  const shipped = new Map<string, Promise<string>>();
  const written: string[] = [];
  const isFile = (file: string) => stat(file).then(facts => facts.isFile(), () => false);
  const renamed = async (text: string, from: string[]): Promise<string> => {
    let kept = "";
    let last = 0;
    for (const match of text.matchAll(jsonStrings)) {
      const file = resolve(folder, JSON.parse(match[0]) as string);
      if (!file.startsWith(folder + sep) || !await isFile(file)) continue;
      kept += text.slice(last, match.index) + JSON.stringify(await ship(relative(folder, file).split(sep).join("/"), from));
      last = match.index + match[0].length;
    }
    return kept + text.slice(last);
  };
  const ship = (path: string, from: string[]): Promise<string> => {
    if (from.includes(path)) throw new Error(`${[...from, path].join(" -> ")}: files that name each other cannot be content-hashed`);
    const known = shipped.get(path);
    if (known) return known;
    const shipping = (async () => {
      const raw = await readFile(resolve(folder, path));
      const bytes = path.endsWith(".json") ? Buffer.from(await renamed(raw.toString("utf8"), [...from, path])) : raw;
      const extension = extname(path);
      const name = `${path.slice(0, path.length - extension.length)}.${createHash("sha256").update(bytes).digest("hex").slice(0, 16)}${extension}`;
      const target = resolve(out, name);
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, bytes);
      written.push(target);
      return name;
    })();
    shipped.set(path, shipping);
    return shipping;
  };
  for (const entry of await readdir(folder, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".scene.json")) continue;
    const target = resolve(out, entry.name);
    const text = await renamed(await readFile(resolve(folder, entry.name), "utf8"), [entry.name]);
    await mkdir(out, { recursive: true });
    await writeFile(target, text);
    written.push(target);
  }
  return written;
}

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
      for (const { game, folder } of games) {
        for (const file of await shipScenes(resolve(options.dir ?? "", game), resolve(workspace, folder))) await compressed(file);
      }
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
      find: wasmImport(wasm), replacement: resolve(import.meta.dirname, `pkg-${instrumented.get(mode) ?? (command === "serve" ? "dev" : "ship")}/${wasm}.js`),
    })),
  },
  build: {
    outDir: instrumented.has(mode) ? `dist-${instrumented.get(mode)}` : "dist",
    rollupOptions: {
      input: Object.fromEntries(games.map(({ game, page }) => [game, resolve(import.meta.dirname, page)])),
    },
  },
}));
