import { defineConfig } from "vite";
import { createHash } from "node:crypto";
import { dirname, extname, relative, resolve, sep } from "node:path";
import { existsSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { json } from "node:stream/consumers";
import { createReadStream } from "node:fs";
import { mkdir, readdir, stat } from "node:fs/promises";
import { promisify } from "node:util";
import { brotliCompress, constants, gzip } from "node:zlib";
import clients from "./src/generated/clients.json";
import { wasmImport } from "./scripts/wasm-alias.mjs";

const instrumented = new Map([["profiling", "profile"]]);
const selected = process.env.ENGINE_GAME;
const games = clients.flatMap(({ wasm, game, page, folder }) => game && page && (!selected || game === selected) ? [{ wasm, game, page, folder }] : []);
const publicFolder = resolve(import.meta.dirname, "public");
const packIndex = resolve(publicFolder, "cooked/index.json");
const workspace = resolve(import.meta.dirname, "..");
const gameFolders = clients.map(client => resolve(workspace, client.folder) + sep);

const jsonStrings = /"(?:[^"\\]|\\.)*"/g;

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
  define: { __PUBLIC_PACKS__: JSON.stringify(existsSync(packIndex)) },
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
          if (request.method !== "GET" || !gameFolders.some(folder => file.startsWith(folder))) throw new Error("not a game's file");
          const bytes = await readFile(file);
          response.setHeader("Content-Type", file.endsWith(".json") ? "application/json; charset=utf-8" : file.endsWith(".png") ? "image/png" : file.endsWith(".jpg") ? "image/jpeg" : "application/octet-stream");
          response.setHeader("Cache-Control", "no-store");
          response.end(bytes);
        } catch { response.statusCode = 404; response.end("Source file not found"); }
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
