import { defineConfig } from "vite";
import { resolve } from "node:path";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
// The browser clients, generated from xtask (`cargo xtask schema`).
import clients from "./src/generated/clients.json";

// Instrumented Stress Lab builds (`cargo xtask profile [--attribution]`) by
// mode: the suffix of their binding and output directories.
const instrumented = new Map([["profiling", "profile"], ["attribution", "attribution"]]);

export default defineConfig(({ command, mode }) => ({
  // Authored saves become visible only through an explicit revision-checked reload.
  server: { watch: { ignored: ["**/public/scenes/**"] } },
  plugins: [{
    name: "explicit-scene-reload",
    // Vite caches public file names. Read scenes and the image, buffer and
    // virtual mesh files they reference directly
    // so newly authored files work while source changes remain excluded from
    // automatic reload.
    configureServer(server) {
      const types = new Map([["json", "application/json; charset=utf-8"], ["png", "image/png"], ["jpg", "image/jpeg"],
        ["jpeg", "image/jpeg"], ["webp", "image/webp"], ["ktx2", "image/ktx2"], ["bin", "application/octet-stream"],
        ["evg", "application/octet-stream"], ["etr", "application/octet-stream"], ["r16", "application/octet-stream"]]);
      server.middlewares.use(async (request, response, next) => {
        const path = request.url?.split("?")[0];
        const scene = !!path && /^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(path);
        const asset = !!path && /^\/scenes\/[A-Za-z0-9_-][A-Za-z0-9_\/-]*(\.[A-Za-z0-9_-]+)*\.(png|jpe?g|webp|ktx2|bin|evg|etr|r16)$/.test(path)
          && !path.split("/").some(part => part === "." || part === "..");
        if (!path || (!scene && !asset)) return next();
        if (request.method !== "GET" && request.method !== "HEAD") { response.statusCode = 405; response.end(); return; }
        // Sanity bounds: scene JSON and one referenced file.
        const limit = scene ? 64 * 1024 * 1024 : 1024 * 1024 * 1024;
        try {
          const file = resolve(import.meta.dirname, "public", path.slice(1));
          const metadata = await stat(file);
          if (!metadata.isFile()) { response.statusCode = 404; response.end(); return; }
          if (metadata.size > limit) { response.statusCode = 413; response.end("Scene file exceeds its size bound"); return; }
          response.setHeader("Content-Type", types.get(path.split(".").pop()!.toLowerCase()) ?? "application/octet-stream");
          response.setHeader("Cache-Control", "no-store");
          response.setHeader("Accept-Ranges", "bytes");
          // One byte range (virtual meshes stream their pages).
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
          // Include at most one overflow byte if the file grows after stat.
          const stream = createReadStream(file, { start, end });
          stream.on("error", () => response.destroy());
          response.on("close", () => stream.destroy());
          stream.pipe(response);
        } catch { response.statusCode = 404; response.end("Scene source not found"); }
      });
    },
  }],
  resolve: {
    alias: instrumented.has(mode) ? [{
      find: "../pkg/stress_lab", replacement: resolve(import.meta.dirname, `pkg-${instrumented.get(mode)}/stress_lab.js`),
    }] : command === "serve" ? clients.map(client => client.wasm).map(name => ({
      find: `../pkg/${name}`, replacement: resolve(import.meta.dirname, `pkg-dev/${name}.js`),
    })) : [],
  },
  // Job workers are modules: they import the engine's job
  // worker bindings as the page imports the game's.
  worker: { format: "es" },
  build: {
    outDir: instrumented.has(mode) ? `dist-${instrumented.get(mode)}` : "dist",
    rollupOptions: {
      input: instrumented.has(mode) ? { stress: resolve(import.meta.dirname, "stress.html") } : {
        stress: resolve(import.meta.dirname, "stress.html"),
      },
    },
  },
}));
