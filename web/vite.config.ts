import { defineConfig } from "vite";
import { resolve } from "node:path";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";

// Instrumented Stress Lab builds (`cargo xtask profile [--attribution]`) by
// mode: the suffix of their binding and output directories.
const instrumented = new Map([["profiling", "profile"], ["attribution", "attribution"]]);

export default defineConfig(({ command, mode }) => ({
  // Authored saves become visible only through an explicit revision-checked reload.
  server: { watch: { ignored: ["**/public/scenes/**"] } },
  plugins: [{
    name: "explicit-scene-reload",
    // Vite caches public file names. Read scenes directly so newly authored
    // files work while source changes remain excluded from automatic reload.
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const path = request.url?.split("?")[0];
        if (!path || !/^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(path)) return next();
        if (request.method !== "GET" && request.method !== "HEAD") { response.statusCode = 405; response.end(); return; }
        try {
          const file = resolve(import.meta.dirname, "public", path.slice(1));
          const metadata = await stat(file);
          if (!metadata.isFile()) { response.statusCode = 404; response.end(); return; }
          if (metadata.size > 1024 * 1024) { response.statusCode = 413; response.end("Scene exceeds 1 MiB"); return; }
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.setHeader("Cache-Control", "no-store");
          if (request.method === "HEAD") { response.end(); return; }
          // Include at most one overflow byte if the file grows after stat.
          const stream = createReadStream(file, { start: 0, end: 1024 * 1024 });
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
    }] : command === "serve" ? ["orbit_demo", "playground", "stress_lab"].map(name => ({
      find: `../pkg/${name}`, replacement: resolve(import.meta.dirname, `pkg-dev/${name}.js`),
    })) : [],
  },
  build: {
    outDir: instrumented.has(mode) ? `dist-${instrumented.get(mode)}` : "dist",
    rollupOptions: {
      input: instrumented.has(mode) ? { stress: resolve(import.meta.dirname, "stress.html") } : {
        index: resolve(import.meta.dirname, "index.html"),
        playground: resolve(import.meta.dirname, "playground.html"),
        stress: resolve(import.meta.dirname, "stress.html"),
      },
    },
  },
}));
