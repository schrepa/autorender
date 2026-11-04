// @ts-check
import { createServer } from "node:http";

export const VERSION = 1;
export const MAX_BYTES = 1024 * 1024;
const MAX_REQUEST_IDS = 65_536;
const ID_PATTERN = /^[A-Za-z0-9_.:-]{1,96}$/;

const DETAIL = ["expected", "allowed", "nearest", "got"];
const MAX_GOT = 256;
const REQUEST_FIELDS = ["version", "id", "session", "command"];

export class AgentError extends Error {
  /**
   * @param {string} code
   * @param {string} [path]
   * @param {{ expected?: string, allowed?: object, nearest?: unknown, got?: unknown }} [detail]
   */
  constructor(code, path = "$", detail = {}) {
    super(code);
    this.code = code;
    this.path = path;
    this.detail = detail;
  }
}

export const atMost = max => ({ kind: "at_most", max });
export const range = (min, max) => ({ kind: "range", min, max });
export const variants = values => ({ kind: "variants", values: [...values].map(String) });

export function detailOf(error) {
  if (error instanceof AgentError) return error.detail;
  if (typeof error?.code === "string" && typeof error?.path === "string") {
    return Object.fromEntries(DETAIL.filter(key => error[key] !== undefined).map(key => [key, error[key]]));
  }
  return { got: String(error?.message ?? error) };
}

export function refusal(error, code = "bridge_error") {
  const { got, ...detail } = detailOf(error);
  return {
    code: String(error?.code ?? code).slice(0, 96),
    path: String(error?.path ?? "$").slice(0, 256),
    ...detail,
    ...(got === undefined ? {} : { got: typeof got === "string" ? got.slice(0, MAX_GOT) : got }),
  };
}

export function failure(error) {
  return { ok: false, error: refusal(error) };
}

export function loopbackUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new AgentError("invalid_url", "url", { expected: "uri", got: value }); }
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.username || url.password) {
    throw new AgentError("invalid_url", "url", { allowed: variants(["http://127.0.0.1", "http://localhost", "http://[::1]"]), got: url.origin });
  }
  url.searchParams.set("engine_dev", "1");
  return url;
}

export function closing(command) {
  return command?.verb === "control" && command?.name === "close";
}

function envelope(request, session, response) {
  return { ...response, version: VERSION, id: typeof request?.id === "string" && ID_PATTERN.test(request.id) ? request.id : null, session };
}

function validate(request, session) {
  if (!request || typeof request !== "object" || Array.isArray(request)) throw new AgentError("invalid_request", "$", { expected: "object" });
  if (request.version !== VERSION) throw new AgentError("unsupported_version", "version", { allowed: variants([VERSION]), got: request.version });
  if (typeof request.id !== "string" || !ID_PATTERN.test(request.id)) throw new AgentError("invalid_request_id", "id", { expected: "id", got: request.id });
  if (request.session !== session) throw new AgentError("session_mismatch", "session", { expected: "matching", got: request.session });
  if (!request.command || typeof request.command !== "object" || Array.isArray(request.command) || typeof request.command.verb !== "string") {
    throw new AgentError("invalid_command", "command.verb", { expected: "string" });
  }
  for (const key of Object.keys(request)) {
    if (!REQUEST_FIELDS.includes(key)) throw new AgentError("invalid_request", key, { allowed: variants(REQUEST_FIELDS), got: key });
  }
}

async function readRequest(request) {
  if (Number(request.headers["content-length"]) > MAX_BYTES) throw new AgentError("request_too_large", "$", { allowed: atMost(MAX_BYTES), got: Number(request.headers["content-length"]) });
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BYTES) throw new AgentError("request_too_large", "$", { allowed: atMost(MAX_BYTES), got: size });
    chunks.push(chunk);
  }
  try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { throw new AgentError("invalid_json", "$", { expected: "json" }); }
}

/** @param {{ session: string, url: string, build?: string, execute: (command: any, onRecord?: (record: any) => void) => any, onClose: () => any, development?: () => any, initiallyUnavailable?: boolean, deadlineMs?: number, queueLimit?: number }} options */
export function createBridge({ session, url, build, execute, onClose, development, initiallyUnavailable = false, deadlineMs = 15_000, queueLimit = 16 }) {
  let startedAt = new Date().toISOString();
  const requestIds = new Set();
  const queue = [];
  let running = false;
  let unavailable = initiallyUnavailable;
  let replacing = false;
  let closed = false;

  async function pump() {
    if (running) return;
    running = true;
    while (queue.length) {
      const item = queue.shift();
      if (item.replacement) {
        try {
          if (closed) throw new AgentError("session_closed", "$", { expected: "live" });
          if (item.replacement.validate && !await item.replacement.validate()) throw new AgentError("build_superseded", "$", { expected: "matching" });
          item.replacement.commit?.();
          ({ session, url, build, execute } = item.replacement);
          startedAt = new Date().toISOString();
          requestIds.clear();
          unavailable = false;
          item.resolve();
        } catch (error) { item.reject(error); }
        finally { replacing = false; }
        continue;
      }
      const acceptedSession = session;
      const acceptedBuild = build;
      let timer;
      let response;
      try {
        const closes = closing(item.request.command) && onClose;
        if (unavailable && !closes) throw new AgentError("session_unavailable", "$", { expected: "live" });
        const remaining = item.deadline - Date.now();
        if (remaining <= 0) throw new AgentError("queue_timeout", "$", { allowed: atMost(deadlineMs) });
        if (closes) {
          if (Object.keys(item.request.command).some(key => !["verb", "name"].includes(key))) throw new AgentError("invalid_command", "command.args", { expected: "absent" });
          unavailable = true;
          closed = true;
          response = { ok: true, result: { state: "closing" } };
        } else {
          const { command } = item.request;
          let expired = false;
          response = await Promise.race([
            Promise.resolve().then(() => execute(command, record => { if (!expired) item.onRecord?.(record); })),
            new Promise((_, reject) => {
              timer = setTimeout(() => {
                expired = true;
                reject(new AgentError("command_timeout", "$", { allowed: atMost(remaining), got: { verb: command.verb, name: command.name, running: true } }));
              }, remaining);
            }),
          ]);
        }
        if (!response || typeof response.ok !== "boolean") throw new AgentError("invalid_browser_response", "$", { expected: "object" });
        if (Buffer.byteLength(JSON.stringify(response)) > MAX_BYTES - 1024) throw new AgentError("response_too_large", "$", { allowed: atMost(MAX_BYTES - 1024) });
      } catch (error) { response = failure(error); }
      finally { clearTimeout(timer); }
      item.resolve(envelope(item.request, acceptedSession, { ...response, ...(acceptedBuild === undefined ? {} : { build: acceptedBuild }) }));
    }
    running = false;
  }

  function submit(request, onRecord) {
    validate(request, session);
    if (replacing) throw new AgentError("session_replacing", "$", { expected: "live" });
    const closes = closing(request.command) && onClose;
    if (unavailable && !closes) throw new AgentError("session_unavailable", "$", { expected: "live" });
    if (requestIds.has(request.id)) throw new AgentError("duplicate_request", "id", { expected: "unique", got: request.id });
    if (requestIds.size >= MAX_REQUEST_IDS && !closes) throw new AgentError("session_request_limit", "$", { allowed: atMost(MAX_REQUEST_IDS) });
    if (queue.length + Number(running) >= queueLimit) throw new AgentError("queue_full", "$", { allowed: atMost(queueLimit) });
    requestIds.add(request.id);
    return new Promise(resolve => {
      queue.push({ request, onRecord, deadline: Date.now() + deadlineMs, resolve });
      void pump();
    });
  }

  function send(response, status, value) {
    if (response.destroyed) return;
    response.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
    response.end(JSON.stringify(value));
  }

  const server = createServer({ requestTimeout: 5_000, headersTimeout: 5_000, connectionsCheckingInterval: 1000 }, async (request, response) => {
    let parsed;
    try {
      const address = /** @type {import("node:net").AddressInfo} */ (server.address());
      if (request.headers.host !== `127.0.0.1:${address.port}` || request.headers.origin !== undefined) {
        throw new AgentError("origin_rejected", "$", { allowed: variants([`127.0.0.1:${address.port}`]), got: request.headers.host });
      }
      if (request.method === "GET" && request.url === "/sessions") {
        send(response, 200, { version: VERSION, sessions: [{ session, url, ...(build === undefined ? {} : { build }), started_at: startedAt, state: unavailable ? "unavailable" : "ready" }] });
        return;
      }
      if (request.method === "GET" && request.url === "/development" && development) {
        const result = { version: VERSION, ok: true, result: development() };
        if (Buffer.byteLength(JSON.stringify(result)) > MAX_BYTES) throw new AgentError("response_too_large", "$", { allowed: atMost(MAX_BYTES) });
        send(response, 200, result);
        return;
      }
      if (request.method !== "POST" || request.url !== "/request") {
        const routes = ["GET /sessions", ...(development ? ["GET /development"] : []), "POST /request"];
        send(response, 404, envelope(null, session, failure(new AgentError("route_not_found", "$", { allowed: variants(routes), got: `${request.method} ${request.url}` }))));
        return;
      }
      if (request.headers["content-type"]?.split(";")[0].trim().toLowerCase() !== "application/json") throw new AgentError("content_type", "$", { allowed: variants(["application/json"]), got: request.headers["content-type"] });
      parsed = await readRequest(request);
      if (parsed?.command?.verb === "watch") {
        const held = [];
        let open = false;
        const write = line => { if (open) response.write(line); else held.push(line); };
        const pending = submit(parsed, record => write(`${JSON.stringify({ record })}\n`));
        response.writeHead(200, { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
        open = true;
        for (const line of held.splice(0)) response.write(line);
        const result = await pending;
        response.end(`${JSON.stringify(result)}\n`);
        return;
      }
      const result = await submit(parsed);
      if (closing(parsed.command) && result.ok && onClose) {
        response.once("finish", () => { Promise.resolve().then(onClose).catch(() => {}); });
      }
      send(response, 200, result);
    } catch (error) {
      send(response, error?.code === "request_too_large" ? 413 : error?.code === "origin_rejected" ? 403 : 400, envelope(parsed, session, failure(error)));
    }
  });
  server.maxHeadersCount = 32;
  return {
    server, submit,
    markUnavailable: () => { unavailable = true; },
    replaceSession(next) {
      if (closed || replacing) return Promise.reject(new AgentError("session_replacing", "$", { expected: "live" }));
      replacing = true;
      return new Promise((resolve, reject) => {
        queue.push({ replacement: next, resolve, reject });
        void pump();
      });
    },
  };
}
