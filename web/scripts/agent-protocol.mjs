// @ts-check
import { createServer } from "node:http";

export const VERSION = 1;
export const MAX_BYTES = 1024 * 1024;
const MAX_REQUEST_IDS = 65_536;
const DEADLINE_MS = 15_000;
const QUEUE_LIMIT = 16;
const ID_PATTERN = /^[A-Za-z0-9_.:-]{1,96}$/;

const DETAIL = ["expected", "allowed", "nearest", "got"];
const MAX_GOT = 4096;
const REQUEST_FIELDS = ["version", "id", "session", "command", "timeout_ms", "brief"];
const CLOSE_FIELDS = ["verb", "name", "version", "args"];
const FETCH_HEADERS_LIMIT_MS = 300_000;
export const MAX_TIMEOUT_MS = FETCH_HEADERS_LIMIT_MS - 5_000;

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

/** @param {string[]} args @param {string[]} [flags] */
export function options(args, flags = []) {
  const result = {};
  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    const name = key.slice(2);
    if (!key.startsWith("--") || name in result) throw new AgentError("invalid_arguments", key, { got: key });
    if (flags.includes(name)) { result[name] = true; continue; }
    const value = args[++i];
    if (!value || value.startsWith("--")) throw new AgentError("invalid_arguments", key, { got: key });
    result[name] = value;
  }
  return result;
}

export function checkOptions(values, names) {
  for (const key of Object.keys(values)) if (!names.includes(key)) throw new AgentError("invalid_arguments", `--${key}`, { allowed: variants(names.map(name => `--${name}`)), got: `--${key}` });
}

/** @param {string} text */
export function isNotice(text) {
  try { return JSON.parse(text)?.level === "notice"; } catch { return false; }
}

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

function checkClose(command) {
  const unknown = Object.keys(command).find(key => !CLOSE_FIELDS.includes(key));
  if (unknown) throw new AgentError("invalid_command", `command.${unknown}`, { allowed: variants(CLOSE_FIELDS), got: unknown });
  const { version, args } = command;
  if (version !== undefined && version !== VERSION) throw new AgentError("unsupported_version", "command.version", { allowed: variants([VERSION]), got: version });
  if (args !== undefined && (!args || typeof args !== "object" || Array.isArray(args) || Object.keys(args).length)) {
    throw new AgentError("invalid_command", "command.args", { expected: "absent", got: args });
  }
}

export function checkTimeout(value) {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 1 || value > MAX_TIMEOUT_MS)) {
    throw new AgentError("invalid_request", "timeout_ms", { allowed: range(1, MAX_TIMEOUT_MS), got: value });
  }
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
  checkTimeout(request.timeout_ms);
  if (request.brief !== undefined && request.brief !== true) throw new AgentError("invalid_request", "brief", { allowed: variants([true]), got: request.brief });
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

/** @param {{ session: string, url: string, execute: (command: any) => any, activity?: () => any, onClose: () => any }} options */
export function createBridge({ session, url, execute, activity, onClose }) {
  const startedAt = new Date().toISOString();
  const requestIds = new Set();
  const queue = [];
  let running = false;
  /** @type {"ready" | "unavailable" | "closed"} */
  let state = "ready";
  let carrying;

  /** @param {"lost" | "close"} event */
  function move(event) {
    if (state !== "closed") state = event === "lost" ? "unavailable" : "closed";
  }

  const live = () => state === "ready";

  const answer = (request, response) => envelope(request, session, response);

  const carried = () => carrying && { verb: carrying.command.verb, name: carrying.command.name, running_ms: Date.now() - carrying.started };

  function carry(command) {
    const reply = Promise.resolve().then(() => execute(command));
    /** @type {{ command: any, started: number, finished: Promise<void> | undefined }} */
    const entry = { command, started: Date.now(), finished: undefined };
    const finish = () => { if (carrying === entry) carrying = undefined; };
    entry.finished = reply.then(finish, finish);
    carrying = entry;
    return reply;
  }

  async function overdue(command, limit) {
    const { verb, name } = command;
    const progress = carrying?.command === command
      ? { running: true, running_ms: Date.now() - carrying.started }
      : { running: false, behind: carried() };
    return new AgentError("command_timeout", "timeout_ms", { allowed: atMost(limit), got: { verb, name, ...progress, ...await activity?.() } });
  }

  async function pump() {
    if (running) return;
    running = true;
    while (queue.length) {
      const item = queue.shift();
      clearTimeout(item.expiry);
      let timer;
      let response;
      try {
        const { command } = item.request;
        let expired = false;
        const limit = new Promise((_, reject) => {
          timer = setTimeout(() => {
            expired = true;
            void overdue(command, item.limit).then(reject, reject);
          }, item.deadline - Date.now());
          timer.unref();
        });
        if (carrying) await Promise.race([carrying.finished, limit]);
        if (expired) await limit;
        if (!live()) throw new AgentError("session_unavailable", "$", { expected: "live" });
        response = await Promise.race([carry(command), limit]);
        if (!response || typeof response.ok !== "boolean") throw new AgentError("invalid_browser_response", "$", { expected: "object" });
        if (item.request.brief === true && response.result) delete response.result.rendering;
        if (Buffer.byteLength(JSON.stringify(response)) > MAX_BYTES - 1024) throw new AgentError("response_too_large", "$", { allowed: atMost(MAX_BYTES - 1024) });
      } catch (error) { response = failure(error); }
      finally { clearTimeout(timer); }
      item.resolve(answer(item.request, response));
    }
    running = false;
  }

  function submit(request) {
    validate(request, session);
    const closes = closing(request.command) && onClose;
    if (!live() && !closes) throw new AgentError("session_unavailable", "$", { expected: "live" });
    if (requestIds.has(request.id)) throw new AgentError("duplicate_request", "id", { expected: "unique", got: request.id });
    if (closes) {
      checkClose(request.command);
      move("close");
      return Promise.resolve(answer(request, { ok: true, result: { state: "closing" } }));
    }
    if (requestIds.size >= MAX_REQUEST_IDS) throw new AgentError("session_request_limit", "$", { allowed: atMost(MAX_REQUEST_IDS) });
    if (queue.length + Number(running) >= QUEUE_LIMIT) throw new AgentError("queue_full", "$", { allowed: atMost(QUEUE_LIMIT) });
    requestIds.add(request.id);
    const limit = request.timeout_ms ?? DEADLINE_MS;
    return new Promise(resolve => {
      /** @type {{ request: any, limit: number, deadline: number, resolve: (value: any) => void, expiry: NodeJS.Timeout | undefined }} */
      const item = { request, limit, deadline: Date.now() + limit, resolve, expiry: undefined };
      item.expiry = setTimeout(() => {
        queue.splice(queue.indexOf(item), 1);
        const { verb, name } = request.command;
        resolve(answer(request, failure(new AgentError("queue_timeout", "timeout_ms", { allowed: atMost(limit), got: { verb, name, running: false, behind: carried() } }))));
      }, limit);
      item.expiry.unref();
      queue.push(item);
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
        send(response, 200, { version: VERSION, sessions: [{ session, url, started_at: startedAt, state: live() ? "ready" : "unavailable" }] });
        return;
      }
      if (request.method !== "POST" || request.url !== "/request") {
        send(response, 404, envelope(null, session, failure(new AgentError("route_not_found", "$", { allowed: variants(["GET /sessions", "POST /request"]), got: `${request.method} ${request.url}` }))));
        return;
      }
      if (request.headers["content-type"]?.split(";")[0].trim().toLowerCase() !== "application/json") throw new AgentError("content_type", "$", { allowed: variants(["application/json"]), got: request.headers["content-type"] });
      parsed = await readRequest(request);
      const result = await submit(parsed);
      if (closing(parsed.command) && result.ok && onClose) {
        response.once("close", () => { Promise.resolve().then(onClose).catch(() => {}); });
      }
      send(response, 200, result);
    } catch (error) {
      send(response, error?.code === "request_too_large" ? 413 : error?.code === "origin_rejected" ? 403 : 400, envelope(parsed, session, failure(error)));
    }
  });
  server.maxHeadersCount = 32;
  return { server, markUnavailable: () => move("lost") };
}
