/** Bounded, loopback-only transport for an explicitly enabled engine dev session. */
import { createServer } from "node:http";

export const VERSION = 1;
export const MAX_BYTES = 1024 * 1024;
const MAX_REQUEST_IDS = 65_536;
const ID_PATTERN = /^[A-Za-z0-9_.:-]{1,96}$/;

export class AgentError extends Error {
  constructor(code, message, path = "$") {
    super(message);
    this.code = code;
    this.path = path;
  }
}

export function failure(error) {
  return { ok: false, error: {
    code: String(error?.code ?? "bridge_error").slice(0, 96),
    path: String(error?.path ?? "$").slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4000),
  } };
}

export function loopbackUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new AgentError("invalid_url", "Expected an absolute loopback HTTP URL.", "url"); }
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.username || url.password) {
    throw new AgentError("invalid_url", "The development URL must use HTTP on localhost, 127.0.0.1, or [::1], without credentials.", "url");
  }
  url.searchParams.set("engine_dev", "1");
  return url;
}

function envelope(request, session, response) {
  return { ...response, version: VERSION, id: typeof request?.id === "string" && ID_PATTERN.test(request.id) ? request.id : null, session };
}

function validate(request, session) {
  if (!request || typeof request !== "object" || Array.isArray(request)) throw new AgentError("invalid_request", "Expected a request object.");
  if (request.version !== VERSION) throw new AgentError("protocol_version", "Only protocol version 1 is supported.", "version");
  if (typeof request.id !== "string" || !ID_PATTERN.test(request.id)) throw new AgentError("invalid_request_id", "Use 1–96 ASCII letters, digits, underscores, dots, colons, or hyphens.", "id");
  if (request.session !== session) throw new AgentError("session_mismatch", "This request names a different development session.", "session");
  if (!request.command || typeof request.command !== "object" || Array.isArray(request.command) || typeof request.command.op !== "string") {
    throw new AgentError("invalid_command", "Expected command.op to be a string.", "command");
  }
  for (const key of Object.keys(request)) {
    if (!["version", "id", "session", "command"].includes(key)) throw new AgentError("invalid_request", `Unknown request field: ${key}.`, key);
  }
}

async function readRequest(request) {
  if (Number(request.headers["content-length"]) > MAX_BYTES) throw new AgentError("request_too_large", "Request exceeds the 1 MiB limit.");
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > MAX_BYTES) throw new AgentError("request_too_large", "Request exceeds the 1 MiB limit.");
    chunks.push(chunk);
  }
  try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(chunks))); }
  catch { throw new AgentError("invalid_json", "Request must be valid UTF-8 JSON."); }
}

/** Injecting execute keeps transport and failure contracts testable without a GPU. */
export function createBridge({ session, url, build, execute, onTimeout = () => {}, onClose, development, initiallyUnavailable = false, deadlineMs = 15_000, queueLimit = 16 }) {
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
          if (closed) throw new AgentError("session_closed", "The bridge closed before replacement.");
          if (item.replacement.validate && !await item.replacement.validate()) throw new AgentError("build_superseded", "Sources changed before this build could be activated.");
          // Commit coordinator state synchronously at the same boundary as identity.
          item.replacement.commit?.();
          ({ session, url, build, execute, onTimeout = () => {} } = item.replacement);
          startedAt = new Date().toISOString();
          requestIds.clear();
          unavailable = false;
          item.resolve();
        } catch (error) { item.reject(error); }
        finally { replacing = false; }
        continue;
      }
      // A response always belongs to the session that accepted its command.
      const acceptedSession = session;
      const acceptedBuild = build;
      let timer;
      let response;
      try {
        const closing = item.request.command.op === "close" && onClose;
        if (unavailable && !closing) throw new AgentError("session_unavailable", "The session is unavailable; start a new bridge session.");
        const remaining = item.deadline - Date.now();
        if (remaining <= 0) throw new AgentError("queue_timeout", "The request expired before execution; it was not dispatched.");
        if (closing) {
          if (Object.keys(item.request.command).some(key => key !== "op")) throw new AgentError("invalid_command", "Close accepts only its op field.");
          unavailable = true;
          closed = true;
          response = { ok: true, result: { state: "closing" } };
        } else response = await Promise.race([
          Promise.resolve().then(() => execute(item.request.command)),
          new Promise((_, reject) => {
            timer = setTimeout(() => {
              unavailable = true;
              // A timed-out mutation has an uncertain outcome. Never overlap later work with it.
              Promise.resolve().then(onTimeout).catch(() => {});
              reject(new AgentError("command_timeout", "The command deadline expired. Its outcome is unknown; this session is now unavailable."));
            }, remaining);
          }),
        ]);
        if (!response || typeof response.ok !== "boolean") throw new AgentError("invalid_browser_response", "The browser returned an invalid development response.");
        if (Buffer.byteLength(JSON.stringify(response)) > MAX_BYTES - 1024) throw new AgentError("response_too_large", "The response exceeds the 1 MiB limit; narrow the inspection request.");
      } catch (error) { response = failure(error); }
      finally { clearTimeout(timer); }
      item.resolve(envelope(item.request, acceptedSession, { ...response, ...(acceptedBuild === undefined ? {} : { build: acceptedBuild }) }));
    }
    running = false;
  }

  function submit(request) {
    validate(request, session);
    if (replacing) throw new AgentError("session_replacing", "A validated build is replacing this session; discover the new session before sending commands.");
    const closing = request.command.op === "close" && onClose;
    if (unavailable && !closing) throw new AgentError("session_unavailable", "The session is unavailable; start a new bridge session.");
    if (requestIds.has(request.id)) throw new AgentError("duplicate_request", "This request ID was already accepted. Inspect the current state with a new ID; mutations are never replayed automatically.", "id");
    if (requestIds.size >= MAX_REQUEST_IDS && !closing) throw new AgentError("session_request_limit", "The session reached its 65,536-request limit; start a new session.");
    if (queue.length + Number(running) >= queueLimit) throw new AgentError("queue_full", "The session command queue is full; retry with a new request ID after pending work completes.");
    requestIds.add(request.id);
    return new Promise(resolve => {
      queue.push({ request, deadline: Date.now() + deadlineMs, resolve });
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
      const address = server.address();
      if (request.headers.host !== `127.0.0.1:${address.port}` || request.headers.origin !== undefined) {
        throw new AgentError("origin_rejected", "Use a local command-line client with the bridge's 127.0.0.1 address.");
      }
      if (request.method === "GET" && request.url === "/sessions") {
        send(response, 200, { version: VERSION, sessions: [{ session, url, ...(build === undefined ? {} : { build }), started_at: startedAt, state: unavailable ? "unavailable" : "ready" }] });
        return;
      }
      if (request.method === "GET" && request.url === "/development" && development) {
        const result = { version: VERSION, ok: true, result: development() };
        if (Buffer.byteLength(JSON.stringify(result)) > MAX_BYTES) throw new AgentError("response_too_large", "Development status exceeds the response limit.");
        send(response, 200, result);
        return;
      }
      if (request.method !== "POST" || request.url !== "/request") {
        send(response, 404, envelope(null, session, failure(new AgentError("route_not_found", "Use GET /sessions or POST /request."))));
        return;
      }
      if (request.headers["content-type"]?.split(";")[0].trim().toLowerCase() !== "application/json") throw new AgentError("content_type", "Requests must use Content-Type: application/json.");
      parsed = await readRequest(request);
      const result = await submit(parsed);
      if (parsed.command.op === "close" && result.ok && onClose) {
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
    /** Drain accepted work before swapping identity; reject arrivals during this boundary. */
    replaceSession(next) {
      if (closed || replacing) return Promise.reject(new AgentError("session_replacing", "The bridge is closing or already replacing a session."));
      replacing = true;
      return new Promise((resolve, reject) => {
        queue.push({ replacement: next, resolve, reject });
        void pump();
      });
    },
  };
}
