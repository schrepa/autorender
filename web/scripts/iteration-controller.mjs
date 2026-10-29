// @ts-check
import { randomUUID } from "node:crypto";
import { AgentError, atMost, refusal } from "./agent-protocol.mjs";

const MAX_RESULT_BYTES = 512 * 1024;

class IterationError extends AgentError {}

function failure(error) {
  return refusal(error, "iteration_error");
}

function source(snapshot) {
  if (!snapshot || typeof snapshot.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(snapshot.fingerprint)
    || !Number.isInteger(snapshot.files) || snapshot.files < 0 || snapshot.files > 20_000) {
    throw new IterationError("invalid_fingerprint", "$", { got: snapshot });
  }
  return { fingerprint: snapshot.fingerprint, files: snapshot.files };
}

function buildResult(result) {
  if (!result || typeof result !== "object" || typeof result.ok !== "boolean") {
    throw new IterationError("invalid_build_result", "$.ok", { expected: "boolean" });
  }
  const json = JSON.stringify(result);
  if (Buffer.byteLength(json) > MAX_RESULT_BYTES) {
    throw new IterationError("build_result_too_large", "$", { allowed: atMost(MAX_RESULT_BYTES), got: Buffer.byteLength(json) });
  }
  return JSON.parse(json);
}

export class IterationController {
  #operations;
  #desired;
  #sequence = 0;
  #runner;
  #closed = false;
  #state = "idle";
  #active;
  #latest;
  #abort;

  /** @param {{ build: any, prepare: any, promote: any, discard: any, fingerprint: any, onEvent?: (status: any) => void, signal?: AbortSignal }} operations */
  constructor({ build, prepare, promote, discard, fingerprint, onEvent = () => {}, signal }) {
    for (const callback of [build, prepare, promote, discard, fingerprint, onEvent]) {
      if (typeof callback !== "function") throw new TypeError("Iteration operations must be functions.");
    }
    this.#operations = { build, prepare, promote, discard, fingerprint, onEvent, signal };
    this.#abort = () => { void this.stop(); };
    if (signal?.aborted) {
      this.#closed = true;
      this.#state = "stopped";
    } else signal?.addEventListener("abort", this.#abort, { once: true });
  }

  enqueue(snapshot) {
    if (this.#closed) return;
    snapshot = source(snapshot);
    if (snapshot.fingerprint === this.#desired?.fingerprint) return;
    this.#desired = snapshot;
    this.#sequence++;
    this.#start();
  }

  status() {
    return JSON.parse(JSON.stringify({ state: this.#state, active: this.#active, latest: this.#latest }));
  }

  async whenIdle() {
    while (this.#runner) await this.#runner;
  }

  stop() {
    if (!this.#closed) {
      this.#closed = true;
      this.#desired = undefined;
      this.#state = "stopped";
      this.#operations.signal?.removeEventListener("abort", this.#abort);
      this.#emit();
    }
    return this.whenIdle();
  }

  #emit() {
    try { Promise.resolve(this.#operations.onEvent(this.status())).catch(() => {}); } catch {}
  }

  #stage(state, extra = {}) {
    this.#latest = { ...this.#latest, ...extra, state };
    if (!this.#closed) this.#state = state;
    this.#emit();
  }

  #start() {
    if (this.#runner || this.#closed || !this.#desired) return;
    if (this.#desired.fingerprint === this.#active?.fingerprint) {
      this.#state = "ready";
      this.#emit();
      return;
    }
    this.#runner = Promise.resolve().then(() => this.#run()).finally(() => {
      this.#runner = undefined;
      if (this.#desired?.fingerprint !== this.#latest?.fingerprint) this.#start();
    });
  }

  async #fresh(snapshot) {
    if (this.#closed) return false;
    const sequence = this.#sequence;
    const current = source(await this.#operations.fingerprint());
    if (this.#closed) return false;
    if (sequence === this.#sequence) this.#desired = current;
    return current.fingerprint === snapshot.fingerprint && this.#desired?.fingerprint === snapshot.fingerprint;
  }

  async #discard(attempt, candidate) {
    try { await this.#operations.discard(attempt, candidate); }
    catch (error) { this.#stage(this.#latest.state, { cleanup_error: failure(error) }); }
  }

  async #run() {
    while (!this.#closed && this.#desired && this.#desired.fingerprint !== this.#active?.fingerprint) {
      const snapshot = this.#desired;
      const attempt = Object.freeze({ id: randomUUID(), ...snapshot });
      this.#latest = { build: attempt.id, fingerprint: snapshot.fingerprint, files: snapshot.files, state: "building" };
      this.#stage("building");
      let candidate;
      let committed = false;
      let activationOpen = false;
      try {
        if (this.#closed) throw new IterationError("iteration_stopped", "$", { expected: "running" });
        const result = buildResult(await this.#operations.build(attempt, snapshot));
        this.#latest.result = result;
        if (!await this.#fresh(snapshot)) {
          this.#stage(this.#closed ? "stopped" : "superseded");
        } else if (!result.ok) {
          this.#stage("failed", result.error ? { error: failure(result.error) } : {});
        } else {
          this.#stage("preparing");
          candidate = await this.#operations.prepare(attempt, result);
          if (!candidate || typeof candidate.session !== "string" || !/^[A-Za-z0-9_.:-]{1,96}$/.test(candidate.session)) {
            throw new IterationError("invalid_candidate", "$.session", { expected: "id", got: candidate?.session });
          }
          if (!await this.#fresh(snapshot)) {
            this.#stage(this.#closed ? "stopped" : "superseded");
          } else {
            this.#stage("activating");
            activationOpen = true;
            try { await this.#operations.promote(candidate, snapshot, () => {
              if (!activationOpen) throw new IterationError("activation_expired", "$", { expected: "live" });
              if (committed) throw new IterationError("activation_committed", "$", { expected: "unique" });
              if (this.#closed) throw new IterationError("iteration_stopped", "$", { expected: "running" });
              if (this.#desired?.fingerprint !== snapshot.fingerprint) {
                throw new IterationError("build_superseded", "$", { expected: "matching", got: snapshot.fingerprint });
              }
              committed = true;
              this.#active = { build: attempt.id, session: candidate.session, fingerprint: snapshot.fingerprint };
              this.#stage("ready");
            }); } finally { activationOpen = false; }
            if (!committed) throw new IterationError("activation_not_committed", "$", { expected: "present" });
          }
        }
      } catch (error) {
        if (committed) {
          this.#stage("ready", { retirement_error: failure(error) });
        } else if (this.#closed) {
          this.#stage("stopped", { error: failure(error) });
        } else if (["build_superseded", "source_changed"].includes(error?.code)) {
          this.#stage("superseded", { error: failure(error) });
          try { await this.#fresh(snapshot); }
          catch (scanError) { this.#stage("failed", { error: failure(scanError) }); }
        } else this.#stage("failed", { error: failure(error) });
      } finally {
        if (!committed) await this.#discard(attempt, candidate);
      }
      if (this.#desired?.fingerprint === snapshot.fingerprint) break;
    }
  }
}
