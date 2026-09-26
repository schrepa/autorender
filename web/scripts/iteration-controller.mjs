/** Serial, dependency-injected ownership of development build attempts.
 * Only the newest requested source fingerprint is retained while work runs.
 * Browser/process/filesystem policy lives in the supervisor adapters.
 */
import { randomUUID } from "node:crypto";

const MAX_RESULT_BYTES = 512 * 1024;

class IterationError extends Error {
  constructor(code, message, path = "$") {
    super(message);
    this.code = code;
    this.path = path;
  }
}

function failure(error) {
  return {
    code: String(error?.code ?? "iteration_error").slice(0, 96),
    path: String(error?.path ?? "$").slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4000),
  };
}

function source(snapshot) {
  if (!snapshot || typeof snapshot.fingerprint !== "string" || !/^[a-f0-9]{64}$/.test(snapshot.fingerprint)
    || !Number.isInteger(snapshot.files) || snapshot.files < 0 || snapshot.files > 20_000) {
    throw new IterationError("invalid_fingerprint", "Expected a SHA-256 source fingerprint and a bounded file count.");
  }
  return { fingerprint: snapshot.fingerprint, files: snapshot.files };
}

function buildResult(result) {
  if (!result || typeof result !== "object" || typeof result.ok !== "boolean") {
    throw new IterationError("invalid_build_result", "Build results must contain an ok boolean.");
  }
  const json = JSON.stringify(result);
  if (Buffer.byteLength(json) > MAX_RESULT_BYTES) {
    throw new IterationError("build_result_too_large", "Build results exceed the 512 KiB status budget.");
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

  /** Notifications are values: callers cannot mutate pending input or status. */
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

  /** Wait for all currently pending work, including coalesced follow-up work. */
  async whenIdle() {
    while (this.#runner) await this.#runner;
  }

  /** Prevent activation immediately, then wait for adapter work to settle.
   * The supervisor cancels processes through its own shared AbortSignal.
   */
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
    // Status observers cannot break build ownership or create unhandled
    // promise rejections. The adapter should log its own observer failures.
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
    // Deferral assigns #runner before adapters/observers can enqueue again.
    this.#runner = Promise.resolve().then(() => this.#run()).finally(() => {
      this.#runner = undefined;
      // Include a notification arriving after #run settled but before this
      // cleanup microtask, without retrying an unchanged failed build.
      if (this.#desired?.fingerprint !== this.#latest?.fingerprint) this.#start();
    });
  }

  async #fresh(snapshot) {
    if (this.#closed) return false;
    const sequence = this.#sequence;
    const current = source(await this.#operations.fingerprint());
    if (this.#closed) return false;
    // A later explicit notification takes precedence over an older scan.
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
        if (this.#closed) throw new IterationError("iteration_stopped", "Development iteration is stopped.");
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
            throw new IterationError("invalid_candidate", "Prepared browsers must expose a bounded session identifier.");
          }
          if (!await this.#fresh(snapshot)) {
            this.#stage(this.#closed ? "stopped" : "superseded");
          } else {
            this.#stage("activating");
            activationOpen = true;
            try { await this.#operations.promote(candidate, snapshot, () => {
              if (!activationOpen) throw new IterationError("activation_expired", "This activation callback has expired.");
              if (committed) throw new IterationError("activation_committed", "This build has already been activated.");
              if (this.#closed) throw new IterationError("iteration_stopped", "Development iteration is stopped.");
              if (this.#desired?.fingerprint !== snapshot.fingerprint) {
                throw new IterationError("build_superseded", "A newer source fingerprint superseded this build.");
              }
              // The adapter invokes commit at the same synchronous boundary
              // where its bridge makes the new session authoritative.
              committed = true;
              this.#active = { build: attempt.id, session: candidate.session, fingerprint: snapshot.fingerprint };
              this.#stage("ready");
            }); } finally { activationOpen = false; }
            if (!committed) throw new IterationError("activation_not_committed", "Session promotion returned without committing activation.");
          }
        }
      } catch (error) {
        if (committed) {
          // Retirement is adapter-owned. A failure after the atomic switch
          // cannot roll the bridge back or discard its now-active browser.
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
      // A failed unchanged source waits for an edit. Intermediate desired
      // versions were never queued and cannot create redundant builds.
      if (this.#desired?.fingerprint === snapshot.fingerprint) break;
    }
  }
}
