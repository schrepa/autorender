import { build, extend, parse, type Run, type RunFile } from "./record.ts";

export interface RunListing {
  name: string;
  open: boolean;
  files: RunFile[];
}

export const POLL_MS = 500;

const ROUTE = "/_engine/runs";
const RECORD = "record.jsonl";

export function fileUrl(run: string, file: string): string {
  return `${ROUTE}/${encodeURIComponent(run)}/${encodeURIComponent(file)}`;
}

export async function listRuns(): Promise<RunListing[]> {
  const response = await fetch(ROUTE, { cache: "no-store" });
  if (!response.ok) throw new Error(`${ROUTE}: ${response.status}`);
  const listed = (await response.json()) as unknown;
  if (!Array.isArray(listed)) return [];
  return listed.map((item: Partial<RunListing>) => ({
    name: String(item.name ?? ""),
    open: item.open === true,
    files: Array.isArray(item.files)
      ? item.files.map((file: Partial<RunFile>) => ({
          name: String(file.name ?? ""),
          bytes: Number(file.bytes ?? 0),
          modified_ms: Number(file.modified_ms ?? 0),
        }))
      : [],
  }));
}

export async function fetchText(run: string, file: string): Promise<string | null> {
  const response = await fetch(fileUrl(run, file), { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${file}: ${response.status}`);
  return response.text();
}

export class Follower {
  run: Run;
  status: "loading" | "following" | "stopped" | "no_stop_line" | "error" = "loading";
  error: string | null = null;
  private decoder = new TextDecoder();
  private rest = "";
  private timer: ReturnType<typeof setTimeout> | undefined;
  private halted = true;
  private fetched = false;

  constructor(
    readonly name: string,
    private changed: (run: Run) => void,
  ) {
    this.run = build(name, []);
  }

  start(): void {
    if (!this.halted) return;
    this.halted = false;
    void this.list().then(() => this.poll());
  }

  stop(): void {
    this.halted = true;
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = undefined;
  }

  private async list(): Promise<boolean> {
    try {
      const found = (await listRuns()).find((item) => item.name === this.name);
      const open = found?.open ?? false;
      const files = found?.files ?? [];
      const moved =
        open !== this.run.open ||
        files.length !== this.run.files.length ||
        files.some((file, at) => {
          const held = this.run.files[at];
          return held.name !== file.name || held.bytes !== file.bytes || held.modified_ms !== file.modified_ms;
        });
      this.run.open = open;
      this.run.files = files;
      return moved;
    } catch {
      return false;
    }
  }

  private reset(): void {
    this.decoder = new TextDecoder();
    this.rest = "";
    const { open, files } = this.run;
    this.run = build(this.name, []);
    this.run.open = open;
    this.run.files = files;
    this.fetched = false;
  }

  private settle(status: Follower["status"], error: string | null, moved: boolean): void {
    const differs = moved || status !== this.status || error !== this.error;
    this.status = status;
    this.error = error;
    if (differs && !this.halted) this.changed(this.run);
  }

  private schedule(delay: number): void {
    if (this.halted) return;
    this.timer = setTimeout(() => void this.poll(), delay);
  }

  private async poll(): Promise<void> {
    this.timer = undefined;
    if (this.halted) return;
    let response: Response;
    try {
      response = await fetch(`${fileUrl(this.name, RECORD)}?from=${this.run.held}`, { cache: "no-store" });
    } catch (failure) {
      this.settle("error", String(failure), false);
      this.schedule(POLL_MS);
      return;
    }
    if (this.halted) return;
    if (!response.ok) {
      const moved = await this.list();
      this.settle(response.status === 404 ? "loading" : "error", `${RECORD}: ${response.status}`, moved);
      this.schedule(POLL_MS);
      return;
    }
    const size = Number(response.headers.get("X-Size"));
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (this.halted) return;
    if (Number.isFinite(size) && size < this.run.held) {
      this.reset();
      this.settle("loading", null, true);
      this.schedule(0);
      return;
    }
    const first = !this.fetched;
    this.fetched = true;
    let moved = false;
    if (bytes.length > 0) {
      const { lines, rest } = parse(this.rest + this.decoder.decode(bytes, { stream: true }));
      this.rest = rest;
      this.run.held += bytes.length;
      const held = this.run.entries.length;
      if (lines.length > 0) extend(this.run, lines);
      if (this.run.entries.slice(held).some((entry) => entry.picture !== undefined)) await this.list();
      moved = true;
    }
    if (this.halted) return;
    if (this.run.stopped) {
      await this.list();
      this.settle("stopped", null, true);
      this.stop();
      return;
    }
    if (bytes.length === 0) {
      moved = await this.list();
      this.settle(this.run.open ? "following" : "no_stop_line", null, moved);
    } else {
      this.settle(first && !this.run.open ? "no_stop_line" : "following", null, moved);
    }
    this.schedule(POLL_MS);
  }
}
