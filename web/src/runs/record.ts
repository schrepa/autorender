export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface Fact {
  name: string;
  kind: string;
  unit: string | null;
}

export interface Instrument {
  name: string;
  changes?: string;
  unavailable?: string;
}

export interface Header {
  name: string;
  host: string;
  machine: Json;
  build: Json;
  shell?: Json;
  scene: Json;
  pack: Json;
  picture: Json;
  every: string;
  queue: Json;
  events: Json;
  instruments: Instrument[];
  facts: Fact[];
  links: Json;
  changes: Json;
  raw: { [key: string]: Json };
}

export interface Dropped {
  entries: number;
  events: number;
}

export interface RunEvent {
  kind: string;
  frame: number;
  at_ms: number;
  cause?: number;
  entry: number;
  [field: string]: Json | undefined;
}

export interface Entry {
  index: number;
  frame: number;
  tick: number | null;
  t: number;
  mark: string | null;
  values: { [name: string]: Json };
  events: RunEvent[];
  dropped: Dropped;
  picture?: Json;
  stop?: boolean;
}

export interface RunFile {
  name: string;
  bytes: number;
  modified_ms: number;
}

export interface Run {
  name: string;
  header: Header | null;
  entries: Entry[];
  events: RunEvent[];
  byKind: Map<string, RunEvent[]>;
  byCause: Map<number, RunEvent[]>;
  byFrame: Map<number, RunEvent[]>;
  dropped: Dropped;
  held: number;
  stopped: boolean;
  open: boolean;
  files: RunFile[];
}

export type Reading =
  | { value: number }
  | { unavailable: string }
  | { missing: "not_recorded" | "instrument_off" | "not_at_start" };

export interface Span {
  from: number | "launch";
  to: number;
}

export interface CauseRow {
  cause: number;
  origin: string | null;
  what: string | null;
  frame: number | null;
  at_ms: number | null;
  opened: RunEvent | null;
  events: RunEvent[];
  jobs: number;
  ran_s: number;
  writes: number;
  bytes: number | null;
  pipelines: number;
  firstDrawn: number | null;
}

type JsonObject = { [key: string]: Json };

const INSTRUMENT_NAMES: [string, string][] = [
  ["gpu.pipelines.work", "gpu.pipelines.work"],
  ["gpu.passes.time", "gpu.passes"],
  ["gpu.passes.timed", "gpu.passes"],
  ["gpu.resources.unchanged", "unchanged_writes"],
];

function isObject(value: Json | undefined): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function num(value: Json | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function str(value: Json | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

export function unavailable(value: Json | undefined): string | undefined {
  if (!isObject(value)) return undefined;
  const keys = Object.keys(value);
  return keys.length === 1 ? str(value.unavailable) : undefined;
}

export function parse(text: string): { lines: Json[]; rest: string } {
  const pieces = text.split("\n");
  const rest = pieces.pop() ?? "";
  const lines: Json[] = [];
  for (const piece of pieces) {
    const line = piece.endsWith("\r") ? piece.slice(0, -1) : piece;
    if (line.trim() === "") continue;
    try {
      lines.push(JSON.parse(line) as Json);
    } catch {
      continue;
    }
  }
  return { lines, rest };
}

function isHeader(line: Json): line is JsonObject {
  return isObject(line) && !("values" in line) && ("facts" in line || "host" in line);
}

function header(line: JsonObject): Header {
  const instruments: Instrument[] = [];
  const listed = line.instruments;
  if (Array.isArray(listed)) {
    for (const item of listed) {
      if (!isObject(item)) continue;
      const instrument: Instrument = { name: str(item.name) ?? "" };
      const changes = str(item.changes);
      if (changes !== undefined) instrument.changes = changes;
      const reason = str(item.unavailable);
      if (reason !== undefined) instrument.unavailable = reason;
      instruments.push(instrument);
    }
  }
  const facts: Fact[] = [];
  const declared = line.facts;
  if (Array.isArray(declared)) {
    for (const item of declared) {
      if (!isObject(item)) continue;
      const name = str(item.name);
      if (name === undefined) continue;
      facts.push({ name, kind: str(item.kind) ?? "", unit: str(item.unit) ?? null });
    }
  }
  const made: Header = {
    name: str(line.name) ?? "",
    host: str(line.host) ?? "",
    machine: line.machine ?? null,
    build: line.build ?? null,
    scene: line.scene ?? null,
    pack: line.pack ?? null,
    picture: line.picture ?? null,
    every: str(line.every) ?? "",
    queue: line.queue ?? null,
    events: line.events ?? null,
    instruments,
    facts,
    links: line.links ?? null,
    changes: line.changes ?? null,
    raw: line,
  };
  if ("shell" in line) made.shell = line.shell;
  return made;
}

function dropped(value: Json | undefined): Dropped {
  if (!isObject(value)) return { entries: 0, events: 0 };
  return { entries: num(value.entries) ?? 0, events: num(value.events) ?? 0 };
}

function event(raw: Json, entry: number): RunEvent | undefined {
  if (!isObject(raw)) return undefined;
  const made: RunEvent = {
    ...raw,
    kind: str(raw.kind) ?? "",
    frame: num(raw.frame) ?? 0,
    at_ms: num(raw.at_ms) ?? 0,
    entry,
  };
  const cause = num(raw.cause);
  if (cause === undefined) delete made.cause;
  else made.cause = cause;
  return made;
}

function push<K>(map: Map<K, RunEvent[]>, key: K, item: RunEvent): void {
  const list = map.get(key);
  if (list) list.push(item);
  else map.set(key, [item]);
}

function clear(run: Run): void {
  run.header = null;
  run.entries = [];
  run.events = [];
  run.byKind = new Map();
  run.byCause = new Map();
  run.byFrame = new Map();
  run.dropped = { entries: 0, events: 0 };
  run.stopped = false;
}

function add(run: Run, line: JsonObject): void {
  const index = run.entries.length;
  const events: RunEvent[] = [];
  const listed = line.events;
  if (Array.isArray(listed)) {
    for (const raw of listed) {
      const made = event(raw, index);
      if (!made) continue;
      events.push(made);
      run.events.push(made);
      push(run.byKind, made.kind, made);
      push(run.byCause, made.cause ?? 0, made);
      push(run.byFrame, made.frame, made);
    }
  }
  const lost = dropped(line.dropped);
  const entry: Entry = {
    index,
    frame: num(line.frame) ?? 0,
    tick: num(line.tick) ?? null,
    t: num(line.t) ?? 0,
    mark: str(line.mark) ?? null,
    values: isObject(line.values) ? line.values : {},
    events,
    dropped: lost,
  };
  if ("picture" in line) entry.picture = line.picture;
  if (line.stop === true) {
    entry.stop = true;
    run.stopped = true;
  }
  run.dropped.entries += lost.entries;
  run.dropped.events += lost.events;
  run.entries.push(entry);
}

export function build(name: string, lines: Json[]): Run {
  const run: Run = {
    name,
    header: null,
    entries: [],
    events: [],
    byKind: new Map(),
    byCause: new Map(),
    byFrame: new Map(),
    dropped: { entries: 0, events: 0 },
    held: 0,
    stopped: false,
    open: false,
    files: [],
  };
  return extend(run, lines);
}

export function extend(run: Run, lines: Json[]): Run {
  for (const line of lines) {
    if (isHeader(line)) {
      clear(run);
      run.header = header(line);
    } else if (isObject(line)) {
      add(run, line);
    }
  }
  return run;
}

export function entryLabel(entry: Entry): string {
  const mark = entry.mark === null ? "" : ` mark ${entry.mark}`;
  return `${entry.index} frame ${entry.frame}${mark}`;
}

export function defaultSpan(run: Run): Span {
  const marked = run.entries.filter((entry) => entry.mark !== null);
  if (marked.length >= 2) return { from: marked[0].index, to: marked[marked.length - 1].index };
  if (run.entries.length >= 2) return { from: 0, to: run.entries.length - 1 };
  return { from: "launch", to: run.entries.length - 1 };
}

export function fact(run: Run, name: string): Fact | undefined {
  const facts = run.header?.facts ?? [];
  let path = name;
  for (;;) {
    const found = facts.find((item) => item.name === path);
    if (found) return found;
    const cut = path.lastIndexOf(".");
    if (cut < 0) return undefined;
    path = path.slice(0, cut);
  }
}

export function instrument(run: Run, name: string): "on" | "off" | { unavailable: string } {
  const found = run.header?.instruments.find((item) => item.name === name);
  if (!found) return "off";
  if (found.unavailable !== undefined) return { unavailable: found.unavailable };
  return "on";
}

function owner(name: string): string | undefined {
  for (const [prefix, instrumentName] of INSTRUMENT_NAMES) {
    if (name === prefix || name.startsWith(`${prefix}.`)) return instrumentName;
  }
  return undefined;
}

function absent(run: Run, name: string): Reading {
  const instrumentName = owner(name);
  if (instrumentName !== undefined && instrument(run, instrumentName) !== "on") {
    return { missing: "instrument_off" };
  }
  return { missing: "not_recorded" };
}

function joined(name: string, path: string[]): string {
  return path.length === 0 ? name : `${name}.${path.join(".")}`;
}

function dig(value: Json | undefined, path: string[]): Json | undefined | { unavailable: string } {
  let current = value;
  for (const key of path) {
    const reason = unavailable(current);
    if (reason !== undefined) return { unavailable: reason };
    if (isObject(current)) current = current[key];
    else if (Array.isArray(current)) {
      const at = Number(key);
      current = Number.isInteger(at) ? current[at] : undefined;
    } else return undefined;
  }
  return current;
}

export function at(run: Run, name: string, index: number, path: string[] = []): Json | Reading {
  if (!fact(run, joined(name, path))) return absent(run, name);
  const entry = run.entries[index];
  if (!entry) return { missing: "not_recorded" };
  const found = dig(entry.values[name], path);
  if (found === undefined) return { missing: "not_recorded" };
  return found;
}

function leafAt(run: Run, name: string, path: string[], index: number): Reading {
  const entry = run.entries[index];
  if (!entry || !(name in entry.values)) return { missing: "not_recorded" };
  const found = dig(entry.values[name], path);
  const reason = unavailable(found as Json | undefined);
  if (reason !== undefined) return { unavailable: reason };
  const value = num(found as Json | undefined);
  if (value === undefined) return { missing: "not_recorded" };
  return { value };
}

export function reading(run: Run, name: string, path: string[], span: Span): Reading {
  const declared = fact(run, joined(name, path));
  if (!declared) return absent(run, name);
  const end = leafAt(run, name, path, span.to);
  if (!["count", "bytes", "time"].includes(declared.kind) || span.from === "launch") return end;
  if ("missing" in end) return end;
  const start = leafAt(run, name, path, span.from);
  if ("unavailable" in end) return end;
  if ("unavailable" in start) return start;
  if ("missing" in start) return { missing: "not_at_start" };
  return { value: end.value - start.value };
}

function gather(prefix: string, value: Json, out: [string, number | { unavailable: string }][]): void {
  const key = (part: string): string => (prefix === "" ? part : `${prefix}.${part}`);
  if (typeof value === "number") {
    if (Number.isFinite(value)) out.push([prefix, value]);
    return;
  }
  if (isObject(value)) {
    const reason = unavailable(value);
    if (reason !== undefined) {
      out.push([prefix, { unavailable: reason }]);
      return;
    }
    for (const field of Object.keys(value).sort()) gather(key(field), value[field], out);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, at) => {
      let part = String(at);
      const name = isObject(item) ? str(item.name) : undefined;
      if (name !== undefined) part = name;
      else if (isObject(item)) {
        const names = Object.keys(item)
          .sort()
          .map((field) => item[field])
          .filter((field): field is string => typeof field === "string");
        if (names.length > 0) part = names.join("/");
      }
      gather(key(part), item, out);
    });
  }
}

export function leaves(value: Json): Map<string, number | { unavailable: string }> {
  const out: [string, number | { unavailable: string }][] = [];
  gather("", value, out);
  out.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return new Map(out);
}

function entriesOf(run: Run, span: Span): Entry[] {
  const first = span.from === "launch" ? 0 : span.from + 1;
  return run.entries.slice(Math.max(0, first), Math.max(0, span.to + 1));
}

export function spanEvents(run: Run, span: Span): RunEvent[] {
  return entriesOf(run, span).flatMap((entry) => entry.events);
}

export function spanSize(run: Run, span: Span): { frames: number; ticks: number | null } {
  const end = run.entries[span.to];
  if (!end) return { frames: 0, ticks: null };
  if (span.from === "launch") return { frames: end.frame, ticks: end.tick };
  const start = run.entries[span.from];
  if (!start) return { frames: end.frame, ticks: end.tick };
  const ticks = end.tick !== null && start.tick !== null ? end.tick - start.tick : null;
  return { frames: end.frame - start.frame, ticks };
}

export function causes(run: Run, span?: Span): CauseRow[] {
  const rows = new Map<number, CauseRow>();
  const row = (cause: number): CauseRow => {
    let found = rows.get(cause);
    if (!found) {
      found = {
        cause,
        origin: null,
        what: null,
        frame: null,
        at_ms: null,
        opened: null,
        events: [],
        jobs: 0,
        ran_s: 0,
        writes: 0,
        bytes: cause === 0 ? null : 0,
        pipelines: 0,
        firstDrawn: null,
      };
      rows.set(cause, found);
    }
    return found;
  };
  row(0);
  for (const item of span ? spanEvents(run, span) : run.events) {
    const total = row(item.cause ?? 0);
    total.events.push(item);
    switch (item.kind) {
      case "cause":
        total.origin = str(item.name) ?? "";
        total.what = str(item.what) ?? "";
        total.frame = item.frame;
        total.at_ms = item.at_ms;
        total.opened = item;
        break;
      case "job":
        total.jobs += 1;
        total.ran_s += num(item.ran_s) ?? 0;
        break;
      case "write":
        total.writes += 1;
        if (total.bytes !== null) total.bytes += num(item.bytes) ?? 0;
        break;
      case "pipeline":
        if (item.state === "asked") total.pipelines += 1;
        break;
      case "drawn":
        total.firstDrawn = total.firstDrawn === null ? item.frame : Math.min(total.firstDrawn, item.frame);
        break;
    }
  }
  return [...rows.values()].sort((a, b) => a.cause - b.cause);
}
