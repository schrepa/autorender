import type { Json, Reading, Run, Span } from "./record";
import { at, entryLabel, fact, instrument, num, reading, spanSize, unavailable } from "./record";
import type { Cell, Column, Focus, View } from "./table";
import { cell, element, link, node, note, number, off, section, table, text } from "./table";

export type Obj = { [key: string]: Json };

export function obj(value: Json | Reading | undefined): Obj | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  if (unavailable(value as Json) !== undefined || "missing" in value) return undefined;
  return value as Obj;
}

export function blank(): Cell {
  return cell(null);
}

export function col(title: string, numeric = false): Column {
  return { title, numeric };
}

export function links(items: [string, Focus, string][], sort?: string | number | null): Cell {
  const holder = element("span");
  items.forEach(([view, focus, shown], index) => {
    if (index > 0) holder.append(", ");
    holder.append(link(view, focus, shown));
  });
  return node(holder, sort === undefined ? items.map(item => item[2]).join(", ") : sort);
}

export function spanLine(run: Run, span: Span): HTMLElement {
  const end = run.entries[span.to];
  if (!end) return element("p", "", "No entries yet.");
  const start = span.from === "launch" ? "launch" : run.entries[span.from] ? entryLabel(run.entries[span.from]) : `entry ${span.from}`;
  const size = spanSize(run, span);
  const ticks = size.ticks === null ? "ticks not recorded" : `${size.ticks} ticks`;
  return element("p", "", `Span: ${start} to ${entryLabel(end)}; ${size.frames} frames, ${ticks}.`);
}

export function droppedLine(run: Run): HTMLElement[] {
  if (run.dropped.entries === 0 && run.dropped.events === 0) return [];
  return [element("p", "", `The record dropped ${number(run.dropped.entries)} entries and ${number(run.dropped.events)} events.`)];
}

export function instrumentLine(run: Run, name: string): HTMLElement[] {
  const state = instrument(run, name);
  if (state === "on") return [];
  if (state === "off") return [off(name)];
  return [note(`instrument ${name} unavailable: ${state.unavailable}`)];
}

function grouped(names: string[]): string {
  const counts = new Map<string, number>();
  for (const name of names) {
    const base = name.replace(/#\d+$/, "");
    counts.set(base, (counts.get(base) ?? 0) + 1);
  }
  return [...counts].map(([base, count]) => (count > 1 ? `${base} (${count})` : base)).join(", ");
}

export interface LinkRow { pass: string; part: string; pipeline: string | null; entry: string | null; module: string | null }

export function passLinks(run: Run): LinkRow[] {
  const rows: LinkRow[] = [];
  const passes = obj(obj(run.header?.links)?.passes) ?? {};
  for (const [pass, partsJson] of Object.entries(passes)) {
    for (const [part, pipesJson] of Object.entries(obj(partsJson) ?? {})) {
      const pipes = Object.entries(obj(pipesJson) ?? {});
      if (pipes.length === 0) rows.push({ pass, part, pipeline: null, entry: null, module: null });
      for (const [pipeline, entriesJson] of pipes) {
        const entries = Object.entries(obj(entriesJson) ?? {});
        if (entries.length === 0) rows.push({ pass, part, pipeline, entry: null, module: null });
        for (const [entry, module] of entries) rows.push({ pass, part, pipeline, entry, module: typeof module === "string" ? module : null });
      }
    }
  }
  return rows;
}

function gather<K extends keyof LinkRow>(rows: LinkRow[], key: K, value: K): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  for (const row of rows) {
    const from = row[key];
    const to = row[value];
    if (from === null) continue;
    if (!out.has(from)) out.set(from, new Set());
    if (to !== null) out.get(from)!.add(to);
  }
  return out;
}

export function timeSpanPass(spanName: string, passes: Set<string>): string | null {
  if (!spanName.startsWith("stage_")) return null;
  const pass = spanName.slice("stage_".length);
  return passes.has(pass) ? pass : null;
}

function keysAt(run: Run, name: string, span: Span, path: string[] = []): string[] {
  const keys = new Set<string>(Object.keys(obj(at(run, name, span.to, path)) ?? {}));
  if (span.from !== "launch") for (const key of Object.keys(obj(at(run, name, span.from, path)) ?? {})) keys.add(key);
  return [...keys];
}

function unitOf(run: Run, name: string): string | null {
  return fact(run, name)?.unit ?? null;
}

function readCell(run: Run, name: string, path: string[], span: Span): Cell {
  return cell(reading(run, name, path, span), unitOf(run, name));
}

export function perTimed(run: Run, spanName: string, span: Span): { time: Reading; timed: Reading; per: Cell } {
  const time = reading(run, "gpu.passes.time", [spanName], span);
  const timed = reading(run, "gpu.passes.timed", [spanName], span);
  if (!("value" in time)) return { time, timed, per: cell(time) };
  if (!("value" in timed)) return { time, timed, per: cell(timed) };
  if (timed.value === 0) return { time, timed, per: text("no timed frames") };
  return { time, timed, per: cell({ value: Math.round(time.value / timed.value) }, unitOf(run, "gpu.passes.time")) };
}

function resources(run: Run): [string, string[], string[]][] {
  const out: [string, string[], string[]][] = [];
  for (const [name, value] of Object.entries(obj(obj(run.header?.links)?.resources) ?? {})) {
    const strings = (list: Json | undefined) => (Array.isArray(list) ? list.filter((item): item is string => typeof item === "string") : []);
    out.push([name, strings(obj(value)?.readers), strings(obj(value)?.writers)]);
  }
  return out;
}

function targets(run: Run): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const [name, list] of Object.entries(obj(obj(run.header?.links)?.targets) ?? {})) {
    if (Array.isArray(list)) out.set(name, list.filter((item): item is string => typeof item === "string"));
  }
  return out;
}

interface DroppedRow { key: string; kind: string; pass: string; reason: string; count: Reading }

function droppedRows(run: Run, span: Span): DroppedRow[] {
  return keysAt(run, "gpu.draws.dropped", span).map(key => {
    const parts = key.split("/");
    const reason = parts.length >= 3 ? parts[parts.length - 1] : "";
    const pass = parts.length >= 3 ? parts[parts.length - 2] : parts[1] ?? "";
    const kind = parts.length >= 3 ? parts.slice(0, -2).join("/") : parts[0];
    return { key, kind, pass, reason, count: reading(run, "gpu.draws.dropped", [key], span) };
  });
}

function wholeValue(run: Run, name: string, span: Span): HTMLElement[] {
  const value = at(run, name, span.to);
  if (obj(value)) return [];
  return [element("p", "", `${name}: `, cell(value as Json).node)];
}

function valueOr(run: Run, name: string, span: Span, build: () => HTMLElement): HTMLElement[] {
  const whole = wholeValue(run, name, span);
  return whole.length > 0 ? whole : [build()];
}

export const passes: View = (run, span, focus) => {
  if (!run.header) return section("Passes", note("no header yet"));
  const linkRows = passLinks(run);
  const passNames = new Set<string>(linkRows.map(row => row.pass));
  for (const key of keysAt(run, "gpu.passes.draws", span)) passNames.add(key);
  for (const key of keysAt(run, "gpu.passes.opened", span, ["opened"])) passNames.add(key);
  const parts = gather(linkRows, "pass", "part");
  const pipes = gather(linkRows, "pass", "pipeline");
  const entries = gather(linkRows, "pass", "entry");
  const spanNames = keysAt(run, "gpu.passes.time", span);
  const allResources = resources(run);
  const allTargets = targets(run);
  const dropped = droppedRows(run, span);
  const timeValue = obj(at(run, "gpu.passes.time", span.to));
  const droppedValue = at(run, "gpu.draws.dropped", span.to);
  const rows = [...passNames].map(pass => {
    const spanName = `stage_${pass}`;
    const timedCells = timeValue && !(spanName in timeValue) ? { per: text(`no time span ${spanName}`), timed: blank() } : (() => {
      const found = perTimed(run, spanName, span);
      return { per: found.per, timed: cell(found.timed, unitOf(run, "gpu.passes.timed")) };
    })();
    const own = dropped.filter(row => row.pass === pass);
    const droppedCell = !obj(droppedValue) ? cell(droppedValue) : own.length === 0 ? blank() : own.some(row => !("value" in row.count)) ? cell(own.find(row => !("value" in row.count))!.count) : cell({ value: own.reduce((sum, row) => sum + ("value" in row.count ? row.count.value : 0), 0) });
    const written = allResources.filter(([, , writers]) => writers.includes(pass)).map(([name]) => name);
    const read = allResources.filter(([, readers]) => readers.includes(pass)).map(([name]) => name);
    const targetNames = [...allTargets].filter(([, list]) => list.includes(pass)).map(([name]) => name);
    return [
      links([["passes", { what: "pass", key: pass }, pass]], pass),
      text([...(parts.get(pass) ?? [])].join(", ")),
      cell({ value: pipes.get(pass)?.size ?? 0 }),
      links([...(entries.get(pass) ?? [])].map(entry => ["shaders", { what: "entry", key: entry }, entry] as [string, Focus, string])),
      readCell(run, "gpu.passes.draws", [pass], span),
      readCell(run, "gpu.passes.instances", [pass], span),
      readCell(run, "gpu.passes.indirect_draws", [pass], span),
      readCell(run, "gpu.passes.opened", ["opened", pass], span),
      readCell(run, "gpu.passes.opened", ["merged", pass], span),
      timedCells.per,
      timedCells.timed,
      droppedCell,
      text(targetNames.join(", ")),
      text(grouped(written)),
      text(grouped(read)),
      links([["draws", { what: "pass", key: pass }, "draws"]], null),
    ];
  });
  const timeRows = spanNames.map(name => {
    const joined = timeSpanPass(name, passNames);
    const found = perTimed(run, name, span);
    return [
      text(name),
      joined === null ? blank() : links([["passes", { what: "pass", key: joined }, joined]], joined),
      cell(found.time, unitOf(run, "gpu.passes.time")),
      cell(found.timed, unitOf(run, "gpu.passes.timed")),
      found.per,
    ];
  });
  const focusPass = focus?.what === "pass" ? focus.key : null;
  const shownLinks = linkRows.filter(row => focusPass === null || row.pass === focusPass);
  const shownResources = allResources.filter(([, readers, writers]) => focusPass === null || readers.includes(focusPass) || writers.includes(focusPass));
  return section(
    "Passes",
    spanLine(run, span),
    ...instrumentLine(run, "gpu.passes"),
    table(
      [col("Pass"), col("Parts"), col("Pipelines", true), col("Entries"), col("Draws", true), col("Instances", true), col("Indirect draws", true), col("Opened", true), col("Merged", true), col("Time per timed frame", true), col("Timed frames", true), col("Dropped draws", true), col("Targets"), col("Writes"), col("Reads"), col("Draws view")],
      rows,
    ),
    element("h3", "", "Time spans"),
    note("Time spans nest and overlap, so they are never added up. A span named stage_<pass> is joined to that pass; every span is listed here under its own name. Time is divided by the span's timed frames only."),
    ...wholeValue(run, "gpu.passes.time", span),
    ...(timeValue ? [table([col("Time span"), col("Pass"), col("Time", true), col("Timed frames", true), col("Per timed frame", true)], timeRows)] : []),
    element("h3", "", focusPass === null ? "Parts, pipelines and entries" : `Parts, pipelines and entries of ${focusPass}`),
    table(
      [col("Pass"), col("Part"), col("Pipeline"), col("Entry"), col("Module")],
      shownLinks.map(row => [
        links([["passes", { what: "pass", key: row.pass }, row.pass]], row.pass),
        text(row.part),
        row.pipeline === null ? text("none") : text(row.pipeline),
        row.entry === null ? text("none") : links([["shaders", { what: "entry", key: row.entry }, row.entry]], row.entry),
        row.module === null ? blank() : text(row.module),
      ]),
    ),
    element("h3", "", focusPass === null ? "Resources" : `Resources of ${focusPass}`),
    table(
      [col("Resource"), col("Writers"), col("Readers"), col("Written", true)],
      shownResources.map(([name, readers, writers]) => [
        text(name),
        links(writers.map(pass => ["passes", { what: "pass", key: pass }, pass] as [string, Focus, string])),
        links(readers.map(pass => ["passes", { what: "pass", key: pass }, pass] as [string, Focus, string])),
        readCell(run, "gpu.resources.written", [name], span),
      ]),
    ),
  );
};

const WORK = ["gpu.pipelines.work.runs", "gpu.pipelines.work.operations", "gpu.pipelines.work.samples"];

function fromFrame(run: Run, index: number | "launch", entry: string): Json | Reading {
  if (index === "launch") return "launch";
  return at(run, WORK[0], index, [entry, "from_frame"]);
}

function workCell(run: Run, name: string, path: string[], entry: string, span: Span): Cell {
  if (span.from !== "launch") {
    const first = num(at(run, name, span.from, [entry, "from_frame"]) as Json);
    const last = num(at(run, name, span.to, [entry, "from_frame"]) as Json);
    if (first !== undefined && last !== undefined && first !== last) return text("no difference: counted from another frame");
  }
  return readCell(run, name, [entry, ...path], span);
}

export const shaders: View = (run, span, focus) => {
  if (!run.header) return section("Shaders", note("no header yet"));
  const linkRows = passLinks(run);
  const entryPasses = gather(linkRows, "entry", "pass");
  const entryModules = gather(linkRows, "entry", "module");
  const names = new Set<string>();
  for (const name of WORK) for (const key of keysAt(run, name, span)) names.add(key);
  for (const entry of entryPasses.keys()) names.add(entry);
  const changes = run.header.instruments.find(item => item.name === "gpu.pipelines.work")?.changes;
  const rows = [...names].map(entry => {
    const used = [...(entryPasses.get(entry) ?? [])];
    return [
      links([["shaders", { what: "entry", key: entry }, entry]], entry),
      text([...(entryModules.get(entry) ?? [])].join(", ")),
      links(used.map(pass => ["passes", { what: "pass", key: pass }, pass] as [string, Focus, string])),
      used.length > 1 ? text(`one counter for ${used.length} passes`) : blank(),
      ...WORK.map(name => workCell(run, name, ["total"], entry, span)),
      cell(fromFrame(run, span.from, entry)),
      cell(fromFrame(run, span.to, entry)),
    ];
  });
  const focusEntry = focus?.what === "entry" ? focus.key : null;
  const pathRows: Cell[][] = [];
  for (const entry of names) {
    if (focusEntry !== null && entry !== focusEntry) continue;
    const paths = new Set<string>();
    for (const name of WORK) for (const key of keysAt(run, name, span, [entry, "paths"])) paths.add(key);
    for (const path of paths) pathRows.push([text(entry), text(path), ...WORK.map(name => workCell(run, name, ["paths", path], entry, span))]);
  }
  return section(
    "Shaders",
    spanLine(run, span),
    ...instrumentLine(run, "gpu.pipelines.work"),
    ...(changes ? [element("p", "", `What the instrument changes: ${changes}`)] : []),
    ...WORK.flatMap(name => wholeValue(run, name, span)),
    note("Counts are differences over the span, never divided by frames. An entry used by several passes has one counter for all of them."),
    table([col("Entry"), col("Module"), col("Passes"), col("Shared"), col("Runs", true), col("Operations", true), col("Samples", true), col("From frame at start", true), col("From frame at end", true)], rows),
    element("h3", "", focusEntry === null ? "Paths" : `Paths of ${focusEntry}`),
    table([col("Entry"), col("Path"), col("Runs", true), col("Operations", true), col("Samples", true)], pathRows),
  );
};

const DRAW_NAMES = ["gpu.draws.draws", "gpu.draws.instances", "gpu.draws.indirect_draws"];

function causeCell(run: Run, cause: number | undefined): Cell {
  if (instrument(run, "causes") !== "on") return cell({ missing: "instrument_off" });
  if (cause === undefined) return text("no cause");
  return links([["launch", { what: "cause", key: String(cause) }, String(cause)]], cause);
}

function frameCell(frame: number): Cell {
  return links([["frame", { what: "frame", key: String(frame) }, String(frame)]], frame);
}

export const draws: View = (run, span, focus) => {
  if (!run.header) return section("Draws", note("no header yet"));
  const linkRows = passLinks(run);
  const partPasses = gather(linkRows, "part", "pass");
  const names = [...DRAW_NAMES];
  for (const item of run.header.facts) if (item.name.startsWith("gpu.draws.") && item.name !== "gpu.draws.dropped" && !names.includes(item.name)) names.push(item.name);
  const byPart = names.filter(name => obj(at(run, name, span.to)));
  const whole = names.filter(name => !byPart.includes(name));
  const partNames = new Set<string>();
  for (const name of byPart) for (const key of keysAt(run, name, span)) partNames.add(key);
  for (const part of partPasses.keys()) partNames.add(part);
  const focusPass = focus?.what === "pass" ? focus.key : null;
  const focusPart = focus?.what === "part" ? focus.key : null;
  const partRows = [...partNames].filter(part => focusPart === null || part === focusPart).filter(part => focusPass === null || partPasses.get(part)?.has(focusPass)).map(part => [
    text(part),
    links([...(partPasses.get(part) ?? [])].map(pass => ["passes", { what: "pass", key: pass }, pass] as [string, Focus, string])),
    ...byPart.map(name => readCell(run, name, [part], span)),
  ]);
  const passNames = keysAt(run, "gpu.passes.draws", span).filter(pass => focusPass === null || pass === focusPass);
  const dropped = droppedRows(run, span).filter(row => focusPass === null || row.pass === focusPass);
  const droppedEvents = (run.byKind.get("draw_dropped") ?? []).filter(item => focusPass === null || String(item.name ?? "").split("/")[1] === focusPass);
  const drawn = run.byKind.get("drawn") ?? [];
  const kinds = Object.entries(obj(obj(run.header.links)?.kinds) ?? {});
  return section(
    "Draws",
    spanLine(run, span),
    ...droppedLine(run),
    note("Counts are differences over the span. Event lists cover the whole record."),
    element("h3", "", "By part"),
    table([col("Part"), col("Passes"), ...byPart.map(name => col(name, true))], partRows),
    ...(whole.length > 0 ? [table([col("Name"), col("Over the span", true)], whole.map(name => [text(name), readCell(run, name, [], span)]))] : []),
    element("h3", "", "By pass"),
    ...valueOr(run, "gpu.passes.draws", span, () => table([col("Pass"), col("Draws", true), col("Instances", true), col("Indirect draws", true)], passNames.map(pass => [
      links([["passes", { what: "pass", key: pass }, pass]], pass),
      readCell(run, "gpu.passes.draws", [pass], span),
      readCell(run, "gpu.passes.instances", [pass], span),
      readCell(run, "gpu.passes.indirect_draws", [pass], span),
    ]))),
    element("h3", "", "Dropped draws over the span"),
    ...valueOr(run, "gpu.draws.dropped", span, () => table([col("Kind"), col("Pass"), col("Reason"), col("Count", true)], dropped.map(row => [
      text(row.kind),
      links([["passes", { what: "pass", key: row.pass }, row.pass]], row.pass),
      text(row.reason),
      cell(row.count, unitOf(run, "gpu.draws.dropped")),
    ]))),
    element("h3", "", "Dropped draw events"),
    table([col("Frame", true), col("Draw"), col("Reason"), col("Cause"), col("Entry", true)], droppedEvents.map(item => [
      frameCell(item.frame),
      cell(item.name ?? null),
      cell(item.why ?? null),
      causeCell(run, item.cause),
      cell(item.entry),
    ])),
    element("h3", "", "First drawn"),
    table([col("Frame", true), col("Kind"), col("Pipeline"), col("Cause"), col("At ms", true)], drawn.map(item => [
      frameCell(item.frame),
      cell(item.name ?? null),
      typeof item.pipeline === "string" ? links([["shaders", { what: "entry", key: item.pipeline }, item.pipeline]], item.pipeline) : cell(item.pipeline ?? null),
      causeCell(run, item.cause),
      cell(item.at_ms, "ms"),
    ])),
    element("h3", "", "Kinds"),
    table([col("Kind"), col("Passes")], kinds.map(([kind, list]) => [
      text(kind),
      links((Array.isArray(list) ? list : []).filter((item): item is string => typeof item === "string").map(pass => ["passes", { what: "pass", key: pass }, pass] as [string, Focus, string])),
    ])),
  );
};

export { causeCell, frameCell };
