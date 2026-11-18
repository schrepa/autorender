import { entryLabel, fact, type Json, type Run, type Span } from "./record.ts";
import { fetchText, fileUrl, type RunListing } from "./follow.ts";
import { cell, element, type Focus, href, link, node, note, number, readPlace, section, table, text, type Cell } from "./table.ts";

type Scalar = string | number | boolean | null | { unavailable: string };

let listing: RunListing[] | null = null;
let listingError: string | null = null;
let redraw: () => void = () => {};
const headers = new Map<string, { bytes: number; header: { [key: string]: Json } | null | "loading" | "failed" }>();
const waiting: string[] = [];
let reading = 0;
const scripts = new Map<string, Promise<string | null>>();
const images = new Map<string, HTMLElement>();
const filters = { names: "" };

export function setListing(list: RunListing[] | null, error: string | null, changed: () => void): void {
  listing = list;
  listingError = error;
  redraw = changed;
  for (const run of list ?? []) {
    const record = run.files.find(file => file.name === "record.jsonl");
    const held = headers.get(run.name);
    if (!record || (held && held.bytes <= record.bytes && held.header !== "failed")) continue;
    headers.set(run.name, { bytes: record.bytes, header: "loading" });
    waiting.push(run.name);
  }
  pump();
}

function pump(): void {
  while (reading < 4 && waiting.length > 0) {
    const name = waiting.shift() as string;
    reading += 1;
    void firstLine(name).then(header => {
      const held = headers.get(name);
      if (held) held.header = header;
    }, () => {
      const held = headers.get(name);
      if (held) held.header = "failed";
    }).finally(() => {
      reading -= 1;
      pump();
      redraw();
    });
  }
}

async function firstLine(name: string): Promise<{ [key: string]: Json } | null> {
  const response = await fetch(fileUrl(name, "record.jsonl"), { cache: "no-store" });
  if (!response.ok || !response.body) return null;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let held = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (value) held += decoder.decode(value, { stream: true });
    const end = held.indexOf("\n");
    if (end >= 0 || done) {
      void reader.cancel();
      const line = end >= 0 ? held.slice(0, end) : held;
      const parsed = JSON.parse(line) as Json;
      return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) ? parsed : null;
    }
  }
}

function joinedText(value: Json | undefined): string {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(joinedText).join(", ");
  return JSON.stringify(value);
}

function buildText(build: Json | undefined): Cell {
  if (typeof build !== "object" || build === null || Array.isArray(build) || "unavailable" in build) return cell(build);
  const features = Array.isArray(build.features) ? build.features.map(joinedText).join(" ") : "";
  return text(`${joinedText(build.version)}${features ? ` (${features})` : ""}`);
}

export function runs(run: Run | null, _span: Span | null, _focus: Focus): HTMLElement {
  const page = element("div", "view");
  if (listing === null) {
    page.append(note(listingError ? `run list failed: ${listingError}` : "reading the run list"));
    return page;
  }
  if (listing.length === 0) {
    page.append(note("no runs under artifacts/runs"));
    return page;
  }
  const columns = ["run", "session", "host", "scene", "every", "build", "record", "modified", "pictures", "files", "entries", "marks"]
    .map(title => ({ title, numeric: ["record", "pictures", "files", "entries", "marks"].includes(title) }));
  const rows = listing.map(listed => {
    const record = listed.files.find(file => file.name === "record.jsonl");
    const held = headers.get(listed.name)?.header;
    const header = typeof held === "object" ? held : null;
    const anchor = element("a", "", listed.name);
    anchor.href = href({ run: listed.name, view: "launch", from: null, to: null, focus: null });
    const from = (key: string): Cell => {
      if (!record) return node(element("span", "empty", "no record"));
      if (held === "loading" || held === undefined) return node(element("span", "empty", "reading"));
      if (!header) return node(element("span", "empty", "no header line"));
      return key in header ? cell(header[key]) : node(element("span", "empty", "not in record"));
    };
    const loaded = run && run.name === listed.name ? run : null;
    const modified = record ?? listed.files.reduce<RunListing["files"][number] | undefined>((last, file) => !last || file.modified_ms > last.modified_ms ? file : last, undefined);
    return [
      node(anchor, listed.name),
      text(listed.open ? "open" : "closed"),
      from("host"),
      from("scene"),
      from("every"),
      header ? buildText(header.build) : from("build"),
      record ? cell(record.bytes, "bytes") : node(element("span", "empty", "no record")),
      modified ? node(document.createTextNode(new Date(modified.modified_ms).toLocaleString()), modified.modified_ms) : cell(undefined),
      text(listed.files.filter(file => file.name.endsWith(".png")).length),
      text(listed.files.length),
      loaded ? text(loaded.entries.length) : node(element("span", "empty", "open the run to count")),
      loaded ? text(loaded.entries.filter(entry => entry.mark !== null).length) : node(element("span", "empty", "open the run to count")),
    ];
  });
  page.append(section("Runs", table(columns, rows)));
  return page;
}

function picture(run: Run, path: string): HTMLElement {
  const file = run.files.find(found => found.name === path);
  const key = `${run.name}/${path}/${file?.modified_ms ?? 0}`;
  const held = images.get(key);
  if (held) return held;
  const url = fileUrl(run.name, path);
  const image = element("img", "thumb");
  image.loading = "lazy";
  image.alt = path;
  image.src = url;
  const anchor = element("a", "", image);
  anchor.href = url;
  anchor.target = "_blank";
  images.set(key, anchor);
  return anchor;
}

function field(value: Json | undefined, key: string): Json | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value[key] : undefined;
}

export function pictures(run: Run, _span: Span, focus: Focus): HTMLElement {
  const page = element("div", "view");
  const only = focus && focus.what === "frame" ? Number(focus.key) : null;
  if (only !== null) page.append(element("p", "", `frame ${only} only. `, link("pictures", null, "all pictures")));
  const listed = run.entries.filter(entry => entry.picture !== undefined && (only === null || entry.frame === only));
  const named = new Set<string>();
  const rows = listed.map(entry => {
    const shot = entry.picture as Json;
    const path = field(shot, "path");
    const width = field(shot, "width");
    const height = field(shot, "height");
    if (typeof path === "string") named.add(path);
    const frameOf = field(shot, "frame");
    const frame = typeof frameOf === "number" ? frameOf : entry.frame;
    return [
      node(link("names", null, entryLabel(entry), { from: "launch", to: entry.index }), entry.index),
      cell(entry.mark),
      node(link("frame", { what: "frame", key: String(frame) }, String(frame)), frame),
      typeof width === "number" && typeof height === "number" ? text(`${width} x ${height}`) : cell(field(shot, "unavailable") === undefined ? undefined : shot),
      cell(field(shot, "sha256")),
      typeof path === "string" ? node(picture(run, path), path) : cell(shot),
    ];
  });
  page.append(section("Pictures in the record", table(["entry", "mark", "frame", "size", "sha256", "picture"].map(title => ({ title, numeric: title === "frame" })), rows)));
  for (const entry of run.entries) {
    const path = field(entry.picture as Json, "path");
    if (typeof path === "string") named.add(path);
  }
  if (only === null) {
    const others = run.files.filter(file => file.name.endsWith(".png") && !named.has(file.name));
    const extra = others.map(file => [text(file.name), cell(file.bytes, "bytes"), node(picture(run, file.name), file.name)]);
    page.append(section("Pictures not in the record", extra.length === 0 ? note("none in this folder") : table([{ title: "file" }, { title: "size", numeric: true }, { title: "picture" }], extra)));
  }
  return page;
}

function flat(prefix: string, value: Json | undefined, out: Map<string, Scalar>, skip: Set<string> = new Set()): void {
  const key = (part: string) => prefix === "" ? part : `${prefix}.${part}`;
  if (value === undefined) return;
  if (value === null || typeof value !== "object") { out.set(prefix, value); return; }
  if (Array.isArray(value)) {
    value.forEach((item, at) => {
      const object = typeof item === "object" && item !== null && !Array.isArray(item) ? item : null;
      const name = object && typeof object.name === "string" ? object.name : null;
      const strings = object ? Object.keys(object).sort().filter(field => typeof object[field] === "string") : [];
      const used = name !== null ? new Set(["name"]) : new Set(strings);
      const part = name ?? (strings.length > 0 ? strings.map(field => object?.[field]).join("/") : String(at));
      flat(key(part), item, out, used);
    });
    return;
  }
  const keys = Object.keys(value);
  if (keys.length === 1 && typeof value.unavailable === "string") { out.set(prefix, { unavailable: value.unavailable }); return; }
  for (const field of keys.sort()) if (!skip.has(field)) flat(key(field), value[field], out);
}

function scalarCell(value: Scalar | undefined, unit: string | null): Cell {
  if (value === undefined) return node(element("span", "empty", "not in this entry"));
  return cell(value, unit);
}

export function names(run: Run, span: Span, focus: Focus): HTMLElement {
  const page = element("div", "view");
  const end = run.entries[span.to];
  const start = span.from === "launch" ? null : run.entries[span.from];
  if (!end) {
    page.append(note("no entry in this record yet"));
    return page;
  }
  if (focus && focus.what === "name") filters.names = focus.key;
  const order: string[] = [];
  const known = new Set<string>();
  for (const entry of run.entries) for (const name of Object.keys(entry.values)) if (!known.has(name)) { known.add(name); order.push(name); }
  for (const declared of run.header?.facts ?? []) if (!known.has(declared.name)) { known.add(declared.name); order.push(declared.name); }
  const rows: Cell[][] = [];
  for (const name of order) {
    const atEnd = new Map<string, Scalar>();
    const atStart = new Map<string, Scalar>();
    flat("", end.values[name], atEnd);
    if (start) flat("", start.values[name], atStart);
    if (!(name in end.values) && !(start && name in start.values)) {
      const declared = fact(run, name);
      rows.push([text(name), cell(declared?.kind), cell(declared?.unit ?? null), start ? cell(undefined) : text("launch"), cell(undefined), cell(undefined)]);
      continue;
    }
    const paths = [...atEnd.keys(), ...[...atStart.keys()].filter(path => !atEnd.has(path))];
    for (const path of paths) {
      const full = path === "" ? name : `${name}.${path}`;
      const declared = fact(run, full);
      const kind = declared?.kind;
      const unit = declared?.unit ?? null;
      const before = atStart.get(path);
      const after = atEnd.get(path);
      let change: Cell = text("");
      if (kind === "count" || kind === "bytes" || kind === "time") {
        if (typeof after !== "number") change = scalarCell(after, unit);
        else if (!start) change = cell(after, unit);
        else if (before === undefined || !(name in start.values)) change = cell({ missing: "not_at_start" });
        else if (typeof before !== "number") change = scalarCell(before, unit);
        else change = cell(after - before, unit);
      }
      rows.push([text(full), cell(kind), cell(unit), start ? scalarCell(before, unit) : text("launch"), scalarCell(after, unit), change]);
    }
  }
  const filter = element("input", "filter");
  filter.type = "search";
  filter.placeholder = "filter names";
  filter.value = filters.names;
  const from = start ? entryLabel(start) : "launch";
  const box = table(["name", "kind", "unit", `at ${from}`, `at ${entryLabel(end)}`, "difference"].map(title => ({ title, numeric: !["name", "kind", "unit"].includes(title) })), rows);
  const apply = () => {
    filters.names = filter.value;
    const wanted = filter.value.trim().toLowerCase();
    for (const line of box.querySelectorAll<HTMLTableRowElement>("tbody tr")) {
      line.hidden = wanted !== "" && !(line.cells[0]?.textContent ?? "").toLowerCase().includes(wanted);
    }
  };
  filter.addEventListener("input", apply);
  apply();
  page.append(section("Names", element("p", "", "Count, bytes and time names show the difference over the span; state names show the value at its end."), filter, box));
  return page;
}

function rowsOf(value: Json | undefined, prefix: string): Cell[][] {
  const out = new Map<string, Scalar>();
  flat(prefix, value, out);
  return [...out].map(([key, shown]) => [text(key), cell(shown)]);
}

const headerKeys = ["name", "host", "machine", "build", "shell", "scene", "pack", "picture", "every", "queue", "events"];
const knownInstruments = ["causes", "gpu.passes", "gpu.pipelines.work", "deep"];
const elsewhere = new Set([...headerKeys, "instruments", "facts", "links", "changes"]);

function objectOf(value: Json | undefined): { [key: string]: Json } | null {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : null;
}

function script(run: Run, name: string): HTMLElement {
  const file = run.files.find(found => found.name === name);
  if (!file) return note(`no ${name} in this folder`);
  const key = `${run.name}/${name}/${file.bytes}/${file.modified_ms}`;
  let loading = scripts.get(key);
  if (!loading) {
    loading = fetchText(run.name, name);
    scripts.set(key, loading);
  }
  const holder = element("div", "", note("reading"));
  void loading.then(body => {
    if (body === null) { holder.replaceChildren(note(`no ${name} in this folder`)); return; }
    const lines = body.split("\n").filter(line => line.trim() !== "");
    const rows = lines.map((line, at) => {
      let parsed: Json = line;
      try { parsed = JSON.parse(line) as Json; } catch { parsed = line; }
      const object = objectOf(parsed);
      return [text(at + 1), cell(object?.verb), cell(object?.name), cell(object ? object.args : parsed)];
    });
    holder.replaceChildren(table([{ title: "line", numeric: true }, { title: "verb" }, { title: "name" }, { title: "args" }], rows));
  }, failure => holder.replaceChildren(note(`could not read ${name}: ${String(failure)}`)));
  return holder;
}

export function about(run: Run, _span: Span, _focus: Focus): HTMLElement {
  const page = element("div", "view");
  const header = run.header;
  if (!header) {
    page.append(note("no header line yet"));
    return page;
  }
  const raw = header.raw;
  const facts: Cell[][] = [];
  for (const key of headerKeys) {
    if (!(key in raw)) { facts.push([text(key), node(element("span", "empty", "not in record"))]); continue; }
    const value = raw[key];
    const object = objectOf(value);
    if (key === "build" && object && !("unavailable" in object)) { facts.push([text(key), buildText(value)]); continue; }
    if (key === "picture" && Array.isArray(value) && value.length === 2 && value.every(side => typeof side === "number")) { facts.push([text(key), text(`${value[0]} x ${value[1]}`)]); continue; }
    if ((object && !("unavailable" in object && Object.keys(object).length === 1)) || Array.isArray(value)) facts.push(...rowsOf(value, key));
    else facts.push([text(key), cell(value)]);
  }
  for (const key of Object.keys(raw)) if (!elsewhere.has(key)) facts.push(...rowsOf(raw[key], key));
  page.append(section("Record", table([{ title: "name" }, { title: "value" }], facts)));
  const instruments = header.instruments.map(found => [text(found.name), found.unavailable !== undefined ? cell({ unavailable: found.unavailable }) : text("on"), cell(found.changes)]);
  for (const name of knownInstruments) if (!header.instruments.some(found => found.name === name)) instruments.push([text(name), text("off"), cell(null)]);
  page.append(section("Instruments", table([{ title: "instrument" }, { title: "state" }, { title: "what it changes" }], instruments)));
  const changes = Array.isArray(header.changes) ? header.changes.map(item => {
    const object = objectOf(item);
    return [cell(object?.name), cell(object?.changes), cell(object?.value)];
  }) : [];
  page.append(section("Names that change the frame", Array.isArray(header.changes) ? table([{ title: "name" }, { title: "what it changes" }, { title: "value at start" }], changes) : cell(header.changes).node));
  page.append(section("Facts", table([{ title: "name" }, { title: "kind" }, { title: "unit" }], header.facts.map(found => [text(found.name), text(found.kind), cell(found.unit)]))));
  const links = objectOf(header.links);
  const kinds = objectOf(links?.kinds);
  const targets = objectOf(links?.targets);
  const resources = objectOf(links?.resources);
  const passes = objectOf(links?.passes);
  page.append(section("Links",
    element("h3", "", "Passes"),
    passes ? table([{ title: "pass" }, { title: "parts", numeric: true }, { title: "pipelines", numeric: true }], Object.entries(passes).map(([pass, parts]) => {
      const held = objectOf(parts) ?? {};
      const pipelines = Object.values(held).reduce<number>((sum, part) => sum + Object.keys(objectOf(part) ?? {}).length, 0);
      return [node(link("passes", { what: "pass", key: pass }, pass), pass), text(Object.keys(held).length), text(pipelines)];
    })) : note("not in record"),
    element("h3", "", "Kinds"),
    kinds ? table([{ title: "kind" }, { title: "passes" }], Object.entries(kinds).map(([kind, list]) => [text(kind), text(joinedText(list))])) : note("not in record"),
    element("h3", "", "Targets"),
    targets ? table([{ title: "target" }, { title: "written by" }], Object.entries(targets).map(([target, list]) => [text(target), text(joinedText(list))])) : note("not in record"),
    element("h3", "", "Resources"),
    resources ? table([{ title: "resource" }, { title: "written by" }, { title: "read by" }], Object.entries(resources).map(([resource, held]) => {
      const object = objectOf(held);
      return [text(resource), text(joinedText(object?.writers)), text(joinedText(object?.readers))];
    })) : note("not in record"),
  ));
  const dropped = run.entries.filter(entry => entry.dropped.entries !== 0 || entry.dropped.events !== 0);
  page.append(section("Dropped",
    element("p", "", `entries ${number(run.dropped.entries)}, events ${number(run.dropped.events)} over the whole record`),
    table(["entry", "frame", "entries", "events"].map(title => ({ title, numeric: title !== "entry" })), dropped.map(entry => [
      node(document.createTextNode(entryLabel(entry)), entry.index), text(entry.frame), text(entry.dropped.entries), text(entry.dropped.events)]))));
  const here = readPlace();
  page.append(section("Entries", table(["entry", "frame", "tick", "t", "mark", "events", "names", "picture", "stop"].map(title => ({ title, numeric: ["entry", "frame", "tick", "t", "events", "names"].includes(title) })),
    run.entries.map(entry => [
      node(link("names", null, String(entry.index), { from: typeof here.from === "number" && here.from < entry.index ? here.from : "launch", to: entry.index }), entry.index),
      node(link("frame", { what: "frame", key: String(entry.frame) }, String(entry.frame)), entry.frame),
      cell(entry.tick), cell(entry.t),cell(entry.mark), text(entry.events.length), text(Object.keys(entry.values).length),
      entry.picture === undefined ? text("") : cell(field(entry.picture, "path") ?? entry.picture), text(entry.stop ? "stop" : ""),
    ]))));
  page.append(section("Script", script(run, "script.jsonl")));
  page.append(section("Files", table([{ title: "file" }, { title: "size", numeric: true }, { title: "modified" }], run.files.map(file => {
    const anchor = element("a", "", file.name);
    anchor.href = fileUrl(run.name, file.name);
    anchor.target = "_blank";
    return [node(anchor, file.name), cell(file.bytes, "bytes"), node(document.createTextNode(new Date(file.modified_ms).toLocaleString()), file.modified_ms)];
  }))));
  return page;
}
