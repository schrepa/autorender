import type { Entry, Json, Run, RunEvent } from "./record";
import { at, num, reading } from "./record";
import { fileUrl } from "./follow";
import type { Cell, Column, Focus, View } from "./table";
import { cell, element, link, note, node, number, section, table, text } from "./table";
import { blank, causeCell, col, droppedLine, frameCell, links, obj, perTimed, spanLine } from "./gpu";

const KNOWN_FIELDS = new Set(["kind", "frame", "at_ms", "cause", "entry", "name"]);

function ms(value: number): Cell {
  return cell({ value: Math.round(value * 10000) / 10000 }, "ms");
}

function depthOf(item: RunEvent): number | undefined {
  return num(item.depth as Json);
}

function entriesByFrame(run: Run): Map<number, Entry[]> {
  const out = new Map<number, Entry[]>();
  for (const entry of run.entries) {
    if (!out.has(entry.frame)) out.set(entry.frame, []);
    out.get(entry.frame)!.push(entry);
  }
  return out;
}

function frameList(run: Run, byEntry: Map<number, Entry[]>): number[] {
  const all = new Set<number>(run.byFrame.keys());
  for (const frame of byEntry.keys()) all.add(frame);
  return [...all].sort((a, b) => a - b);
}

function pictureCell(run: Run, entries: Entry[]): Cell {
  const holder = element("span");
  let sort: string | null = null;
  for (const entry of entries) {
    if (entry.picture === undefined) continue;
    const picture = obj(entry.picture);
    if (holder.childNodes.length > 0) holder.append(", ");
    if (picture && typeof picture.path === "string") {
      const anchor = element("a", "", picture.path);
      anchor.href = fileUrl(run.name, picture.path);
      anchor.target = "_blank";
      holder.append(anchor);
      if (typeof picture.sha256 === "string") {
        const sha = element("code", "", ` sha256 ${picture.sha256.slice(0, 8)}...`);
        sha.title = picture.sha256;
        holder.append(sha);
      }
      sort = sort ?? picture.path;
    } else {
      holder.append(cell(entry.picture).node);
      sort = sort ?? "";
    }
  }
  return node(holder, sort);
}

function entryCells(run: Run, entries: Entry[]): Cell[] {
  if (entries.length === 0) return [blank(), blank(), blank(), blank(), blank(), blank(), blank(), blank(), blank(), blank()];
  const last = entries[entries.length - 1];
  const marks = entries.map(entry => entry.mark).filter((mark): mark is string => mark !== null);
  const lost = entries.reduce((sum, entry) => sum + entry.dropped.events, 0);
  return [
    text(entries.map(entry => entry.index).join(", ")),
    marks.length > 0 ? text(marks.join(", ")) : blank(),
    pictureCell(run, entries),
    lost > 0 ? node(document.createTextNode(`events dropped (${lost})`), lost) : cell({ value: 0 }),
    cell(at(run, "frame", last.index, ["render_ms"]), "ms"),
    cell(at(run, "frame", last.index, ["simulation_ms"]), "ms"),
    cell(at(run, "frame", last.index, ["resolution", "gpu_ms"]), "ms"),
    cell(at(run, "frame", last.index, ["draw_calls"])),
    cell(at(run, "frame", last.index, ["upload_bytes"]), "bytes"),
    cell(last.tick),
  ];
}

export const frames: View = (run, span) => {
  if (!run.header) return section("Frames", note("no header yet"));
  const byEntry = entriesByFrame(run);
  const list = frameList(run, byEntry);
  const first: string[] = [];
  const second: string[] = [];
  const sums = new Map<number, Map<string, { ms: number; count: number }>>();
  for (const [frame, events] of run.byFrame) {
    const own = new Map<string, { ms: number; count: number }>();
    for (const item of events) {
      if (item.kind !== "span") continue;
      const depth = depthOf(item);
      if (depth !== 0 && depth !== 1) continue;
      const key = `${depth} ${String(item.name)}`;
      const columns = depth === 0 ? first : second;
      if (!columns.includes(key)) columns.push(key);
      const was = own.get(key) ?? { ms: 0, count: 0 };
      own.set(key, { ms: was.ms + (num(item.ms as Json) ?? 0), count: was.count + 1 });
    }
    sums.set(frame, own);
  }
  const spanKeys = [...first, ...second];
  const nameCount = new Map<string, number>();
  for (const key of spanKeys) nameCount.set(key.slice(2), (nameCount.get(key.slice(2)) ?? 0) + 1);
  const spanTitle = (key: string) => (nameCount.get(key.slice(2))! > 1 ? `${key.slice(2)} (depth ${key[0]})` : key.slice(2));
  const lastOf = (frame: number) => byEntry.get(frame)?.at(-1);
  const paired = list.some(frame => lastOf(frame) && lastOf(frame - 1));
  const lastEntry = run.entries.at(-1);
  const stageKeys = paired && lastEntry ? Object.keys(obj(at(run, "stages.time", lastEntry.index)) ?? {}) : [];
  const timeKeys = paired && lastEntry ? Object.keys(obj(at(run, "gpu.passes.time", lastEntry.index)) ?? {}) : [];
  const columns: Column[] = [
    col("Frame", true), col("Events", true), col("Entry"), col("Mark"), col("Picture"), col("Dropped", true),
    col("render_ms", true), col("simulation_ms", true), col("gpu_ms", true), col("draw_calls", true), col("upload_bytes", true), col("Tick", true),
    ...spanKeys.map(key => col(spanTitle(key), true)),
    ...stageKeys.map(key => col(`stages.time ${key}`, true)),
    ...timeKeys.map(key => col(`gpu ${key} per timed frame`, true)),
  ];
  const rows = list.map(frame => {
    const events = run.byFrame.get(frame) ?? [];
    const own = sums.get(frame);
    const current = lastOf(frame);
    const previous = lastOf(frame - 1);
    const pair = current && previous ? { from: previous.index, to: current.index } : null;
    return [
      frameCell(frame),
      events.length > 0 ? cell({ value: events.length }) : text("none"),
      ...entryCells(run, byEntry.get(frame) ?? []),
      ...spanKeys.map(key => {
        const found = own?.get(key);
        if (!found) return blank();
        if (found.count === 1) return ms(found.ms);
        const shown = Math.round(found.ms * 10000) / 10000;
        return node(document.createTextNode(`${shown} ms in ${found.count}`), shown);
      }),
      ...stageKeys.map(key => (pair ? cell(reading(run, "stages.time", [key], pair), "ns") : blank())),
      ...timeKeys.map(key => (pair ? perTimed(run, key, pair).per : blank())),
    ];
  });
  const spanEvents = run.byKind.get("span")?.length ?? 0;
  const lines: (Node | string)[] = [
    element("p", "", `${number(list.length)} frames (${list[0] ?? "none"} to ${list.at(-1) ?? "none"}), ${number(run.entries.length)} entries, ${number(run.events.length)} events, ${number(spanEvents)} of them spans.`),
    spanLine(run, span),
    ...droppedLine(run),
    note("Rows are every frame seen in the events or in an entry. Span columns show time spans at depth 0 and 1 by name; a name seen more than once in a frame shows its total and how many. Entry columns come from the last entry written at that frame."),
  ];
  if (paired) lines.push(note("stages.time and gpu columns are differences from the entry of the frame before; gpu time is divided by its timed frames only."));
  if (!run.header.facts.some(item => item.name === "frame")) lines.push(note("frame: not recorded"));
  return section("Frames", ...lines, table(columns, rows));
};

function details(item: RunEvent): string {
  return Object.entries(item)
    .filter(([key]) => !KNOWN_FIELDS.has(key))
    .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join("; ");
}

function eventLinks(item: RunEvent): Cell {
  const items: [string, Focus, string][] = [];
  const name = typeof item.name === "string" ? item.name : null;
  if (item.kind === "pipeline" && name !== null) items.push(["shaders", { what: "entry", key: name }, "shaders"]);
  if (item.kind === "pipeline" && typeof item.pass === "string") items.push(["passes", { what: "pass", key: item.pass }, "passes"]);
  if (item.kind === "drawn" && typeof item.pipeline === "string") items.push(["shaders", { what: "entry", key: item.pipeline }, "shaders"]);
  if ((item.kind === "drawn" || item.kind === "draw_dropped") && name !== null) {
    const pass = name.split("/")[1];
    items.push(["draws", pass ? { what: "pass", key: pass } : null, "draws"]);
  }
  if (item.kind === "cause" && typeof item.cause === "number") items.push(["launch", { what: "cause", key: String(item.cause) }, "launch"]);
  return items.length === 0 ? blank() : links(items, null);
}

export const frame: View = (run, span, focus) => {
  if (!run.header) return section("Frame", note("no header yet"));
  const byEntry = entriesByFrame(run);
  const list = frameList(run, byEntry);
  const asked = focus?.what === "frame" ? Number(focus.key) : NaN;
  const chosen = Number.isFinite(asked) ? asked : run.entries[span.to]?.frame ?? list[0];
  if (chosen === undefined) return section("Frame", note("none in this record"));
  const events = run.byFrame.get(chosen) ?? [];
  const place = list.indexOf(chosen);
  const nav = element("p", "");
  if (place > 0) nav.append(link("frame", { what: "frame", key: String(list[place - 1]) }, `frame ${list[place - 1]}`), " ");
  nav.append(link("frames", null, "all frames"));
  if (place >= 0 && place < list.length - 1) nav.append(" ", link("frame", { what: "frame", key: String(list[place + 1]) }, `frame ${list[place + 1]}`));
  const carriers = [...new Set(events.map(item => item.entry))];
  const carried = carriers.map(index => {
    const entry = run.entries[index];
    return entry ? `entry ${index} (events dropped in it: ${entry.dropped.events})` : `entry ${index}`;
  });
  const spans = events
    .map((item, order) => ({ item, order }))
    .filter(({ item }) => item.kind === "span")
    .sort((a, b) => a.item.at_ms - b.item.at_ms || (depthOf(a.item) ?? 0) - (depthOf(b.item) ?? 0) || a.order - b.order);
  const counters = events.map((item, order) => ({ item, order })).filter(({ item }) => item.kind === "counter");
  const others = events.map((item, order) => ({ item, order })).filter(({ item }) => item.kind !== "span" && item.kind !== "counter");
  const entries = byEntry.get(chosen) ?? [];
  return section(
    `Frame ${chosen}`,
    nav,
    element("p", "", events.length === 0 ? "No events of this frame are in the record." : `${events.length} events, carried by ${carried.join(", ")}.`),
    ...droppedLine(run),
    element("h3", "", "Entries written at this frame"),
    table(
      [col("Entry"), col("Mark"), col("Picture"), col("Dropped", true), col("render_ms", true), col("simulation_ms", true), col("gpu_ms", true), col("draw_calls", true), col("upload_bytes", true), col("Tick", true)],
      entries.map(entry => entryCells(run, [entry])),
    ),
    element("h3", "", "Time spans"),
    note("Ordered by start, indented by depth. Spans nest, so their times are never added up."),
    table(
      [col("Start ms", true), col("Span"), col("Depth", true), col("Category"), col("Time", true), col("Cause"), col("Order", true)],
      spans.map(({ item, order }) => {
        const depth = depthOf(item) ?? 0;
        const label = element("span", "", String(item.name ?? ""));
        label.style.paddingLeft = `${depth * 1.2}em`;
        return [
          ms(item.at_ms),
          node(label, String(item.name ?? "")),
          cell(depth),
          cell(item.category ?? null),
          typeof item.ms === "number" ? ms(item.ms) : cell(item.ms ?? undefined),
          causeCell(run, item.cause),
          cell(order),
        ];
      }),
    ),
    element("h3", "", "Counters"),
    table(
      [col("Name"), col("Value", true), col("At ms", true), col("Cause"), col("Order", true)],
      counters.map(({ item, order }) => [cell(item.name ?? null), cell(item.value ?? undefined), ms(item.at_ms), causeCell(run, item.cause), cell(order)]),
    ),
    element("h3", "", "Other events"),
    table(
      [col("Kind"), col("Name"), col("At ms", true), col("Cause"), col("Details"), col("Views"), col("Order", true)],
      others.map(({ item, order }) => [text(item.kind), cell(item.name ?? null), ms(item.at_ms), causeCell(run, item.cause), text(details(item)), eventLinks(item), cell(order)]),
    ),
  );
};
