import {
  type CauseRow,
  type Json,
  type Run,
  type RunEvent,
  at,
  causes,
  instrument,
  num,
} from "./record.ts";
import {
  type Cell,
  type Column,
  type Focus,
  type View,
  cell,
  element,
  link,
  node,
  note,
  number,
  off,
  section,
  table,
  text,
} from "./table.ts";

const SHARED = ["kind", "frame", "at_ms", "cause", "entry", "name"];
const LISTED = new Set([
  "cause",
  "span",
  "counter",
  "pipeline",
  "job",
  "write",
  "rebuilt",
  "plan",
  "draw_dropped",
  "drawn",
]);
const STAGES = ["asked", "ready", "used"] as const;
type Stage = (typeof STAGES)[number];

interface PipeRow {
  cause: number;
  first: RunEvent;
  asked?: RunEvent;
  ready?: RunEvent;
  used?: RunEvent;
}

interface Group {
  name: string;
  why: string | null;
  count: number;
  bytes: number;
  sized: boolean;
  first: number;
  last: number;
}

function round(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function word(value: Json | undefined): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function causeOf(event: RunEvent): number {
  return typeof event.cause === "number" ? event.cause : 0;
}

function columns(...titles: [string, boolean?][]): Column[] {
  return titles.map(([title, numeric]) => ({ title, numeric: numeric ?? false }));
}

function line(content: string | Node, ...more: (string | Node)[]): HTMLElement {
  const element = document.createElement("p");
  element.append(content, ...more);
  return element;
}

function frameCell(frame: number | undefined): Cell {
  if (frame === undefined) return cell(undefined);
  return node(link("frame", { what: "frame", key: String(frame) }, String(frame)), frame);
}

function causeCell(cause: number): Cell {
  if (cause === 0) return text("no cause");
  return node(link("launch", { what: "cause", key: String(cause) }, String(cause)), cause);
}

function amount(value: number | undefined, unit: string): Cell {
  return value === undefined ? cell(undefined) : cell({ value: round(value) }, unit);
}

function between(from: RunEvent | undefined, to: RunEvent | undefined): Cell {
  if (from === undefined || to === undefined) return text(to === undefined ? "none" : "not recorded");
  return amount(to.at_ms - from.at_ms, "ms");
}

function extras(event: RunEvent, known: string[]): Cell {
  const skip = new Set([...SHARED, ...known]);
  const parts: string[] = [];
  for (const [key, value] of Object.entries(event)) {
    if (skip.has(key) || value === undefined) continue;
    parts.push(`${key} ${typeof value === "string" ? value : JSON.stringify(value)}`);
  }
  return parts.length === 0 ? cell(null) : text(parts.join(", "));
}

function pipelines(events: RunEvent[]): { rows: PipeRow[]; strays: RunEvent[] } {
  const rows: PipeRow[] = [];
  const strays: RunEvent[] = [];
  const latest = new Map<string, PipeRow>();
  for (const event of events) {
    if (event.kind !== "pipeline") continue;
    const state = word(event.state);
    const stage = STAGES.indexOf(state as Stage);
    if (stage < 0) {
      strays.push(event);
      continue;
    }
    const key = [event.name, event.pass, event.part, event.asked_by].map((v) => JSON.stringify(v ?? null)).join("|");
    let row = latest.get(key);
    if (row === undefined || STAGES.slice(stage).some((later) => row?.[later] !== undefined)) {
      row = { cause: causeOf(event), first: event };
      rows.push(row);
      latest.set(key, row);
    }
    row[STAGES[stage]] = event;
  }
  return { rows, strays };
}

function grouped(events: RunEvent[], kind: string, withWhy: boolean): Group[] {
  const groups = new Map<string, Group>();
  for (const event of events) {
    if (event.kind !== kind) continue;
    const name = word(event.name) ?? JSON.stringify(event.name ?? null);
    const why = withWhy ? (word(event.why) ?? null) : null;
    const key = `${name}\u0000${why}`;
    let group = groups.get(key);
    if (group === undefined) {
      group = { name, why, count: 0, bytes: 0, sized: true, first: event.frame, last: event.frame };
      groups.set(key, group);
    }
    group.count += 1;
    const bytes = num(event.bytes);
    if (bytes === undefined) group.sized = false;
    else group.bytes += bytes;
    group.first = Math.min(group.first, event.frame);
    group.last = Math.max(group.last, event.frame);
  }
  return [...groups.values()];
}

function pipelineTable(rows: PipeRow[]): HTMLElement {
  return table(
    columns(
      ["name"],
      ["pass"],
      ["part"],
      ["brick"],
      ["asked by"],
      ["asked frame", true],
      ["asked at", true],
      ["asked to ready", true],
      ["ready frame", true],
      ["ready to used", true],
      ["used frame", true],
    ),
    rows.map((row) => [
      cell(row.first.name),
      cell(row.first.pass),
      cell(row.first.part),
      cell(row.first.brick),
      cell(row.first.asked_by),
      row.asked ? frameCell(row.asked.frame) : text("none"),
      row.asked ? amount(row.asked.at_ms, "ms") : text("none"),
      between(row.asked, row.ready),
      row.ready ? frameCell(row.ready.frame) : text("none"),
      between(row.ready, row.used),
      row.used ? frameCell(row.used.frame) : text("none"),
    ]),
  );
}

function jobTable(events: RunEvent[]): HTMLElement {
  const known = ["id", "cache", "source", "outcome", "asked_s", "sent_s", "ran_s", "landed_s"];
  return table(
    columns(
      ["name"],
      ["id", true],
      ["cache"],
      ["source"],
      ["outcome"],
      ["asked", true],
      ["sent", true],
      ["ran", true],
      ["landed", true],
      ["frame", true],
      ["at", true],
      ["more"],
    ),
    events.map((event) => [
      cell(event.name),
      cell(event.id),
      cell(event.cache),
      cell(event.source),
      cell(event.outcome),
      amount(num(event.asked_s), "s"),
      amount(num(event.sent_s), "s"),
      amount(num(event.ran_s), "s"),
      amount(num(event.landed_s), "s"),
      frameCell(event.frame),
      amount(event.at_ms, "ms"),
      extras(event, known),
    ]),
  );
}

function writeTable(events: RunEvent[]): HTMLElement {
  const groups = grouped(events, "write", false);
  return table(
    columns(["resource"], ["writes", true], ["bytes", true], ["first frame", true], ["last frame", true]),
    groups.map((group) => [
      text(group.name),
      cell({ value: group.count }),
      group.sized ? cell({ value: group.bytes }, "bytes") : cell(undefined),
      frameCell(group.first),
      frameCell(group.last),
    ]),
  );
}

function rebuiltTable(events: RunEvent[]): HTMLElement {
  const groups = grouped(events, "rebuilt", true);
  return table(
    columns(["name"], ["why"], ["times", true], ["first frame", true], ["last frame", true]),
    groups.map((group) => [
      text(group.name),
      cell(group.why ?? undefined),
      cell({ value: group.count }),
      frameCell(group.first),
      frameCell(group.last),
    ]),
  );
}

function eventTable(events: RunEvent[], known: string[], titles: string[]): HTMLElement {
  return table(
    columns(["frame", true], ["at", true], ["name"], ...titles.map((title): [string] => [title]), ["more"]),
    events.map((event) => [
      frameCell(event.frame),
      amount(event.at_ms, "ms"),
      cell(event.name),
      ...known.map((key) => cell(event[key])),
      extras(event, known),
    ]),
  );
}

function otherTable(events: RunEvent[]): HTMLElement {
  return table(
    columns(["frame", true], ["at", true], ["kind"], ["name"], ["fields"]),
    events.map((event) => [
      frameCell(event.frame),
      amount(event.at_ms, "ms"),
      text(event.kind),
      cell(event.name),
      extras(event, []),
    ]),
  );
}

function totals(events: RunEvent[]): string {
  let writes = 0;
  let bytes = 0;
  for (const event of events) {
    if (event.kind !== "write") continue;
    writes += 1;
    bytes += num(event.bytes) ?? 0;
  }
  return `${number(writes)} writes, ${number(bytes, "bytes")}`;
}

function firstDrawn(events: RunEvent[]): RunEvent | undefined {
  let first: RunEvent | undefined;
  for (const event of events) {
    if (event.kind === "drawn" && (first === undefined || event.frame < first.frame)) first = event;
  }
  return first;
}

function lists(events: RunEvent[], rows: PipeRow[], strays: RunEvent[], every: boolean): HTMLElement[] {
  const of = (kind: string) => events.filter((event) => event.kind === kind);
  const parts: HTMLElement[] = [];
  const add = (title: string, count: number, build: () => (Node | string)[]) => {
    if (every || count > 0) parts.push(element("div", "part", element("h3", "", title), ...build()));
  };
  const drawn = of("drawn");
  const first = firstDrawn(drawn);
  const writes = of("write").length;
  const rebuilds = of("rebuilt").length;
  const lead = (shown: boolean, head: () => HTMLElement): HTMLElement[] => (shown ? [head()] : []);
  add("Pipelines", rows.length, () => [pipelineTable(rows)]);
  add("Jobs", of("job").length, () => [jobTable(of("job"))]);
  add("Writes", writes, () => [...lead(writes > 0, () => line(totals(events))), writeTable(events)]);
  add("Rebuilds", rebuilds, () => [...lead(rebuilds > 0, () => line(`${number(rebuilds)} rebuilds`)), rebuiltTable(events)]);
  add("Plans", of("plan").length, () => [eventTable(of("plan"), ["why"], ["why"])]);
  add("Dropped draws", of("draw_dropped").length, () => [eventTable(of("draw_dropped"), ["why"], ["why"])]);
  add("Drawn", drawn.length, () => [
    ...lead(first !== undefined, () =>
      line("first drawn frame ", link("frame", { what: "frame", key: String(first?.frame) }, String(first?.frame))),
    ),
    eventTable(drawn, ["pipeline"], ["pipeline"]),
  ]);
  const other = [...strays, ...events.filter((event) => !LISTED.has(event.kind))];
  add("Other events", other.length, () => [otherTable(other)]);
  const spans = of("span").length;
  const counters = of("counter").length;
  if (every || spans + counters > 0) {
    parts.push(line(`${number(spans)} span events, ${number(counters)} counter events, listed by frame in `, link("frames", null, "Frames")));
  }
  return parts;
}

function opener(row: CauseRow): string {
  if (row.cause === 0) return "No cause";
  if (row.origin === null) return `Cause ${row.cause}: opened before the record`;
  return `Cause ${row.cause}: ${row.origin} ${row.what ?? ""}`.trimEnd();
}

function causeTable(rows: CauseRow[]): HTMLElement {
  const known = ["what", "ms"];
  return table(
    columns(
      ["cause", true],
      ["origin"],
      ["what"],
      ["frame", true],
      ["at", true],
      ["ms", true],
      ["more"],
      ["pipelines asked", true],
      ["jobs", true],
      ["ran", true],
      ["writes", true],
      ["bytes", true],
      ["first drawn frame", true],
      ["events", true],
    ),
    rows.map((row) => {
      const opened = row.opened;
      const before = row.cause !== 0 && row.origin === null;
      return [
        causeCell(row.cause),
        row.cause === 0 ? cell(null) : before ? text("opened before the record") : cell(row.origin ?? undefined),
        row.cause === 0 || before ? cell(null) : cell(row.what ?? undefined),
        row.frame === null ? cell(null) : frameCell(row.frame),
        opened ? amount(opened.at_ms, "ms") : cell(null),
        opened && num(opened.ms) !== undefined ? amount(num(opened.ms), "ms") : cell(null),
        opened ? extras(opened, known) : cell(null),
        cell({ value: row.pipelines }),
        cell({ value: row.jobs }),
        amount(row.ran_s, "s"),
        cell({ value: row.writes }),
        row.bytes === null ? cell({ unavailable: "writes_with_no_cause_are_not_events" }) : cell({ value: row.bytes }, "bytes"),
        row.firstDrawn === null ? text("none") : frameCell(row.firstDrawn),
        cell({ value: row.events.length }),
      ];
    }),
  );
}

function entryTable(run: Run): HTMLElement {
  return table(
    columns(
      ["entry", true],
      ["frame", true],
      ["mark"],
      ["preparation", true],
      ["events", true],
      ["dropped events", true],
      ["dropped entries", true],
      ["stop"],
    ),
    run.entries.map((entry) => {
      const preparation = at(run, "frame", entry.index, ["preparation_ms"]);
      return [
        cell({ value: entry.index }),
        frameCell(entry.frame),
        entry.mark === null ? cell(null) : text(entry.mark),
        typeof preparation === "number" ? amount(preparation, "ms") : cell(preparation),
        cell({ value: entry.events.length }),
        cell({ value: entry.dropped.events }),
        cell({ value: entry.dropped.entries }),
        text(entry.stop ? "stop" : ""),
      ];
    }),
  );
}

function summary(run: Run, state: ReturnType<typeof instrument>): HTMLElement[] {
  const parts: HTMLElement[] = [line("All events of the record, from launch to its last entry.")];
  if (state === "off") parts.push(off("causes"));
  else if (state !== "on") parts.push(note(`instrument causes unavailable: ${state.unavailable}`));
  if (run.dropped.events !== 0 || run.dropped.entries !== 0) {
    parts.push(line(`dropped by the record: ${number(run.dropped.events)} events, ${number(run.dropped.entries)} entries`));
  }
  const first = firstDrawn(run.events);
  if (first === undefined) {
    parts.push(note("no drawn event in this record"));
  } else {
    const where = first.cause === undefined ? "with no cause" : `under cause ${first.cause}`;
    parts.push(
      line(
        "first drawn frame ",
        link("frame", { what: "frame", key: String(first.frame) }, String(first.frame)),
        `, ${where}: ${word(first.name) ?? ""} with ${word(first.pipeline) ?? "no pipeline"}`,
      ),
    );
  }
  return parts;
}

function chosen(focus: Focus): number | null {
  if (focus === null || focus.what !== "cause") return null;
  const value = Number(focus.key);
  return Number.isInteger(value) ? value : null;
}

export const launch: View = (run, _span, focus) => {
  const root = document.createElement("div");
  if (run.header === null) {
    root.append(section("Launch", note("no header line yet")));
    return root;
  }
  const state = instrument(run, "causes");
  const joined = pipelines(run.events);
  root.append(section("Launch", ...summary(run, state)));
  root.append(section("Entries", entryTable(run)));
  if (state !== "on" || run.events.length === 0) {
    root.append(...lists(run.events, joined.rows, joined.strays, true));
    return root;
  }
  const rows = causes(run).sort((a, b) => a.cause - b.cause);
  root.append(section("Causes", causeTable(rows)));
  const only = chosen(focus);
  if (only !== null) {
    root.append(line(link("launch", null, "all causes")));
    if (!rows.some((row) => row.cause === only)) root.append(note(`cause ${only} is not in this record`));
  }
  for (const row of rows) {
    if (only !== null && row.cause !== only) continue;
    const own = row.events.filter((event) => !(event.kind === "cause" && event.cause === row.cause));
    if (own.length === 0 && only === null) continue;
    const mine = joined.rows.filter((pipe) => pipe.cause === row.cause);
    const strays = joined.strays.filter((event) => causeOf(event) === row.cause);
    const body = lists(own, mine, strays, false);
    root.append(section(opener(row), ...(body.length > 0 ? body : [note("no events under this cause")])));
  }
  return root;
};
