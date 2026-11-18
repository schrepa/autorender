import type { Json, Reading, Run, Span } from "./record";

export type View = (run: Run, span: Span, focus: Focus) => HTMLElement;
export type Focus = { what: string; key: string } | null;
export interface Column { title: string; numeric?: boolean }
export interface Cell { node: Node; sort: string | number | null }
export interface Place { run: string | null; view: string; from: number | "launch" | null; to: number | null; focus: Focus }

const sorted = new Map<string, { column: number; down: boolean }>();
const scrolled = new Map<string, { left: number; top: number }>();
const seen = new Map<string, number>();

export function renderStart(): void {
  seen.clear();
}

export function readPlace(hash: string = location.hash): Place {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const whole = (value: string | null) => value !== null && /^[0-9]+$/.test(value) ? Number(value) : null;
  const from = params.get("from");
  const focus = params.get("focus");
  const split = focus === null ? -1 : focus.indexOf(":");
  return {
    run: params.get("run") || null,
    view: params.get("view") || "runs",
    from: from === "launch" ? "launch" : whole(from),
    to: whole(params.get("to")),
    focus: focus !== null && split > 0 ? { what: focus.slice(0, split), key: focus.slice(split + 1) } : null,
  };
}

export function href(place: Place): string {
  const params = new URLSearchParams();
  if (place.run !== null) params.set("run", place.run);
  params.set("view", place.view);
  if (place.from !== null) params.set("from", String(place.from));
  if (place.to !== null) params.set("to", String(place.to));
  if (place.focus) params.set("focus", `${place.focus.what}:${place.focus.key}`);
  return `#${params.toString()}`;
}

export function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const made = document.createElement(tag);
  if (className) made.className = className;
  made.append(...children);
  return made;
}

export function link(view: string, focus: Focus, text: string, span?: Span): HTMLElement {
  const here = readPlace();
  const anchor = element("a", "", text);
  anchor.href = href({ run: here.run, view, from: span ? span.from : here.from, to: span ? span.to : here.to, focus });
  return anchor;
}

export function section(title: string, ...children: (Node | string)[]): HTMLElement {
  return element("section", "part", element("h2", "", title), ...children.map(child => typeof child === "string" ? element("p", "", child) : child));
}

export function off(name: string): HTMLElement {
  return element("p", "empty", `instrument ${name} off`);
}

export function note(text: string): HTMLElement {
  return element("p", "empty", text);
}

export function number(value: number, unit?: string | null): string {
  const shown = Number.isInteger(value) ? value.toLocaleString("en-US") : value.toLocaleString("en-US", { maximumSignificantDigits: 6 });
  return unit ? `${shown} ${unit}` : shown;
}

function empty(text: string): Cell {
  return { node: element("span", "empty", text), sort: null };
}

const missingText: Record<string, string> = { not_recorded: "not recorded", instrument_off: "instrument off", not_at_start: "not recorded at the start" };

export function cell(reading: Reading | Json | undefined, unit?: string | null): Cell {
  if (reading === undefined) return empty("not recorded");
  if (reading === null) return { node: document.createTextNode(""), sort: null };
  if (typeof reading === "number") return { node: document.createTextNode(number(reading, unit)), sort: reading };
  if (typeof reading === "string") return { node: document.createTextNode(reading), sort: reading };
  if (typeof reading === "boolean") return { node: document.createTextNode(String(reading)), sort: String(reading) };
  if (!Array.isArray(reading)) {
    const keys = Object.keys(reading);
    const only = keys.length === 1 ? (reading as { [key: string]: Json })[keys[0]] : undefined;
    if (keys[0] === "value" && typeof only === "number") return { node: document.createTextNode(number(only, unit)), sort: only };
    if (keys[0] === "unavailable" && typeof only === "string") return empty(`unavailable: ${only}`);
    if (keys[0] === "missing" && typeof only === "string") return empty(missingText[only] ?? only);
  }
  const shown = JSON.stringify(reading);
  return { node: element("code", "json", shown), sort: shown };
}

export function text(value: string | number): Cell {
  return cell(value);
}

export function node(shown: Node, sort: string | number | null = null): Cell {
  return { node: shown, sort };
}

function order(a: string | number | null, b: string | number | null): number {
  if (a === null || b === null) return a === b ? 0 : a === null ? 1 : -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "number") return -1;
  if (typeof b === "number") return 1;
  return a.localeCompare(b, "en", { numeric: true });
}

export function table(columns: Column[], rows: Cell[][]): HTMLElement {
  if (rows.length === 0) return note("none in this record");
  const titles = columns.map(column => column.title).join("\u0001");
  const occurrence = (seen.get(titles) ?? 0) + 1;
  seen.set(titles, occurrence);
  const key = `${titles}#${occurrence}`;
  const head = element("tr");
  const body = element("tbody");
  const lines = rows.map(row => {
    const line = element("tr");
    columns.forEach((column, at) => {
      const shown = element("td", column.numeric ? "number" : "");
      const value = row[at];
      if (value) shown.append(value.node);
      line.append(shown);
    });
    return { line, row };
  });
  const headers = columns.map((column, at) => {
    const header = element("th", column.numeric ? "number" : "", column.title);
    header.title = "sort";
    header.addEventListener("click", () => {
      const now = sorted.get(key);
      if (!now || now.column !== at) sorted.set(key, { column: at, down: false });
      else if (!now.down) sorted.set(key, { column: at, down: true });
      else sorted.delete(key);
      arrange();
    });
    head.append(header);
    return header;
  });
  const arrange = () => {
    const now = sorted.get(key);
    headers.forEach((header, at) => {
      header.dataset.sort = now && now.column === at ? now.down ? "down" : "up" : "";
    });
    const placed = now ? lines.map((line, at) => ({ line, at })).sort((a, b) => {
      const first = a.line.row[now.column]?.sort ?? null;
      const second = b.line.row[now.column]?.sort ?? null;
      const by = first === null || second === null ? order(first, second) : now.down ? order(second, first) : order(first, second);
      return by || a.at - b.at;
    }).map(({ line }) => line) : lines;
    body.replaceChildren(...placed.map(({ line }) => line));
  };
  arrange();
  const box = element("div", "box", element("table", "", element("thead", "", head), body));
  const was = scrolled.get(key);
  if (was) requestAnimationFrame(() => { box.scrollLeft = was.left; box.scrollTop = was.top; });
  box.addEventListener("scroll", () => scrolled.set(key, { left: box.scrollLeft, top: box.scrollTop }), { passive: true });
  return box;
}
