import "./style.css";
import { about, names, pictures, runs, setListing } from "./about.ts";
import { Follower, listRuns, POLL_MS } from "./follow.ts";
import { frame, frames } from "./frames.ts";
import { draws, passes, shaders } from "./gpu.ts";
import { launch } from "./launch.ts";
import { defaultSpan, entryLabel, spanSize, type Run, type Span } from "./record.ts";
import { element, href, note, number, type Place, readPlace, renderStart, type View } from "./table.ts";

const views: [string, string, View][] = [
  ["runs", "Runs", runs],
  ["launch", "Launch", launch],
  ["frames", "Frames", frames],
  ["frame", "Frame", frame],
  ["passes", "Passes", passes],
  ["shaders", "Shaders", shaders],
  ["draws", "Draws", draws],
  ["pictures", "Pictures", pictures],
  ["about", "About", about],
  ["names", "Names", names],
];

const runName = element("strong", "run-name");
const status = element("span", "status");
const tabs = element("nav", "tabs");
const fromPick = element("select");
const toPick = element("select");
const size = element("span", "size");
const reset = element("a", "", "default span");
const picker = element("div", "span", element("label", "", "from ", fromPick), element("label", "", "to ", toPick), size, reset);
const content = element("main", "content");
document.body.append(element("header", "bar", element("div", "top", element("h1", "", "Runs"), runName, status), tabs, picker), content);

let follower: Follower | null = null;
let pending = false;
let shown = "";
let options: { run: Run | null; count: number } = { run: null, count: -1 };
let listing = "";
let listBusy = false;

function schedule(): void {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    draw();
  });
}

function follow(name: string | null): void {
  if ((follower?.name ?? null) === name) return;
  follower?.stop();
  follower = name === null ? null : new Follower(name, () => schedule());
  follower?.start();
  options = { run: null, count: -1 };
}

function spanOf(run: Run, place: Place): Span {
  const fallback = defaultSpan(run);
  const last = run.entries.length - 1;
  let to = place.to ?? fallback.to;
  if (to > last) to = last;
  let from = place.from ?? fallback.from;
  if (typeof from === "number" && from > last) from = fallback.from;
  if (typeof from === "number" && from > to) [from, to] = [to, from];
  return { from, to };
}

function listed(place: Place): boolean {
  return place.view === "runs" || !place.run;
}

function list(): void {
  if (listBusy || !listed(readPlace())) return;
  listBusy = true;
  void listRuns().then(found => {
    const now = JSON.stringify(found);
    if (now !== listing) {
      listing = now;
      setListing(found, null, schedule);
      schedule();
    }
  }, failure => {
    setListing(null, String(failure), schedule);
    schedule();
  }).finally(() => setTimeout(() => {
    listBusy = false;
    list();
  }, POLL_MS));
}

function statusText(): string {
  if (!follower) return "";
  const held = `${number(follower.run.held)} bytes`;
  switch (follower.status) {
    case "loading": return follower.error ? `waiting: ${follower.error}` : "reading";
    case "following": return `following, ${held}`;
    case "stopped": return `stopped, ${held}`;
    case "no_stop_line": return `no stop line, ${held}`;
    case "error": return `error: ${follower.error ?? "unknown"}`;
  }
}

function fill(select: HTMLSelectElement, run: Run, launch: boolean): void {
  const wanted = run.entries.length + (launch ? 1 : 0);
  if (options.run !== run || select.options.length > wanted) select.replaceChildren();
  if (launch && select.options.length === 0) select.append(new Option("launch", "launch"));
  for (let index = select.options.length - (launch ? 1 : 0); index < run.entries.length; index += 1) {
    select.append(new Option(entryLabel(run.entries[index]), String(index)));
  }
}

function drawBar(place: Place, run: Run | null, span: Span | null): void {
  runName.textContent = place.run ?? "no run open";
  document.title = place.run ? `${place.run} runs` : "Runs";
  status.textContent = statusText();
  tabs.replaceChildren(...views.map(([key, title]) => {
    const tab = element("a", key === place.view ? "tab active" : "tab", title);
    tab.href = href({ ...place, view: key, focus: null });
    return tab;
  }));
  picker.hidden = !run || !run.header || run.entries.length === 0;
  if (!run || !span || picker.hidden) return;
  if (options.run !== run || options.count !== run.entries.length) {
    fill(fromPick, run, true);
    fill(toPick, run, false);
    options = { run, count: run.entries.length };
  }
  fromPick.value = String(span.from);
  toPick.value = String(span.to);
  const counted = spanSize(run, span);
  size.textContent = `${number(counted.frames)} frames, ${counted.ticks === null ? "ticks not recorded" : `${number(counted.ticks)} ticks`}`;
  reset.href = href({ ...place, from: null, to: null });
}

function pick(): void {
  const place = readPlace();
  const from = fromPick.value === "launch" ? "launch" : Number(fromPick.value);
  const to = Number(toPick.value);
  location.hash = href({ ...place, from: typeof from === "number" && from > to ? to : from, to: typeof from === "number" && from > to ? from : to });
}

function body(place: Place, run: Run | null, span: Span | null): HTMLElement {
  if (place.view === "runs") return runs(run, span, place.focus);
  if (!place.run || !follower || !run) return element("div", "view", note("open a run from the list"), runs(null, null, null));
  if (!run.header) return element("div", "view", note(follower.error ? `${follower.status}: ${follower.error}` : "reading the record"));
  const chosen = views.find(([key]) => key === place.view);
  if (!chosen || !span) return element("div", "view", note(`no view named ${place.view}`));
  try {
    return chosen[2](run, span, place.focus);
  } catch (failure) {
    return element("div", "view", note(`this view failed: ${String(failure)}`));
  }
}

function draw(): void {
  const place = readPlace();
  follow(place.run);
  const run = follower?.run ?? null;
  const span = run && run.header ? spanOf(run, place) : null;
  drawBar(place, run, span);
  list();
  const where = `${place.run}|${place.view}|${place.from}|${place.to}|${place.focus?.what}:${place.focus?.key}`;
  const y = where === shown ? window.scrollY : 0;
  const active = document.activeElement;
  const typing = active instanceof HTMLInputElement && active.classList.contains("filter") ? active.selectionStart : null;
  renderStart();
  content.replaceChildren(body(place, run, span));
  shown = where;
  window.scrollTo(0, y);
  if (typing !== null) {
    const again = content.querySelector<HTMLInputElement>("input.filter");
    again?.focus();
    again?.setSelectionRange(typing, typing);
  }
}

fromPick.addEventListener("change", pick);
toPick.addEventListener("change", pick);
window.addEventListener("hashchange", schedule);
draw();
