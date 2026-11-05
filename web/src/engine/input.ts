import { CURSOR_MODES, WHEEL_PIXELS } from "../generated/host";
import type { Runtime } from "../generated/runtime";

const MAX_TOUCH_POINTERS = 32;
const MOUSE_BUTTONS = ["MouseLeft", "MouseMiddle", "MouseRight"];
const WHEEL_LINES = 3;

export class BrowserInput {
  game: Pick<Runtime, "key" | "look" | "motion" | "pointer" | "pointer_left" | "wheel" | "pad"> | undefined;
  private held = new Set<string>();
  private pointers = new Map<number, { key: string; target: HTMLElement }>();
  private abort = new AbortController();
  private dragging: number | undefined;
  private x = 0;
  private y = 0;
  private mode: number = CURSOR_MODES.drag;
  private locked = false;
  private contextMenu = true;
  private padAxes = new Float32Array(6);

  constructor(canvas: HTMLCanvasElement) {
    const options = { signal: this.abort.signal };
    canvas.addEventListener("keydown", event => {
      if (this.held.has(event.code)) { event.preventDefault(); return; }
      if (event.repeat) return;
      if (this.game?.key(event.code, true)) { this.held.add(event.code); event.preventDefault(); }
    }, options);
    window.addEventListener("keyup", event => {
      if (this.held.delete(event.code)) this.game?.key(event.code, false);
    }, options);
    canvas.addEventListener("blur", () => this.clear(), options);
    window.addEventListener("blur", () => this.clear(), options);
    canvas.addEventListener("pointerdown", event => {
      if (this.dragging !== undefined || event.button !== 0) return;
      canvas.focus();
      if (this.mode !== CURSOR_MODES.drag) return;
      this.dragging = event.pointerId;
      this.x = event.clientX; this.y = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }, options);
    canvas.addEventListener("pointermove", event => {
      if (this.locked) { this.game?.motion(event.movementX, event.movementY); return; }
      if (event.isPrimary) {
        const ratio = window.devicePixelRatio || 1;
        this.game?.pointer(event.offsetX * ratio, event.offsetY * ratio);
      }
      if (this.dragging !== event.pointerId) return;
      this.game?.look(event.clientX - this.x, event.clientY - this.y);
      this.x = event.clientX; this.y = event.clientY;
    }, options);
    canvas.addEventListener("pointerleave", event => { if (event.isPrimary) this.game?.pointer_left(); }, options);
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
      canvas.addEventListener(name, event => { if (this.dragging === event.pointerId) this.dragging = undefined; }, options);
    }
    canvas.addEventListener("mousedown", event => {
      const code = MOUSE_BUTTONS[event.button];
      if (code === undefined) return;
      const used = this.held.has(code) || this.game?.key(code, true) === true;
      if (code === "MouseRight") this.contextMenu = !used;
      if (!used) return;
      this.held.add(code);
      canvas.focus();
      event.preventDefault();
    }, options);
    window.addEventListener("mouseup", event => {
      const code = MOUSE_BUTTONS[event.button];
      if (code !== undefined && this.held.delete(code)) this.game?.key(code, false);
    }, options);
    canvas.addEventListener("contextmenu", event => { if (!this.contextMenu) event.preventDefault(); }, options);
    canvas.addEventListener("wheel", event => {
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? WHEEL_PIXELS : event.deltaMode === WheelEvent.DOM_DELTA_LINE ? WHEEL_LINES : 1;
      this.game?.wheel(-event.deltaX / unit, -event.deltaY / unit);
    }, { signal: this.abort.signal, passive: true });
    canvas.addEventListener("click", () => {
      if (!this.game || this.mode !== CURSOR_MODES.captured || this.locked) return;
      void Promise.resolve(canvas.requestPointerLock()).catch(() => undefined);
    }, options);
    document.addEventListener("pointerlockchange", () => {
      const locked = document.pointerLockElement === canvas;
      if (locked && this.mode !== CURSOR_MODES.captured) { document.exitPointerLock(); return; }
      if (this.locked && !locked && document.hasFocus()) { this.game?.key("Escape", true); this.game?.key("Escape", false); }
      this.locked = locked;
    }, options);
    for (const button of document.querySelectorAll<HTMLElement>("[data-key]")) {
      const key = button.dataset.key!;
      button.addEventListener("pointerdown", event => {
        if (event.button !== 0 || this.pointers.has(event.pointerId) || this.pointers.size >= MAX_TOUCH_POINTERS) return;
        if (!this.game?.key(key, true)) return;
        event.preventDefault(); button.setPointerCapture(event.pointerId);
        this.pointers.set(event.pointerId, { key, target: button });
      }, options);
      for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
        button.addEventListener(name, event => {
          const pointer = this.pointers.get(event.pointerId);
          if (pointer?.target !== button) return;
          this.pointers.delete(event.pointerId); this.game?.key(pointer.key, false);
        }, options);
      }
    }
  }

  setCursorMode(mode: number) {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode !== CURSOR_MODES.drag) this.dragging = undefined;
    if (mode !== CURSOR_MODES.captured) this.unlock();
  }

  pollPads() {
    const game = this.game;
    if (!game || !document.hasFocus()) return;
    let pads: (Gamepad | null)[];
    try { pads = navigator.getGamepads(); } catch { return; }
    for (const pad of pads) {
      if (pad?.mapping !== "standard") continue;
      let buttons = 0;
      pad.buttons.forEach((button, index) => { if (button.pressed) buttons |= 1 << index; });
      this.padAxes.set([pad.axes[0], -pad.axes[1], pad.axes[2], -pad.axes[3], pad.buttons[6]?.value ?? 0, pad.buttons[7]?.value ?? 0]);
      game.pad(buttons, this.padAxes);
    }
  }

  clear() { this.held.clear(); this.pointers.clear(); this.dragging = undefined; this.unlock(); }
  dispose() { this.abort.abort(); this.clear(); this.game = undefined; }

  private unlock() {
    if (!this.locked) return;
    this.locked = false;
    document.exitPointerLock();
  }
}
