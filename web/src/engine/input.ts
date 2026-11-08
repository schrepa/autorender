import { CURSOR_MODES, TOUCH_PHASES, WHEEL_PIXELS } from "../generated/host";
import type { Runtime } from "../generated/runtime";

const MAX_TOUCH_POINTERS = 32;
const MOUSE_RIGHT = 2;
const WHEEL_LINES = 3;

export class BrowserInput {
  game: Pick<Runtime, "key" | "mouse_button" | "motion" | "pointer" | "pointer_left" | "wheel" | "pad" | "touch"> | undefined;
  private held = new Set<string>();
  private used = new Set<string>();
  private buttons = new Set<number>();
  private touches = new Set<number>();
  private pointers = new Map<number, { key: string; target: HTMLElement }>();
  private abort = new AbortController();
  private last: { x: number; y: number } | undefined;
  private mode: number = CURSOR_MODES.free;
  private locked = false;
  private contextMenu = true;
  private padAxes = new Float32Array(6);

  constructor(canvas: HTMLCanvasElement) {
    const options = { signal: this.abort.signal };
    const ratio = () => window.devicePixelRatio || 1;
    canvas.addEventListener("keydown", event => {
      if (this.held.has(event.code)) { if (this.used.has(event.code)) event.preventDefault(); return; }
      if (event.repeat) return;
      this.held.add(event.code);
      if (this.game?.key(event.code, true)) { this.used.add(event.code); event.preventDefault(); } else this.used.delete(event.code);
    }, options);
    window.addEventListener("keyup", event => {
      if (this.held.delete(event.code)) this.game?.key(event.code, false);
    }, options);
    canvas.addEventListener("blur", () => this.clear(), options);
    window.addEventListener("blur", () => this.clear(), options);
    canvas.addEventListener("pointerdown", event => {
      canvas.focus();
      if (event.pointerType === "touch") {
        if (this.touches.size >= MAX_TOUCH_POINTERS) return;
        this.touches.add(event.pointerId);
        canvas.setPointerCapture(event.pointerId);
        event.preventDefault();
        this.game?.touch(event.pointerId, event.offsetX * ratio(), event.offsetY * ratio(), TOUCH_PHASES.began);
        return;
      }
      if (event.button === 0 && this.mode === CURSOR_MODES.drag) canvas.setPointerCapture(event.pointerId);
    }, options);
    canvas.addEventListener("pointermove", event => {
      if (event.pointerType === "touch") {
        if (this.touches.has(event.pointerId)) this.game?.touch(event.pointerId, event.offsetX * ratio(), event.offsetY * ratio(), TOUCH_PHASES.moved);
        return;
      }
      if (this.locked) { this.game?.motion(event.movementX, event.movementY); return; }
      if (this.last) this.game?.motion(event.clientX - this.last.x, event.clientY - this.last.y);
      this.last = { x: event.clientX, y: event.clientY };
      if (event.isPrimary) this.game?.pointer(event.offsetX * ratio(), event.offsetY * ratio());
    }, options);
    canvas.addEventListener("pointerleave", event => {
      if (event.pointerType === "touch" || !event.isPrimary) return;
      this.last = undefined;
      this.game?.pointer_left();
    }, options);
    for (const [name, phase] of [["pointerup", TOUCH_PHASES.ended], ["pointercancel", TOUCH_PHASES.cancelled]] as const) {
      canvas.addEventListener(name, event => {
        if (!this.touches.delete(event.pointerId)) return;
        this.game?.touch(event.pointerId, event.offsetX * ratio(), event.offsetY * ratio(), phase);
      }, options);
    }
    canvas.addEventListener("mousedown", event => {
      const used = this.buttons.has(event.button) || this.game?.mouse_button(event.button, true) === true;
      if (event.button === MOUSE_RIGHT) this.contextMenu = !used;
      if (!used) return;
      this.buttons.add(event.button);
      canvas.focus();
      event.preventDefault();
    }, options);
    window.addEventListener("mouseup", event => {
      if (this.buttons.delete(event.button)) this.game?.mouse_button(event.button, false);
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
      this.last = undefined;
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
      game.pad(pad.index, buttons, this.padAxes);
    }
  }

  clear() {
    for (const code of this.held) this.game?.key(code, false);
    for (const button of this.buttons) this.game?.mouse_button(button, false);
    for (const id of this.touches) this.game?.touch(id, 0, 0, TOUCH_PHASES.cancelled);
    for (const { key } of this.pointers.values()) this.game?.key(key, false);
    this.held.clear(); this.used.clear(); this.buttons.clear(); this.touches.clear(); this.pointers.clear();
    this.last = undefined; this.unlock();
  }
  dispose() { this.abort.abort(); this.clear(); this.game = undefined; }

  private unlock() {
    if (!this.locked) return;
    this.locked = false;
    document.exitPointerLock();
  }
}
