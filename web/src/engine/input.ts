import { CURSOR_MODES, TOUCH_PHASES, WHEEL_PIXELS } from "../generated/host";
import type { Runtime } from "../generated/runtime";

const WHEEL_LINES = 3;

export class BrowserInput {
  game: Pick<Runtime, "key" | "mouse_button" | "motion" | "pointer" | "pointer_left" | "wheel" | "pad" | "touch" | "touch_motion"> | undefined;
  private held = new Set<string>();
  private touches = new Map<number, { x: number; y: number }>();
  private abort = new AbortController();
  private last: { x: number; y: number } | undefined;
  private mode: number = CURSOR_MODES.free;
  private locked = false;
  private padAxes = new Float32Array(6);

  constructor(canvas: HTMLCanvasElement) {
    const options = { signal: this.abort.signal };
    const ratio = () => window.devicePixelRatio || 1;
    canvas.addEventListener("keydown", event => {
      if (this.held.has(event.code) || event.repeat) return;
      this.held.add(event.code);
      this.game?.key(event.code, true);
    }, options);
    window.addEventListener("keyup", event => {
      if (this.held.delete(event.code)) this.game?.key(event.code, false);
    }, options);
    canvas.addEventListener("blur", () => this.clear(), options);
    window.addEventListener("blur", () => this.clear(), options);
    canvas.addEventListener("pointerdown", event => {
      canvas.focus();
      if (event.pointerType === "touch") {
        this.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
        canvas.setPointerCapture(event.pointerId);
        event.preventDefault();
        this.game?.touch(event.pointerId, event.offsetX * ratio(), event.offsetY * ratio(), TOUCH_PHASES.began);
        return;
      }
      if (event.button === 0 && this.mode === CURSOR_MODES.drag) canvas.setPointerCapture(event.pointerId);
    }, options);
    canvas.addEventListener("pointermove", event => {
      if (event.pointerType === "touch") {
        const last = this.touches.get(event.pointerId);
        if (!last) return;
        this.game?.touch(event.pointerId, event.offsetX * ratio(), event.offsetY * ratio(), TOUCH_PHASES.moved);
        this.game?.touch_motion(event.pointerId, event.clientX - last.x, event.clientY - last.y);
        last.x = event.clientX; last.y = event.clientY;
        return;
      }
      if (this.locked) { this.game?.motion(event.movementX, event.movementY); return; }
      if (this.last && this.mode !== CURSOR_MODES.captured) this.game?.motion(event.clientX - this.last.x, event.clientY - this.last.y);
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
      this.game?.mouse_button(event.button, true);
    }, options);
    window.addEventListener("mouseup", event => {
      this.game?.mouse_button(event.button, false);
    }, options);
    canvas.addEventListener("wheel", event => {
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? WHEEL_PIXELS : event.deltaMode === WheelEvent.DOM_DELTA_LINE ? WHEEL_LINES : 1;
      this.game?.wheel(-event.deltaX / unit, -event.deltaY / unit);
    }, { signal: this.abort.signal, passive: true });
    canvas.addEventListener("click", () => {
      if (!this.game || this.mode !== CURSOR_MODES.captured || this.locked) return;
      void Promise.resolve(canvas.requestPointerLock({ unadjustedMovement: true }))
        .catch(() => canvas.requestPointerLock()).catch(() => undefined);
    }, options);
    document.addEventListener("pointerlockchange", () => {
      const locked = document.pointerLockElement === canvas;
      if (locked && this.mode !== CURSOR_MODES.captured) { document.exitPointerLock(); return; }
      if (this.locked && !locked && document.hasFocus()) { this.game?.key("Escape", true); this.game?.key("Escape", false); }
      this.locked = locked;
      this.last = undefined;
    }, options);
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
    for (const id of this.touches.keys()) this.game?.touch(id, 0, 0, TOUCH_PHASES.cancelled);
    this.held.clear(); this.touches.clear();
    this.last = undefined; this.unlock();
  }
  dispose() { this.abort.abort(); this.clear(); this.game = undefined; }

  private unlock() {
    if (!this.locked) return;
    this.locked = false;
    document.exitPointerLock();
  }
}
