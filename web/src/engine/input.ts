import { CURSOR_MODES, TOUCH_PHASES, WHEEL_PIXELS } from "../generated/host";
import type { Runtime } from "../generated/runtime";

const WHEEL_LINES = 3;

type InputGame = Pick<
  Runtime,
  "key" | "mouse_button" | "motion" | "pointer" | "pointer_left" | "wheel" | "pad" | "touch" | "touch_motion" | "set_input_capabilities"
>;

export class BrowserInput {
  private _game: InputGame | undefined;
  private held = new Set<string>();
  private mouseHeld = new Set<number>();
  private touches = new Map<number, { x: number; y: number }>();
  private touchEventsActive = "ontouchstart" in window;
  private abort = new AbortController();
  private last: { x: number; y: number } | undefined;
  private mode: number = CURSOR_MODES.free;
  private locked = false;
  private padAxes = new Float32Array(6);
  private readonly coarseQuery: MediaQueryList | undefined;
  private readonly hoverQuery: MediaQueryList | undefined;
  private readonly canLockPointer: boolean;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly ratio: () => number) {
    const options = { signal: this.abort.signal };
    const activeTouchOptions = { signal: this.abort.signal, passive: false };
    const passiveTouchOptions = { signal: this.abort.signal, passive: true };
    this.coarseQuery = typeof window.matchMedia === "function" ? window.matchMedia("(pointer: coarse)") : undefined;
    this.hoverQuery = typeof window.matchMedia === "function" ? window.matchMedia("(hover: hover)") : undefined;
    this.canLockPointer = typeof canvas.requestPointerLock === "function";
    const reportCapabilities = () => this.reportCapabilities();
    this.coarseQuery?.addEventListener("change", reportCapabilities, options);
    this.hoverQuery?.addEventListener("change", reportCapabilities, options);

    canvas.addEventListener("contextmenu", event => event.preventDefault(), options);

    canvas.addEventListener("keydown", event => {
      if (this.held.has(event.code) || event.repeat) return;
      this.held.add(event.code);
      this._game?.key(event.code, true);
    }, options);
    window.addEventListener("keyup", event => {
      if (this.held.delete(event.code)) this._game?.key(event.code, false);
    }, options);
    canvas.addEventListener("blur", () => this.clear(), options);
    window.addEventListener("blur", () => this.clear(), options);

    canvas.addEventListener("touchstart", event => {
      this.touchEventsActive = true;
      if (document.activeElement !== canvas) canvas.focus({ preventScroll: true });
      event.preventDefault();
      this.reconcileTouchList(event.touches);
      const bounds = canvas.getBoundingClientRect();
      const r = this.ratio();
      for (let i = 0; i < event.changedTouches.length; i++) {
        const touch = event.changedTouches[i];
        if (!touch) continue;
        const id = touch.identifier >>> 0;
        const x = (touch.clientX - bounds.left) * r;
        const y = (touch.clientY - bounds.top) * r;
        this.touches.set(id, { x: touch.clientX, y: touch.clientY });
        this._game?.touch(id, x, y, TOUCH_PHASES.began);
      }
    }, activeTouchOptions);

    canvas.addEventListener("touchmove", event => {
      if (!this.touchEventsActive) return;
      event.preventDefault();
      this.reconcileTouchList(event.touches);
      const bounds = canvas.getBoundingClientRect();
      const r = this.ratio();
      for (let i = 0; i < event.changedTouches.length; i++) {
        const touch = event.changedTouches[i];
        if (!touch) continue;
        const id = touch.identifier >>> 0;
        const prev = this.touches.get(id);
        const x = (touch.clientX - bounds.left) * r;
        const y = (touch.clientY - bounds.top) * r;
        if (!prev) {
          this.touches.set(id, { x: touch.clientX, y: touch.clientY });
          this._game?.touch(id, x, y, TOUCH_PHASES.began);
          continue;
        }
        this._game?.touch(id, x, y, TOUCH_PHASES.moved);
        this._game?.touch_motion(id, (touch.clientX - prev.x) * r, (touch.clientY - prev.y) * r);
        prev.x = touch.clientX;
        prev.y = touch.clientY;
      }
    }, activeTouchOptions);

    const onTouchEnd = (event: TouchEvent, phase: number) => {
      if (!this.touchEventsActive) return;
      const bounds = canvas.getBoundingClientRect();
      const r = this.ratio();
      for (let i = 0; i < event.changedTouches.length; i++) {
        const touch = event.changedTouches[i];
        if (!touch) continue;
        const id = touch.identifier >>> 0;
        if (!this.touches.delete(id)) continue;
        const x = (touch.clientX - bounds.left) * r;
        const y = (touch.clientY - bounds.top) * r;
        this._game?.touch(id, x, y, phase);
      }
      this.reconcileTouchList(event.touches);
    };

    for (const target of [canvas, window] as const) {
      target.addEventListener("touchend", event => onTouchEnd(event as TouchEvent, TOUCH_PHASES.ended), passiveTouchOptions);
      target.addEventListener("touchcancel", event => onTouchEnd(event as TouchEvent, TOUCH_PHASES.cancelled), passiveTouchOptions);
    }

    canvas.addEventListener("pointerdown", event => {
      if (event.pointerType === "touch") {
        if (this.touchEventsActive) return;
        if (document.activeElement !== canvas) canvas.focus({ preventScroll: true });
        const bounds = canvas.getBoundingClientRect();
        const r = this.ratio();
        this.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
        try { canvas.setPointerCapture(event.pointerId); } catch {}
        event.preventDefault();
        this._game?.touch(event.pointerId, (event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r, TOUCH_PHASES.began);
        return;
      }
      if (document.activeElement !== canvas) canvas.focus({ preventScroll: true });
      const bounds = canvas.getBoundingClientRect();
      const r = this.ratio();
      if (event.isPrimary) this._game?.pointer((event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r);
      if (event.button === 0 && this.mode === CURSOR_MODES.drag) {
        try { canvas.setPointerCapture(event.pointerId); } catch {}
      }
      if (!this.mouseHeld.has(event.button)) {
        this.mouseHeld.add(event.button);
        this._game?.mouse_button(event.button, true);
      }
    }, options);

    canvas.addEventListener("pointermove", event => {
      const bounds = canvas.getBoundingClientRect();
      const r = this.ratio();
      if (event.pointerType === "touch") {
        if (this.touchEventsActive) return;
        const last = this.touches.get(event.pointerId);
        if (!last) return;
        this._game?.touch(event.pointerId, (event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r, TOUCH_PHASES.moved);
        this._game?.touch_motion(event.pointerId, (event.clientX - last.x) * r, (event.clientY - last.y) * r);
        last.x = event.clientX;
        last.y = event.clientY;
        return;
      }
      if (this.locked) { this._game?.motion(event.movementX, event.movementY); return; }
      if (this.last && this.mode !== CURSOR_MODES.captured) {
        this._game?.motion(event.clientX - this.last.x, event.clientY - this.last.y);
      }
      this.last = { x: event.clientX, y: event.clientY };
      if (event.isPrimary) this._game?.pointer((event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r);
    }, options);

    canvas.addEventListener("pointerleave", event => {
      if (event.pointerType === "touch" || !event.isPrimary) return;
      this.last = undefined;
      this._game?.pointer_left();
    }, options);

    const onPointerEnd = (event: PointerEvent, phase: number) => {
      if (event.pointerType === "touch") {
        if (this.touchEventsActive) return;
        if (!this.touches.delete(event.pointerId)) return;
        const bounds = canvas.getBoundingClientRect();
        const r = this.ratio();
        this._game?.touch(event.pointerId, (event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r, phase);
        return;
      }
      if (this.mouseHeld.delete(event.button)) {
        this._game?.mouse_button(event.button, false);
      }
    };

    for (const target of [canvas, window] as const) {
      target.addEventListener("pointerup", event => onPointerEnd(event as PointerEvent, TOUCH_PHASES.ended), options);
      target.addEventListener("pointercancel", event => onPointerEnd(event as PointerEvent, TOUCH_PHASES.cancelled), options);
    }

    canvas.addEventListener("lostpointercapture", event => {
      if (event.pointerType === "touch") {
        if (this.touchEventsActive) return;
        if (!this.touches.delete(event.pointerId)) return;
        const bounds = canvas.getBoundingClientRect();
        const r = this.ratio();
        this._game?.touch(event.pointerId, (event.clientX - bounds.left) * r, (event.clientY - bounds.top) * r, TOUCH_PHASES.cancelled);
      }
    }, options);

    canvas.addEventListener("wheel", event => {
      const unit = event.deltaMode === WheelEvent.DOM_DELTA_PIXEL ? WHEEL_PIXELS : event.deltaMode === WheelEvent.DOM_DELTA_LINE ? WHEEL_LINES : 1;
      this._game?.wheel(-event.deltaX / unit, -event.deltaY / unit);
    }, { signal: this.abort.signal, passive: true });

    canvas.addEventListener("click", () => {
      if (!this._game || this.mode !== CURSOR_MODES.captured || this.locked || !this.canLockPointer) return;
      void Promise.resolve(canvas.requestPointerLock({ unadjustedMovement: true }))
        .catch(() => canvas.requestPointerLock())
        .catch(() => undefined);
    }, options);

    document.addEventListener("pointerlockchange", () => {
      const locked = document.pointerLockElement === canvas;
      if (locked && this.mode !== CURSOR_MODES.captured) {
        if (typeof document.exitPointerLock === "function") document.exitPointerLock();
        return;
      }
      if (this.locked && !locked && document.hasFocus()) {
        this._game?.key("Escape", true);
        this._game?.key("Escape", false);
      }
      this.locked = locked;
      this.last = undefined;
    }, options);
  }

  private reconcileTouchList(active: TouchList) {
    if (this.touches.size === 0) return;
    const live = new Set<number>();
    for (let i = 0; i < active.length; i++) {
      const touch = active[i];
      if (touch) live.add(touch.identifier >>> 0);
    }
    const bounds = this.canvas.getBoundingClientRect();
    const r = this.ratio();
    for (const [id, last] of [...this.touches.entries()]) {
      if (live.has(id)) continue;
      this.touches.delete(id);
      this._game?.touch(id, (last.x - bounds.left) * r, (last.y - bounds.top) * r, TOUCH_PHASES.cancelled);
    }
  }

  get game(): InputGame | undefined {
    return this._game;
  }

  set game(next: InputGame | undefined) {
    this._game = next;
    this.reportCapabilities();
  }

  private reportCapabilities() {
    this._game?.set_input_capabilities(
      this.coarseQuery?.matches ?? false,
      this.hoverQuery?.matches ?? true,
      this.canLockPointer,
    );
  }

  setCursorMode(mode: number) {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode !== CURSOR_MODES.captured) this.unlock();
  }

  pollPads() {
    const game = this._game;
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
    for (const code of this.held) this._game?.key(code, false);
    for (const button of this.mouseHeld) this._game?.mouse_button(button, false);
    for (const id of this.touches.keys()) this._game?.touch(id, 0, 0, TOUCH_PHASES.cancelled);
    this.held.clear();
    this.mouseHeld.clear();
    this.touches.clear();
    this.last = undefined;
    this.unlock();
  }

  dispose() { this.abort.abort(); this.clear(); this._game = undefined; }

  private unlock() {
    if (!this.locked) return;
    this.locked = false;
    if (typeof document.exitPointerLock === "function") document.exitPointerLock();
  }
}
