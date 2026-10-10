const movement = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
const MAX_TOUCH_POINTERS = 32;
// Rust's `ActionId` holds 32 actions.
const MAX_ACTIONS = 32;

/** Bounded input accumulation. One batch per frame; no growing event queue. */
export class BrowserInput {
  // Each physical source owns its hold. Releasing one source must not release
  // another finger or a keyboard key bound to the same logical input.
  private keys = new Map<string, number>();
  private keyboard = new Set<string>();
  private pointers = new Map<number, { key: string; target: HTMLElement }>();
  private abort = new AbortController();
  private dragging: number | undefined;
  private x = 0;
  private y = 0;
  yaw = 0;
  pitch = 0;
  held = 0;
  pressed = 0;
  released = 0;

  // Each bound key's action bit.
  private bindings: Map<string, number>;

  /** Input on `canvas`, with the game's `actions`: keys by action index. */
  constructor(canvas: HTMLCanvasElement, actions: Record<string, number>) {
    this.bindings = new Map(Object.entries(actions).map(([key, action]) => {
      if (!Number.isInteger(action) || action < 0 || action >= MAX_ACTIONS) throw new Error(`Action ${action} of ${key} must be 0 to ${MAX_ACTIONS - 1}.`);
      return [key, 1 << action];
    }));
    const supported = (key: string) => movement.has(key) || this.bindings.has(key);
    const options = { signal: this.abort.signal };
    canvas.addEventListener("keydown", event => {
      if (supported(event.code)) {
        event.preventDefault();
        if (!this.keyboard.has(event.code)) {
          this.keyboard.add(event.code); this.down(event.code);
        }
      }
    }, options);
    window.addEventListener("keyup", event => {
      if (this.keyboard.delete(event.code)) this.up(event.code);
    }, options);
    canvas.addEventListener("blur", () => this.clear(), options);
    window.addEventListener("blur", () => this.clear(), options);
    canvas.addEventListener("pointerdown", event => {
      if (this.dragging !== undefined || event.button !== 0) return;
      canvas.focus(); this.dragging = event.pointerId;
      this.x = event.clientX; this.y = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    }, options);
    canvas.addEventListener("pointermove", event => {
      if (this.dragging !== event.pointerId) return;
      this.yaw = Math.max(-1000, Math.min(1000, this.yaw + event.clientX - this.x));
      this.pitch = Math.max(-1000, Math.min(1000, this.pitch + event.clientY - this.y));
      this.x = event.clientX; this.y = event.clientY;
    }, options);
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
      canvas.addEventListener(name, event => { if (this.dragging === event.pointerId) this.dragging = undefined; }, options);
    }
    for (const button of document.querySelectorAll<HTMLElement>("[data-key]")) {
      const key = button.dataset.key!;
      if (!supported(key)) continue;
      button.addEventListener("pointerdown", event => {
        if (event.button !== 0 || this.pointers.has(event.pointerId) || this.pointers.size >= MAX_TOUCH_POINTERS) return;
        event.preventDefault(); button.setPointerCapture(event.pointerId);
        this.pointers.set(event.pointerId, { key, target: button }); this.down(key);
      }, options);
      for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
        button.addEventListener(name, event => {
          const pointer = this.pointers.get(event.pointerId);
          if (pointer?.target !== button) return;
          this.pointers.delete(event.pointerId); this.up(pointer.key);
        }, options);
      }
    }
  }

  private down(key: string) {
    const sources = this.keys.get(key) ?? 0;
    this.keys.set(key, sources + 1);
    if (sources > 0) return;
    const bit = this.bindings.get(key) ?? 0;
    this.held |= bit; this.pressed |= bit;
  }
  private up(key: string) {
    const sources = this.keys.get(key);
    if (sources === undefined) return;
    if (sources > 1) { this.keys.set(key, sources - 1); return; }
    this.keys.delete(key);
    const bit = this.bindings.get(key) ?? 0;
    this.held &= ~bit; this.released |= bit;
  }
  get forward() { return Number(this.keys.has("KeyW") || this.keys.has("ArrowUp")) - Number(this.keys.has("KeyS") || this.keys.has("ArrowDown")); }
  get right() { return Number(this.keys.has("KeyD") || this.keys.has("ArrowRight")) - Number(this.keys.has("KeyA") || this.keys.has("ArrowLeft")); }
  get upAxis() { return Number(this.keys.has("KeyE")) - Number(this.keys.has("KeyQ")); }
  consumed() { this.yaw = 0; this.pitch = 0; this.pressed = 0; this.released = 0; }
  clear() { this.keys.clear(); this.keyboard.clear(); this.pointers.clear(); this.held = 0; this.dragging = undefined; this.consumed(); }
  dispose() { this.abort.abort(); this.clear(); }
}
