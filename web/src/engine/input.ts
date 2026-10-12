import type { Runtime } from "../generated/runtime";

const MAX_TOUCH_POINTERS = 32;

/** The page's keyboard, drag and touch-button events as the input the
 * engine gathers between frames (`Runtime.key` and `Runtime.look`), which
 * keeps the holds, action edges and look deltas. The page tracks only
 * which keyboard keys and touch pointers hold what, and the drag. */
export class BrowserInput {
  /** The game input goes to; none while the page takes input away. */
  game: Pick<Runtime, "key" | "look"> | undefined;
  // The keys the keyboard holds that the game uses. Each physical source
  // owns its hold: releasing one does not release another finger or a key
  // bound to the same input.
  private keyboard = new Set<string>();
  private pointers = new Map<number, { key: string; target: HTMLElement }>();
  private abort = new AbortController();
  private dragging: number | undefined;
  private x = 0;
  private y = 0;

  constructor(canvas: HTMLCanvasElement) {
    const options = { signal: this.abort.signal };
    canvas.addEventListener("keydown", event => {
      // A held key's repeats keep it from the page; the engine counts it once.
      if (this.keyboard.has(event.code)) { event.preventDefault(); return; }
      if (this.game?.key(event.code, true)) { this.keyboard.add(event.code); event.preventDefault(); }
    }, options);
    window.addEventListener("keyup", event => {
      if (this.keyboard.delete(event.code)) this.game?.key(event.code, false);
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
      this.game?.look(event.clientX - this.x, event.clientY - this.y);
      this.x = event.clientX; this.y = event.clientY;
    }, options);
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
      canvas.addEventListener(name, event => { if (this.dragging === event.pointerId) this.dragging = undefined; }, options);
    }
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

  /** Nothing the page tracks holds anything; the engine's holds go with
   * its suspension or `clear_input`. */
  clear() { this.keyboard.clear(); this.pointers.clear(); this.dragging = undefined; }
  dispose() { this.abort.abort(); this.clear(); this.game = undefined; }
}
