import assert from "node:assert/strict";
import test from "node:test";
import { BrowserInput } from "../src/input.ts";

class Element extends EventTarget {
  constructor(key) {
    super();
    this.dataset = { key };
    this.captures = new Set();
  }
  focus() {}
  setPointerCapture(pointerId) { this.captures.add(pointerId); }
}

function setup(t) {
  const previousWindow = globalThis.window;
  const previousDocument = globalThis.document;
  const canvas = new Element();
  const forward = new Element("KeyW");
  const jump = new Element("Space");
  const window = new EventTarget();
  globalThis.window = window;
  globalThis.document = { querySelectorAll: () => [forward, jump] };
  const input = new BrowserInput(canvas);
  t.after(() => {
    input.dispose();
    globalThis.window = previousWindow;
    globalThis.document = previousDocument;
  });
  return { input, canvas, forward, jump, window };
}

function send(target, type, fields = {}) {
  target.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), fields));
}
function key(target, type, code) { send(target, type, { code }); }
function pointer(target, type, pointerId) { send(target, type, { pointerId, button: 0 }); }

test("keyboard and touch independently hold the same movement key", t => {
  const { input, canvas, forward, window } = setup(t);
  key(canvas, "keydown", "KeyW");
  pointer(forward, "pointerdown", 1);
  pointer(forward, "pointerup", 1);
  assert.equal(input.forward, 1, "touch release must preserve keyboard movement");
  key(window, "keyup", "KeyW");
  assert.equal(input.forward, 0);

  key(canvas, "keydown", "KeyW");
  pointer(forward, "pointerdown", 2);
  key(window, "keyup", "KeyW");
  assert.equal(input.forward, 1, "keyboard release must preserve touch movement");
  pointer(forward, "pointerup", 2);
  assert.equal(input.forward, 0);
});

test("two pointers retain independent holds and duplicate release events are harmless", t => {
  const { input, forward } = setup(t);
  pointer(forward, "pointerdown", 1);
  pointer(forward, "pointerdown", 2);
  pointer(forward, "pointerup", 1);
  pointer(forward, "lostpointercapture", 1);
  pointer(forward, "pointercancel", 1);
  assert.equal(input.forward, 1);
  pointer(forward, "pointercancel", 2);
  assert.equal(input.forward, 0);
});

test("action edges follow aggregate holds, suppress repeats, and retain a quick tap", t => {
  const { input, canvas, jump, window } = setup(t);
  key(canvas, "keydown", "Space");
  assert.equal(input.held, 1);
  assert.equal(input.pressed, 1);
  input.consumed();
  key(canvas, "keydown", "Space");
  pointer(jump, "pointerdown", 3);
  assert.equal(input.pressed, 0, "repeat or an additional held source is not a new edge");
  key(window, "keyup", "Space");
  assert.equal(input.held, 1);
  assert.equal(input.released, 0);
  pointer(jump, "pointerup", 3);
  assert.equal(input.held, 0);
  assert.equal(input.released, 1);
  input.consumed();
  pointer(jump, "lostpointercapture", 3);
  assert.equal(input.released, 0, "lost capture after release cannot create another edge");

  pointer(jump, "pointerdown", 4);
  pointer(jump, "pointerup", 4);
  assert.deepEqual([input.held, input.pressed, input.released], [0, 1, 1]);
  input.consumed();
  assert.deepEqual([input.held, input.pressed, input.released], [0, 0, 0]);
});

test("touch tracking admits at most 32 pointers and ignores unknown releases", t => {
  const { input, forward } = setup(t);
  for (let id = 1; id <= 33; id++) pointer(forward, "pointerdown", id);
  assert.equal(forward.captures.size, 32);
  for (let id = 1; id <= 32; id++) pointer(forward, "pointerup", id);
  assert.equal(input.forward, 0, "the overflow pointer never owns an input hold");
  pointer(forward, "pointerup", 33);
  assert.equal(input.forward, 0);
});

test("focus reset clears every source and disposal removes all input listeners", t => {
  const { input, canvas, forward, jump, window } = setup(t);
  key(canvas, "keydown", "KeyW");
  pointer(forward, "pointerdown", 1);
  pointer(jump, "pointerdown", 2);
  send(canvas, "blur");
  assert.deepEqual([input.forward, input.held, input.pressed, input.released], [0, 0, 0, 0]);
  pointer(jump, "lostpointercapture", 2);
  key(window, "keyup", "KeyW");
  assert.equal(input.released, 0);
  input.dispose();
  key(canvas, "keydown", "KeyW");
  pointer(jump, "pointerdown", 3);
  assert.deepEqual([input.forward, input.held, input.pressed], [0, 0, 0]);
});
