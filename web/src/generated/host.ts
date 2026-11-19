export interface PanicReport {
  location: string;
  message: string;
}

export const PANIC_PREFIX = "engine-panic ";

export const PANIC_GLOBAL = "__enginePanic";

export const WHEEL_PIXELS = 100.0;

export const FRAME_PHASES = {"collecting":5,"compiling":0,"idle":4,"paused":6,"playing":3,"preparing":1,"started":2};

export const CURSOR_MODES = {"captured":2,"drag":0,"free":1};

export const TOUCH_PHASES = {"began":0,"cancelled":3,"ended":2,"moved":1};
