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

export const SOUND_ORDERS = {"decode":{"code":1,"length":2,"size":3,"sound":1},"master":{"code":5,"gain":1,"seconds":2,"size":3},"open":{"code":0,"latency_given":1,"latency_seconds":2,"size":3},"start":{"code":2,"gain":3,"looping":6,"offset_seconds":7,"pan":4,"rate":5,"size":8,"sound":2,"voice":1},"stop":{"code":4,"seconds":2,"size":3,"voice":1},"tune":{"code":3,"gain":2,"pan":3,"rate":4,"seconds":5,"size":6,"voice":1}};
