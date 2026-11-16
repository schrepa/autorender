export interface Distribution {
  max: number;
  median: number;
  min: number;
  p95: number;
  p99: number;
}

export interface PanicReport {
  location: string;
  message: string;
}

export interface AdapterStatus {
  architecture: string;
  backend: string;
  device_type: string;
  driver: string;
  name: string;
  vendor: string;
}

export const PANIC_PREFIX = "engine-panic ";

export const PANIC_GLOBAL = "__enginePanic";

export const FRAME_INTERVAL_MS = 16.666666666666668;

export const FRAME_CORES = 2;

export const JOB_CACHE_MIB = 1024;

export const PACK_CACHE_MIB = 1024;

export const WHEEL_PIXELS = 100.0;

export const BENCHMARK_LIMIT_MS = 10000;

export const READ_BACK_LIMIT_MS = 120000;

export const FRAME_PHASES = {"collecting":5,"compiling":0,"idle":4,"playing":3,"preparing":1,"started":2};

export const HOST_PAUSES = {"focus":0,"hidden":1,"suspended":2};

export const CURSOR_MODES = {"captured":2,"drag":0,"free":1};

export const TOUCH_PHASES = {"began":0,"cancelled":3,"ended":2,"moved":1};

export const IMAGE_FORMATS = ["png","jpeg","ktx2-rgba8"];

export const HOST_SETTINGS = {"job-clock":"wall|frames"};

export const HOST_OVERLAY = "set";
