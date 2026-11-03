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

export const QUIET_FRAMES = 3;

export const PREPARE_LIMIT_MS = 60000;

export const FRAME_INTERVAL_MS = 16.666666666666668;

export const PACE_SMOOTHING = 0.1;

export const PACE_INTERVAL_LIMIT_MS = 100.0;

export const BETWEEN_MARGIN_MS = 4.0;

export const FRAME_CORES = 2;

export const MAX_WORKERS = 32;

export const JOB_CACHE_MIB = 1024;

export const MAX_JOB_CACHE_MIB = 9999999;

export const MAX_ACTIONS = 32;

export const LOOK_LIMIT = 1000.0;

export const WHEEL_PIXELS = 100.0;

export const BENCHMARK_LIMIT_MS = 10000;

export const MAX_PIXEL_RATIO = 1.5;

export const MAX_VIEW_WIDTH = 2560;

export const MAX_VIEW_HEIGHT = 1440;

export const DYNAMIC_STEP = 0.03125;

export const DYNAMIC_OVER = 1.05;

export const DYNAMIC_UNDER = 0.85;

export const DYNAMIC_AIM = 0.9;

export const DYNAMIC_LOWER_FRAMES = 3;

export const DYNAMIC_RAISE_FRAMES = 30;

export const DYNAMIC_PROBE_FRAMES = 120;

export const FRAME_PHASES = {"compiling":0,"idle":4,"playing":3,"preparing":1,"started":2};

export const CURSOR_MODES = {"captured":2,"drag":0,"free":1};

export const IMAGE_FORMATS = ["png","jpeg","ktx2-rgba8"];

export const HOST_SETTINGS = {"between":"on|off","pipeline":"on|off","terrain-clock":"wall|frames"};

export const HOST_OVERLAY = "set";
