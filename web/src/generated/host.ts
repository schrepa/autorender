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

export const IMAGE_FORMATS = {"jpeg":{"what":"JPEG, to RGBA with opaque alpha"},"ktx2-rgba8":{"what":"uncompressed RGBA8 KTX2 (VK_FORMAT_R8G8B8A8_UNORM or _SRGB), its base level as stored"},"png":{"what":"PNG: palettes, low bit depths and transparency expand to 8-bit RGBA, 16-bit channels drop to 8"}};

export const HOST_SETTINGS = {"between":"on (default) or off: terrain loading between frames, in the time they leave free, or within frames alone","debug-view":"off (default), kinds, levels, cards, shadows, casters, overdraw, normals, tiles, materials, sky, detail, water_depth, water_speed, water_froude, water_turbulence, water_white, water_reach, water_span, water_eddies or water_roughness: the debug view frames show in place of the shaded image, over the scene's","dynamic-resolution":"host (default), on or off: dynamic resolution of display frames, over the host's choice: on for play in a window or a page, off offscreen and in measured runs","occlusion":"scene (default), on or off: occlusion culling of GPU-presented frames, over the scene's setting","pipeline":"on (default) or off: fixed ticks split across the frames around their due time, on GPU-presented frames","subpixel":"scene (default), on or off: sub-pixel culling of GPU-presented frames, over the scene's setting","taa":"scene (default), on or off: temporal antialiasing of GPU-presented frames, over the scene's setting","terrain-clock":"wall (default) or frames: what terrain times its work by: the host's clock, or frames alone, each a slice of samples_per_frame with none between them"};
