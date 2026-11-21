import type { FrameStats } from "./stats";

export interface DevTake {
  replies: [number, string][];
  pictures: { file: string; width: number; height: number }[];
  reload: boolean;
}

export type SceneFileList = { uri: string; group: string }[];

export type Json<T> = string & { readonly json?: T };

export function parseJson<T>(json: Json<T>): T {
  return JSON.parse(json) as T;
}

export interface Runtime {
  key(code: string, down: boolean): void;
  mouse_button(button: number, down: boolean): void;
  motion(dx: number, dy: number): void;
  pointer(x: number, y: number): void;
  pointer_left(): void;
  wheel(right_notches: number, up_notches: number): void;
  pad(index: number, buttons: number, axes: Float32Array): void;
  touch(id: number, x: number, y: number, phase: number): void;
  touch_motion(id: number, dx: number, dy: number): void;
  animation_frame(timestamp: number, width: number, height: number): number;
  suspend(): void;
  focused(focused: boolean): void;
  hidden(hidden: boolean): void;
  reset_camera(): void;
  reload_scene(source: string): void;
  file(uri: string, bytes: Uint8Array): void;
  file_failed(uri: string, error: string): void;
  set_file_root(root: string): void;
  set_pixel_ratio(ratio: number): void;
  set_safe_insets(top: number, right: number, bottom: number, left: number): void;
  set_input_capabilities(coarse_pointer: boolean, can_hover: boolean, can_lock_pointer: boolean): void;
  apply_host_settings(query: string): void;
  take_sound(): Float64Array;
  take_sound_bytes(): Uint8Array;
  sound_opened(running: boolean, rate: number, channels: number, latency_seconds: number): void;
  sound_open_failed(error: string): void;
  sound_decoded(sound: number, frames: number, rate: number, channels: number): void;
  sound_failed(sound: number, error: string): void;
  voice_ended(voice: number): void;
  voice_dropped(voice: number, error: string): void;
  stats(): Json<FrameStats>;
  cursor_mode(): number;
  max_pixel_ratio(): number;
  files(): Json<SceneFileList>;
  dev_push?(line: string): number;
  dev_take?(): Json<DevTake>;
  dev_pixels?(): Uint8Array;
  dev_reloaded?(source: string): void;
  dev_reload_ended?(error?: string): void;
  free(): void;
}
