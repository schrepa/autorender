import type { JobOrder } from "../engine/jobs";
import type { SoundOrder } from "../engine/sound";
import type { FrameStats } from "./stats";

export interface DevTake {
  replies: [number, string][];
  pictures: { run: string | null; file: string; width: number; height: number }[];
  reload: boolean;
}

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
  set_preparation(enabled: boolean): void;
  set_job_workers(workers: number): boolean;
  set_job_cache(caching: boolean, verify: boolean): void;
  take_jobs(): JobOrder[];
  job_kept(id: number, key: string, pack: boolean, output?: Uint8Array): void;
  lose_job_worker(worker: number, error: string, replaced: boolean): void;
  job_task(): number | undefined;
  job_task_done(worker: number): void;
  complete_job(id: number, output: Uint8Array): void;
  fail_job(id: number, error: string): void;
  report_job_cache(pack_hits: number, store_hits: number, misses: number, bytes: number, mismatches: number): void;
  suspend(): void;
  pause(reason: number): void;
  resume(reason: number): void;
  reset_camera(): void;
  reload_scene(source: string, files: unknown): void;
  set_file_root(root: string): void;
  set_pixel_ratio(ratio: number): void;
  apply_host_settings(query: string): void;
  take_sound(): SoundOrder[];
  sound_opened(running: boolean, rate: number, channels: number, latency_seconds: number): void;
  sound_open_failed(error: string): void;
  sound_decoded(sound: number, frames: number, rate: number, channels: number): void;
  sound_failed(sound: number, error: string): void;
  voice_ended(voice: number): void;
  voice_dropped(voice: number, error: string): void;
  stats(): Json<FrameStats>;
  cursor_mode(): number;
  job_hash(): Uint8Array;
  dev_push?(line: string): number;
  dev_take?(): Json<DevTake>;
  dev_pixels?(): Uint8Array;
  dev_reloaded?(source: string, files?: unknown): void;
  dev_reload_failed?(error: string): void;
  dev_packs?(packs: string, records: number, ms: number): void;
  free(): void;
}
