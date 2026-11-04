import type { JobOrder } from "../engine/jobs";
import type { Distribution } from "./host";
import type { Reply, Step } from "./reply";
import type { FrameStats } from "./stats";

export type Json<T> = string & { readonly json?: T };

export function parseJson<T>(json: Json<T>): T {
  return JSON.parse(json) as T;
}

export interface Runtime {
  frame(timestamp: number, width: number, height: number): void;
  key(code: string, down: boolean): boolean;
  look(dx: number, dy: number): void;
  motion(dx: number, dy: number): void;
  pointer(x: number, y: number): void;
  pointer_left(): void;
  wheel(right_notches: number, up_notches: number): void;
  pad(buttons: number, axes: Float32Array): void;
  clear_input(): void;
  animation_frame(timestamp: number, width: number, height: number): number;
  work_while_compiling(width: number, height: number): boolean;
  load_between_frames(): boolean;
  set_preparation(enabled: boolean): void;
  set_job_workers(workers: number): void;
  set_job_cache(caching: boolean, verify: boolean): void;
  take_jobs(): JobOrder[];
  job_kept(id: number, key: string, output?: Uint8Array): void;
  lose_job_worker(worker: number, error: string, replaced: boolean): void;
  job_task(): number | undefined;
  job_task_done(worker: number): void;
  complete_job(id: number, output: Uint8Array): void;
  fail_job(id: number, error: string): void;
  report_job_cache(hits: number, misses: number, bytes: number, mismatches: number): void;
  suspend(): void;
  reset_camera(): void;
  reload_scene(source: string, files: unknown): void;
  take_level_request(): string | undefined;
  report_level(done: number, total: number): void;
  fail_level(message: string): void;
  set_pixel_ratio(ratio: number): void;
  set_setting(key: string, value: string): void;
  apply_host_settings(query: string): void;
  stats(): Json<FrameStats>;
  cursor_mode(): number;
  job_hash(): Uint8Array;
  distribution(values: Float64Array): Json<Distribution | null>;
  dev_request?(line: string, context: string): Json<Step>;
  dev_resume?(outcome: string, context: string): Json<Step>;
  dev_reloaded?(source: string, context: string, files?: unknown): Json<Step>;
  dev_refusal?(error: string): Json<Reply>;
  profiling_start?(max_events: number): void;
  profiling_stop?(): string;
  profiling_poll?(): void;
  profiling_report?(): string;
  profiling_status?(): string;
  free(): void;
}
