import { HOST_SETTINGS, MAX_JOB_CACHE_MIB, MAX_WORKERS } from "../generated/host";

export const HOST_PARAMETERS: Record<string, string> = {
  ...HOST_SETTINGS,
  engine_dev: "1",
  prepare: "1",
  engine_build: "ID",
  workers: `0-${MAX_WORKERS}|off`,
  jobcache: `off|verify|0-${MAX_JOB_CACHE_MIB}`,
  smoke: "NAME,...|all",
  environment: "JSON",
};
