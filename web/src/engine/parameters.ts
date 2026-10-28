import { HOST_SETTINGS } from "../generated/host";

export const HOST_PARAMETERS: Record<string, string> = {
  ...HOST_SETTINGS,
  engine_dev: "1: open the development tools (development builds)",
  prepare: "1: start a development session as ordinary sessions start",
  engine_build: "the watched build a development session is pinned to",
  workers: "N job workers, or off for none",
  jobcache: "off, verify, or the MiB of job outputs the page keeps",
  environment: "development sessions: environment sections (JSON, as the environment control takes them) merged over the scene's once it starts",
};
