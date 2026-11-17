import { HOST_SETTINGS } from "../generated/host";

export const HOST_PARAMETERS: Record<string, string> = {
  ...HOST_SETTINGS,
  engine_dev: "1",
  prepare: "1",
  workers: "N|off",
  jobcache: "off|verify|N",
};
