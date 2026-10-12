import { HOST_SETTINGS } from "../generated/host";

/** The page parameters the engine's host reads, with what each does: the
 * engine switches every host takes (which the engine reads), and the
 * page's own. The web check holds every parameter src/engine reads to this
 * list, and the development bridge's describe lists it with the game's
 * own. */
export const HOST_PARAMETERS: Record<string, string> = {
  ...HOST_SETTINGS,
  engine_dev: "1: open the development tools (development builds)",
  prepare: "1: start a development session as ordinary sessions start",
  engine_build: "the watched build a development session is pinned to",
  between: "off: load terrain only within frames",
  workers: "N job workers, or off for none",
  jobcache: "off, verify, or the MiB of job outputs the page keeps",
};
