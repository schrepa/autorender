/** The page parameters the engine's host reads, with what each does. The
 * web check holds every parameter src/engine reads to this list, and the
 * development bridge's describe lists it with the game's own. */
export const HOST_PARAMETERS: Record<string, string> = {
  engine_dev: "1: open the development tools (development builds)",
  prepare: "1: start a development session as ordinary sessions start",
  engine_build: "the watched build a development session is pinned to",
  between: "off: load terrain only within frames",
  workers: "N job workers, or off for none",
  jobcache: "off, verify, or the MiB of job outputs the page keeps",
  presentation: "gpu (default) or cpu: how ordinary frames are presented",
  pipeline: "on (default) or off: split fixed ticks across frames",
  taa: "on or off: temporal antialiasing over the world's setting",
  occlusion: "on or off: occlusion culling over the world's setting",
  subpixel: "on or off: sub-pixel culling over the world's setting",
};
