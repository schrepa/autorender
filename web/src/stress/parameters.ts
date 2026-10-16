/** The page parameters the Stress Lab reads beside the host's, with what
 * each does; the web check holds every parameter src/stress reads to them. */
export const STRESS_PARAMETERS: Record<string, string> = {
  source: "/scenes/NAME.scene.json: run that authored scene in place of a workload",
  without: "scene mode: comma-separated features the engine leaves out (terrain, world_streaming, navigation, ocean, animation, time_of_day, deformation, collision)",
  preset: "start from that workload preset",
  config: "start from that workload configuration (JSON), as the page's share link writes it",
  look: "lighting preset over the scene's own",
  effects: "all, none, or comma-separated no-<effect> exclusions",
  visibility: "on: shade virtual geometry through the visibility buffer",
  "depth-occupancy": "on: filter forward opaque light lists by current depth occupancy",
  lod: "off: draw every instance as its mesh",
  impostors: "off: draw no impostors",
};
