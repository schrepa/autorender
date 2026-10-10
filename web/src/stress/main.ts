// The Stress Lab: `?source=/scenes/NAME.scene.json` loads that authored
// scene in place of a workload; otherwise the workloads run.
if (new URLSearchParams(location.search).has("source")) void import("./scene");
else void import("./workload");
