// The page globals scripts drive inside the browser (`page.evaluate`):
// the page declares their types (web/src/engine/dev.ts, profiling.ts);
// scripts read their replies as JSON.
interface Window {
  __engineDev: any;
  __engineProfile: any;
}
