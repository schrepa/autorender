/** Uncertainty-weighted TAA resets color and uncertainty through toggles,
 * resize, workload replacement, exact camera changes and renderer reload;
 * ordinary frames retain content and moving detail, with Highlands captures. */
import { checkHistory } from "./taa-history-checks.mjs";

try {
  await checkHistory("taa-uncertainty-smoke");
} catch (error) {
  console.error(error);
  process.exitCode = error?.code === "ERR_ASSERTION" ? 3 : 1;
}
