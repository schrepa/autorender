// @ts-check
import { failOnDisabledFeatures } from "./exercised.mjs";

failOnDisabledFeatures();
let failedAssertion = false;
process.on("uncaughtExceptionMonitor", error => { failedAssertion = /** @type {any} */ (error)?.code === "ERR_ASSERTION"; });
process.on("exit", code => { if (code === 1 && failedAssertion) process.exitCode = 3; });
