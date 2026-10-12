/**
 * How a check's process ends: 0 when it passes, 3 when an assertion fails
 * (what it checks is wrong), 1 for anything else (it could not check). The
 * factory tells a check that caught a fault from one that broke by these.
 * Importing this module is enough; the harness does. Node prints the error
 * as always.
 */
// @ts-check
let failedAssertion = false;
process.on("uncaughtExceptionMonitor", error => { failedAssertion = /** @type {any} */ (error)?.code === "ERR_ASSERTION"; });
process.on("exit", code => { if (code === 1 && failedAssertion) process.exitCode = 3; });
