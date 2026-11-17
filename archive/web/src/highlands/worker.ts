import { serveJobs } from "../engine/job-worker";

serveJobs(() => import("../../pkg/highlands"));
