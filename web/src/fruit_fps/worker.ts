import { serveJobs } from "../engine/job-worker";

serveJobs(() => import("../../pkg/fruit_fps"));
