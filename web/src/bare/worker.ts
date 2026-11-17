import { serveJobs } from "../engine/job-worker";

serveJobs(() => import("../../pkg/bare_game"));
