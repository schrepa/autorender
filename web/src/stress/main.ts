import "./page.css";
if (new URLSearchParams(location.search).has("source")) void import("./scene");
else void import("./workload");
