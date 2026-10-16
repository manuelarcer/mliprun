// Bootstrapping and hash routing (#/jobs/<id>).
import { api } from "./api.js";
import { ws } from "./ws.js";
import { el, libsAvailable } from "./util.js";
import { StructuresPanel } from "./structures.js";
import { CalcForm } from "./form.js";
import { JobList, JobDetail } from "./jobs.js";
import { MachinesPanel } from "./machines.js";

function libBanner() {
  const { mol, plotly } = libsAvailable();
  if (mol && plotly) return;
  const missing = [!mol && "3Dmol.js (3D viewer)", !plotly && "Plotly (plots)"].filter(Boolean).join(" and ");
  document.getElementById("banners").append(
    el(
      "div",
      { class: "banner banner-warn", testid: "lib-error-banner", role: "alert" },
      `${missing} could not be loaded. The 3D viewers and plots need internet access to their CDN; everything else (uploads, submitting, results, files) still works.`,
    ),
  );
}

function fatal(message) {
  document.getElementById("banners").append(el("div", { class: "banner banner-error", role: "alert" }, message));
}

async function boot() {
  libBanner();

  const conn = document.getElementById("conn-state");
  ws.onState((state, info) => {
    conn.dataset.state = state;
    conn.textContent = state === "open" ? "live" : state === "connecting" ? "connecting…" : "offline, retrying";
    if (state === "open" && info.reconnect) jobList.refresh();
  });

  let calcTypes, calculators;
  try {
    [calcTypes, calculators] = await Promise.all([api.calcTypes(), api.calculators()]);
  } catch (err) {
    fatal(`Could not load the calculation schema from the server: ${err.message}`);
    return;
  }
  const health = api.health().catch(() => null);
  health.then((h) => {
    if (!h) return;
    document.getElementById("version").textContent = `mliprun ${h.mliprun_version}`;
    document.getElementById("runs-dir").textContent = h.runs_dir;
  });

  const structures = new StructuresPanel(document.getElementById("structures-panel"));
  const jobList = new JobList(document.getElementById("queue-panel"));
  const detail = new JobDetail(document.getElementById("detail-panel"), {
    calcTypes,
    onJobChange: (d) => {
      const j = jobList.jobs.get(d.job_id);
      if (j && j.status !== d.status) {
        jobList.upsert({ job_id: d.job_id, status: d.status });
      }
    },
  });
  const form = new CalcForm(document.getElementById("form-panel"), {
    calcTypes,
    calculators,
    structures,
    onSubmitted: (jobs) => {
      for (const j of jobs) jobList.upsert(j);
      if (jobs.length) location.hash = `#/jobs/${jobs[0].job_id}`;
    },
  });

  const machinesPanel = new MachinesPanel({ onChange: (machines) => form.setMachines(machines) });
  document.getElementById("machines-button").addEventListener("click", () => machinesPanel.open());
  api
    .listMachines()
    .then(({ machines }) => form.setMachines(machines))
    .catch(() => form.setMachines([])); // older server without machines: local only

  health.then((h) => form.setDefaultLocation(h ? h.runs_dir : null));

  // PLAN 6.5: subscribe "*" first, then list jobs.
  ws.connect();
  ws.subscribe("*");
  await Promise.all([jobList.refresh(), structures.loadExisting()]);

  const route = () => {
    const m = location.hash.match(/^#\/jobs\/([^/?#]+)/);
    if (m) {
      const id = decodeURIComponent(m[1]);
      jobList.select(id);
      if (detail.jobId !== id) detail.open(id);
      document.getElementById("detail-panel").scrollIntoView({ block: "start", behavior: "smooth" });
    } else {
      jobList.select(null);
      detail.showPlaceholder();
    }
  };
  window.addEventListener("hashchange", route);
  route();
}

boot().catch((err) => {
  console.error(err);
  fatal(`The page failed to start: ${err.message}`);
});
