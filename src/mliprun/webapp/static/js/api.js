// REST wrapper for the mliprun web app (PLAN section 4).
// Every mutating request carries the `X-Mliprun: 1` header (PLAN D5).
// Errors are thrown as ApiError with the server's {error: {code, message, details}}.

// `transport.fetch` is the only network entry point, so a test double can
// replace it without touching the rest of the app.
export const transport = {
  fetch: (...args) => window.fetch(...args),
};

export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = Array.isArray(details) ? details : [];
  }
}

const MUTATING = new Set(["POST", "PUT", "PATCH", "DELETE"]);

async function request(method, path, { json, form, query } = {}) {
  let url = path;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null) qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += "?" + s;
  }
  const headers = { Accept: "application/json" };
  if (MUTATING.has(method)) headers["X-Mliprun"] = "1";
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form !== undefined) {
    body = form; // browser sets the multipart boundary
  }

  let resp;
  try {
    resp = await transport.fetch(url, { method, headers, body });
  } catch (err) {
    throw new ApiError(0, "network", `Cannot reach the mliprun server (${err.message || err}).`, []);
  }

  let data = null;
  const text = await resp.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!resp.ok) {
    const e = data && data.error ? data.error : null;
    throw new ApiError(
      resp.status,
      e ? e.code : "http_" + resp.status,
      e ? e.message : `HTTP ${resp.status} ${resp.statusText || ""}`.trim(),
      e ? e.details : [],
    );
  }
  return data;
}

export const api = {
  health: () => request("GET", "/api/health"),
  calcTypes: () => request("GET", "/api/calc-types"),
  calculators: (target) => request("GET", "/api/calculators", { query: target && target !== "local" ? { target } : undefined }),
  listMachines: () => request("GET", "/api/machines"),
  addMachine: (body) => request("POST", "/api/machines", { json: body }),
  updateMachine: (mid, body) => request("PUT", `/api/machines/${encodeURIComponent(mid)}`, { json: body }),
  deleteMachine: (mid) => request("DELETE", `/api/machines/${encodeURIComponent(mid)}`),
  testMachine: (mid) => request("POST", `/api/machines/${encodeURIComponent(mid)}/test`),
  addEnv: (mid, body) => request("POST", `/api/machines/${encodeURIComponent(mid)}/envs`, { json: body }),
  updateEnv: (mid, eid, body) => request("PUT", `/api/machines/${encodeURIComponent(mid)}/envs/${encodeURIComponent(eid)}`, { json: body }),
  deleteEnv: (mid, eid) => request("DELETE", `/api/machines/${encodeURIComponent(mid)}/envs/${encodeURIComponent(eid)}`),
  probeEnv: (mid, eid) => request("POST", `/api/machines/${encodeURIComponent(mid)}/envs/${encodeURIComponent(eid)}/probe`),
  listUploads: () => request("GET", "/api/uploads"),
  getUpload: (id) => request("GET", `/api/uploads/${encodeURIComponent(id)}`),
  upload(files, format) {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    if (format) form.append("format", format);
    return request("POST", "/api/uploads", { form });
  },
  listJobs: (query) => request("GET", "/api/jobs", { query }),
  getJob: (id) => request("GET", `/api/jobs/${encodeURIComponent(id)}`),
  submitJob: (payload) => request("POST", "/api/jobs", { json: payload }),
  cancelJob: (id) => request("POST", `/api/jobs/${encodeURIComponent(id)}/cancel`),
  events: (id, since = 0, limit = 1000) =>
    request("GET", `/api/jobs/${encodeURIComponent(id)}/events`, { query: { since, limit } }),
  checkPaths(location, folders, target, localLocation) {
    const json = { location, folders };
    if (target && target !== "local") {
      json.target = target;
      json.local_location = localLocation || null;
    }
    return request("POST", "/api/paths/check", { json });
  },
  files: (id) => request("GET", `/api/jobs/${encodeURIComponent(id)}/files`),
};

export function fileUrl(jobId, path, inline = false) {
  const p = path.split("/").map(encodeURIComponent).join("/");
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${p}${inline ? "?inline=1" : ""}`;
}

export function zipUrl(jobId) {
  return `/api/jobs/${encodeURIComponent(jobId)}/zip`;
}
