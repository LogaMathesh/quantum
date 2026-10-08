const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error("Make sure the FastAPI backend is running and accessible.");
  }

  if (!response.ok) {
    let detail;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string"
        ? body.detail
        : body.detail?.map?.((issue) => issue.msg).join(", ");
    } catch {
      detail = "";
    }
    throw new Error(detail || `API request failed (${response.status}).`);
  }
  return response.json();
}

export function getHealth() {
  return request("/api/health");
}

export function getModels() {
  return request("/api/models");
}

export function getResults() {
  return request("/api/results");
}

export function predict(input) {
  return request("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}
