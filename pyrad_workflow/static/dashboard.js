const STORAGE_KEYS = {
  extractFeatures: "pyrad.extract.features",
  selectedFeatures: "pyrad.selected.features",
};

let appDefaults = {};

function setInputValue(input, value, force = false) {
  if (!input || value == null || value === "") {
    return;
  }
  if (!force && input.dataset.dirty === "true") {
    return;
  }
  input.value = value;
}

function markFieldDirty(input) {
  if (input) {
    input.dataset.dirty = "true";
  }
}

function collectFormPayload(form) {
  const payload = {};
  for (const [key, value] of new FormData(form).entries()) {
    payload[key] = value;
  }
  return payload;
}

function getNested(object, path, fallback) {
  let current = object;
  for (const key of path) {
    if (current == null || typeof current !== "object" || !(key in current)) {
      return fallback;
    }
    current = current[key];
  }
  return current == null ? fallback : current;
}

function pathJoin(base, leaf) {
  if (!base) {
    return "";
  }
  return `${String(base).replace(/[\\/]+$/, "")}\\${leaf}`;
}

function findFilePath(files, fileName) {
  const match = (files || []).find((file) => file && file.name === fileName);
  return match ? match.path : "";
}

function getFeatureCandidate() {
  const currentFeatureInput = document.querySelector('input[name="features"]');
  if (currentFeatureInput && currentFeatureInput.value) {
    return currentFeatureInput.value;
  }
  return (
    window.localStorage.getItem(STORAGE_KEYS.selectedFeatures) ||
    window.localStorage.getItem(STORAGE_KEYS.extractFeatures) ||
    ""
  );
}

function saveKnownPaths(endpoint, data) {
  const files = data.files || [];
  if (endpoint === "/api/extract") {
    const featuresPath = findFilePath(files, "features.csv") || pathJoin(data.output_dir, "features.csv");
    if (featuresPath) {
      window.localStorage.setItem(STORAGE_KEYS.extractFeatures, featuresPath);
    }
  }
  if (endpoint === "/api/select") {
    const selectedPath = findFilePath(files, "selected_features.csv") || pathJoin(data.output_dir, "selected_features.csv");
    if (selectedPath) {
      window.localStorage.setItem(STORAGE_KEYS.selectedFeatures, selectedPath);
    }
  }
  if (endpoint === "/api/full") {
    const featuresPath = findFilePath(files, "features.csv");
    const selectedPath = findFilePath(files, "selected_features.csv");
    if (featuresPath) {
      window.localStorage.setItem(STORAGE_KEYS.extractFeatures, featuresPath);
    }
    if (selectedPath) {
      window.localStorage.setItem(STORAGE_KEYS.selectedFeatures, selectedPath);
    }
  }
}

async function fetchConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error(`Config request failed: ${response.status}`);
  }
  const payload = await response.json();
  appDefaults = payload.defaults || {};

  document.querySelectorAll("[data-default]").forEach((input) => {
    const key = input.getAttribute("data-default");
    setInputValue(input, appDefaults[key], true);
    delete input.dataset.dirty;
  });

  document.querySelectorAll("[data-store-key]").forEach((input) => {
    const storageKey = input.getAttribute("data-store-key");
    if (storageKey === "extract-features") {
      setInputValue(input, window.localStorage.getItem(STORAGE_KEYS.extractFeatures), false);
    }
    if (storageKey === "selected-features") {
      setInputValue(input, window.localStorage.getItem(STORAGE_KEYS.selectedFeatures), false);
    }
  });
}

function setStatus(kind, title, message) {
  const box = document.getElementById("status-box");
  if (!box) {
    return;
  }
  box.classList.remove("hidden", "notice-success", "notice-error");
  box.classList.add(kind === "error" ? "notice-error" : "notice-success");
  box.innerHTML = `<strong>${title}</strong><p>${message}</p>`;
}

function createMetricCard(card) {
  const node = document.createElement("div");
  node.className = `metric-card accent-${card.accent || "brand"}`;
  node.innerHTML = `
    <span class="metric-label">${card.label}</span>
    <strong class="metric-value">${card.value}</strong>
    <span class="metric-detail">${card.detail || ""}</span>
  `;
  return node;
}

function renderCardsInto(container, cards) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  (cards || []).forEach((card) => container.appendChild(createMetricCard(card)));
}

function renderSteps(container, steps) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  (steps || []).forEach((item) => {
    const node = document.createElement("div");
    node.className = "step-card";
    node.innerHTML = `<strong>${item.step}</strong><p>${item.summary || item.detail || ""}</p>`;
    container.appendChild(node);
  });
}

function renderChart(chart) {
  const title = document.getElementById("chart-title");
  const panel = document.getElementById("result-chart");
  if (!title || !panel) {
    return;
  }

  panel.innerHTML = "";
  if (!chart || !chart.items || chart.items.length === 0) {
    title.textContent = "Visual Summary";
    panel.className = "chart-panel empty-state";
    panel.textContent = "Run this step to see a chart.";
    return;
  }

  title.textContent = chart.title || "Visual Summary";
  panel.className = "chart-panel";
  const maxValue = Math.max(...chart.items.map((item) => Number(item.value) || 0), 1);

  chart.items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    const width = `${Math.max((Number(item.value) / maxValue) * 100, 2)}%`;
    row.innerHTML = `
      <div class="chart-meta">
        <span>${item.label}</span>
        <strong>${item.value}</strong>
      </div>
      <div class="chart-track">
        <div class="chart-bar accent-${item.accent || "brand"}" style="width: ${width};"></div>
      </div>
    `;
    panel.appendChild(row);
  });
}

function buildTable(rows) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  const headerRow = document.createElement("tr");

  Object.keys(rows[0]).forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    headerRow.appendChild(th);
  });

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    Object.values(row).forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  thead.appendChild(headerRow);
  table.appendChild(thead);
  table.appendChild(tbody);
  return table;
}

function renderTable(container, rows) {
  if (!container) {
    return;
  }
  container.innerHTML = "";
  if (!rows || rows.length === 0) {
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");
  container.appendChild(buildTable(rows));
}

function renderFiles(files) {
  const container = document.getElementById("result-files");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  (files || []).forEach((file) => {
    const link = document.createElement("a");
    link.className = "file-chip";
    link.href = `/download?path=${encodeURIComponent(file.path)}`;
    link.textContent = file.display_path || file.name || file.path;
    container.appendChild(link);
  });
}

function renderMainResult(data) {
  renderCardsInto(document.getElementById("result-cards"), getNested(data, ["insights", "cards"], []));
  renderSteps(document.getElementById("result-steps"), getNested(data, ["insights", "stage_view", "items"], data.steps || []));
  renderChart(getNested(data, ["insights", "bar_chart"], null));
  renderTable(document.getElementById("result-table"), data.table || []);
  renderFiles(data.files || []);
}

async function submitForm(form) {
  const endpoint = form.getAttribute("data-endpoint");
  const button = form.querySelector('button[type="submit"]');
  const originalText = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "Running...";
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectFormPayload(form)),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Request failed");
    }

    saveKnownPaths(endpoint, data);
    setStatus("success", data.title, data.summary);
    renderMainResult(data);
    await loadDataPreview().catch(() => {});
    await loadFeaturePreview().catch(() => {});
  } catch (error) {
    setStatus("error", "Execution failed", error.message || String(error));
    renderMainResult({ files: [], table: [], steps: [], insights: {} });
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = originalText;
    }
  }
}

async function loadDataPreview() {
  const cards = document.getElementById("data-preview-cards");
  const labelGroups = document.getElementById("data-label-groups");
  const table = document.getElementById("data-preview-table");
  if (!cards || !labelGroups || !table) {
    return;
  }

  const manifestInput = document.querySelector('input[name="manifest"]');
  const labelsInput = document.querySelector('input[name="labels"]');
  const labelColumnInput = document.querySelector('input[name="label_column"]');
  const payload = {
    manifest: manifestInput ? manifestInput.value : appDefaults.manifest || "",
    labels: labelsInput ? labelsInput.value : appDefaults.labels || "",
    label_column: labelColumnInput ? labelColumnInput.value : appDefaults.label_column || "label",
  };

  if (!payload.manifest) {
    return;
  }

  const response = await fetch("/api/inspect/data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to load data preview");
  }

  renderCardsInto(cards, data.cards || []);
  labelGroups.innerHTML = "";
  (data.label_groups || []).forEach((row) => {
    const chip = document.createElement("div");
    chip.className = "group-chip";
    chip.innerHTML = `<strong>${row.label}</strong><span>${row.count}</span>`;
    labelGroups.appendChild(chip);
  });
  if ((data.label_groups || []).length === 0) {
    labelGroups.innerHTML = '<div class="soft-empty">No label grouping information available.</div>';
  }
  renderTable(table, data.rows || []);
}

async function loadFeaturePreview() {
  const cards = document.getElementById("feature-preview-cards");
  const groupList = document.getElementById("feature-group-list");
  const table = document.getElementById("feature-preview-table");
  if (!cards || !groupList || !table) {
    return;
  }

  const candidatePath = getFeatureCandidate();
  if (!candidatePath) {
    renderCardsInto(cards, []);
    groupList.innerHTML = '<div class="soft-empty">No feature file selected yet.</div>';
    renderTable(table, []);
    return;
  }

  const response = await fetch("/api/inspect/features", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features: candidatePath }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to load feature preview");
  }

  renderCardsInto(cards, data.cards || []);
  groupList.innerHTML = "";
  (data.group_rows || []).slice(0, 16).forEach((row) => {
    const item = document.createElement("div");
    item.className = "group-row";
    item.innerHTML = `
      <div>
        <strong>${row.group}</strong>
        <p>${row.examples || ""}</p>
      </div>
      <span>${row.count}</span>
    `;
    groupList.appendChild(item);
  });
  if ((data.group_rows || []).length === 0) {
    groupList.innerHTML = '<div class="soft-empty">No feature grouping information available.</div>';
  }
  renderTable(table, data.feature_rows || []);
}

function bindPreviewRefresh() {
  document.querySelectorAll("[data-refresh-preview]").forEach((button) => {
    button.addEventListener("click", async () => {
      const preview = button.getAttribute("data-refresh-preview");
      try {
        if (preview === "data") {
          await loadDataPreview();
        }
        if (preview === "features") {
          await loadFeaturePreview();
        }
      } catch (error) {
        setStatus("error", "Preview Failed", error.message || String(error));
      }
    });
  });
}

function bindInputTracking() {
  document.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => markFieldDirty(input));
  });
}

function bindMainForm() {
  const form = document.querySelector("form[data-endpoint]");
  if (!form) {
    return;
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitForm(form);
  });
}

window.addEventListener("error", (event) => {
  setStatus("error", "Frontend Error", event.message || "Unknown browser error");
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason && event.reason.message ? event.reason.message : String(event.reason);
  setStatus("error", "Async Error", reason);
});

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await fetchConfig();
    bindInputTracking();
    bindMainForm();
    bindPreviewRefresh();
    await loadDataPreview().catch(() => {});
    await loadFeaturePreview().catch(() => {});
  } catch (error) {
    setStatus("error", "Initialization Failed", error.message || String(error));
  }
});
