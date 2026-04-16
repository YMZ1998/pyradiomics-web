const STORAGE_KEYS = {
  taskName: "pyrad.task.name",
  extractFeatures: "pyrad.extract.features",
  selectedFeatures: "pyrad.selected.features",
  trainedModel: "pyrad.trained.model",
  layoutPrefix: "pyrad.layout.",
};

const WORKFLOW_LABELS = {
  validate: "Validation",
  extract: "Extraction",
  select: "Feature Selection",
  train: "Model Training",
  predict: "Prediction",
  full: "Full Pipeline",
};

let appDefaults = {};

function normalizeTaskName(value) {
  const raw = String(value || "").trim();
  if (!raw) {
    return "default";
  }
  return raw.replace(/[^0-9A-Za-z._-]+/g, "-").replace(/^[.\-_]+|[.\-_]+$/g, "") || "default";
}

function getCurrentTaskName() {
  const taskInput = document.querySelector("[data-task-name]");
  if (taskInput && taskInput.value.trim()) {
    return normalizeTaskName(taskInput.value);
  }
  const stored = window.localStorage.getItem(STORAGE_KEYS.taskName);
  if (stored && stored.trim()) {
    return normalizeTaskName(stored);
  }
  return normalizeTaskName(appDefaults.task_name || "default");
}

function taskScopedKey(baseKey) {
  return `${baseKey}.${getCurrentTaskName()}`;
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

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
    if (Object.prototype.hasOwnProperty.call(payload, key)) {
      if (Array.isArray(payload[key])) {
        payload[key].push(value);
      } else {
        payload[key] = [payload[key], value];
      }
    } else {
      payload[key] = value;
    }
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
    window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.selectedFeatures)) ||
    window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.extractFeatures)) ||
    ""
  );
}

function getModelInput() {
  return document.querySelector('input[name="model"]');
}

function getModelSelect() {
  return document.getElementById("model-select");
}

function saveKnownPaths(endpoint, data) {
  const files = data.files || [];
  const extractKey = taskScopedKey(STORAGE_KEYS.extractFeatures);
  const selectedKey = taskScopedKey(STORAGE_KEYS.selectedFeatures);
  const modelKey = taskScopedKey(STORAGE_KEYS.trainedModel);
  if (endpoint === "/api/extract") {
    const featuresPath = findFilePath(files, "features.csv") || pathJoin(data.output_dir, "features.csv");
    if (featuresPath) {
      window.localStorage.setItem(extractKey, featuresPath);
    }
  }
  if (endpoint === "/api/select") {
    const selectedPath = findFilePath(files, "selected_features.csv") || pathJoin(data.output_dir, "selected_features.csv");
    if (selectedPath) {
      window.localStorage.setItem(selectedKey, selectedPath);
    }
  }
  if (endpoint === "/api/full") {
    const featuresPath = findFilePath(files, "features.csv");
    const selectedPath = findFilePath(files, "selected_features.csv");
    if (featuresPath) {
      window.localStorage.setItem(extractKey, featuresPath);
    }
    if (selectedPath) {
      window.localStorage.setItem(selectedKey, selectedPath);
    }
    if (data.best_model_path) {
      window.localStorage.setItem(modelKey, data.best_model_path);
    }
  }
  if (endpoint === "/api/train" && data.best_model_path) {
    window.localStorage.setItem(modelKey, data.best_model_path);
  }
  if (endpoint === "/api/predict") {
    const modelInput = getModelInput();
    if (modelInput && modelInput.value) {
      window.localStorage.setItem(modelKey, modelInput.value);
    }
  }
}

async function fetchConfig() {
  const task = getCurrentTaskName();
  const response = await fetch(`/api/config?task=${encodeURIComponent(task)}`);
  if (!response.ok) {
    throw new Error(`Config request failed: ${response.status}`);
  }
  const payload = await response.json();
  appDefaults = payload.defaults || {};
  window.localStorage.setItem(STORAGE_KEYS.taskName, normalizeTaskName(appDefaults.task_name || task));

  document.querySelectorAll("[data-default]").forEach((input) => {
    const key = input.getAttribute("data-default");
    setInputValue(input, appDefaults[key], true);
    delete input.dataset.dirty;
  });

  document.querySelectorAll("[data-task-name]").forEach((input) => {
    setInputValue(input, appDefaults.task_name || task, true);
    delete input.dataset.dirty;
  });

  document.querySelectorAll("[data-store-key]").forEach((input) => {
    const storageKey = input.getAttribute("data-store-key");
    if (storageKey === "extract-features") {
      setInputValue(input, window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.extractFeatures)), false);
    }
    if (storageKey === "selected-features") {
      setInputValue(input, window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.selectedFeatures)), false);
    }
    if (storageKey === "trained-model") {
      setInputValue(input, window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.trainedModel)), false);
    }
  });

  document.querySelectorAll('input[type="checkbox"][data-default-list]').forEach((input) => {
    const key = input.getAttribute("data-default-list");
    const values = Array.isArray(appDefaults[key]) ? appDefaults[key] : [];
    input.checked = values.includes(input.value);
    delete input.dataset.dirty;
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

function clearStatus() {
  const box = document.getElementById("status-box");
  if (!box) {
    return;
  }
  box.classList.add("hidden");
  box.innerHTML = "";
}

function setProgressVisible(visible) {
  const box = document.getElementById("progress-box");
  if (!box) {
    return;
  }
  box.classList.toggle("hidden", !visible);
}

function updateProgress(title, percent, detail) {
  const titleNode = document.getElementById("progress-title");
  const percentNode = document.getElementById("progress-percent");
  const bar = document.getElementById("progress-bar");
  const detailNode = document.getElementById("progress-detail");
  if (!titleNode || !percentNode || !bar || !detailNode) {
    return;
  }

  const normalized = clamp(Number(percent) || 0, 0, 100);
  titleNode.textContent = title;
  percentNode.textContent = `${Math.round(normalized)}%`;
  bar.style.width = `${normalized}%`;
  detailNode.textContent = detail || "Running...";
}

function workflowTitle(workflow) {
  return WORKFLOW_LABELS[workflow] || workflow;
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
  const visualOnly = Boolean(container.closest("[data-visual-results='true']"));
  container.innerHTML = "";
  if (visualOnly || !rows || rows.length === 0) {
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

function renderImages(files) {
  const container = document.getElementById("result-images");
  if (!container) {
    return;
  }
  container.innerHTML = "";

  const imageFiles = (files || [])
    .filter((file) => /\.(png|jpg|jpeg|svg)$/i.test(file.name || ""))
    .sort((a, b) => {
      const score = (name) => {
        if (/roc/i.test(name)) return 0;
        if (/confusion/i.test(name)) return 1;
        return 2;
      };
      return score(a.name || "") - score(b.name || "");
    });
  if (imageFiles.length === 0) {
    return;
  }

  imageFiles.forEach((file) => {
    const panel = document.createElement("div");
    panel.className = "image-panel";

    const title = document.createElement("strong");
    title.textContent = humanizeImageName(file.name || file.display_path || "Image");

    const caption = document.createElement("span");
    caption.textContent = file.display_path || file.name || "";

    const image = document.createElement("img");
    image.src = `/download?path=${encodeURIComponent(file.path)}`;
    image.alt = file.name || "result image";

    panel.appendChild(title);
    panel.appendChild(caption);
    panel.appendChild(image);
    container.appendChild(panel);
  });
}

function humanizeImageName(name) {
  const cleaned = String(name)
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned
    .split(" ")
    .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
    .join(" ");
}

function renderSelectedFeatures(features) {
  const container = document.getElementById("selected-feature-list");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  if (!features || features.length === 0) {
    container.innerHTML = '<span class="soft-empty">Run feature selection to view the final feature set.</span>';
    return;
  }
  features.forEach((featureName) => {
    const chip = document.createElement("span");
    chip.className = "group-feature-chip";
    chip.textContent = featureName;
    container.appendChild(chip);
  });
}

function renderModelOptions(models) {
  const select = getModelSelect();
  const modelInput = getModelInput();
  if (!select) {
    return;
  }

  const currentValue = modelInput && modelInput.value ? modelInput.value : window.localStorage.getItem(STORAGE_KEYS.trainedModel) || "";
  const scopedCurrentValue = currentValue || window.localStorage.getItem(taskScopedKey(STORAGE_KEYS.trainedModel)) || "";
  select.innerHTML = "";

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = models && models.length > 0 ? "Select a trained model" : "No trained models found";
  select.appendChild(placeholder);

  (models || []).forEach((item) => {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.label} - ${item.display_path || item.path}`;
    if (item.path === scopedCurrentValue) {
      option.selected = true;
    }
    select.appendChild(option);
  });

  if (!select.value && scopedCurrentValue) {
    const custom = document.createElement("option");
    custom.value = scopedCurrentValue;
    custom.textContent = `Current: ${scopedCurrentValue}`;
    custom.selected = true;
    select.appendChild(custom);
  }
}

function renderMainResult(data) {
  renderCardsInto(document.getElementById("result-cards"), getNested(data, ["insights", "cards"], []));
  renderSteps(document.getElementById("result-steps"), getNested(data, ["insights", "stage_view", "items"], data.steps || []));
  renderChart(getNested(data, ["insights", "bar_chart"], null));
  renderImages(data.files || []);
  renderTable(document.getElementById("result-table"), data.table || []);
  renderFiles(data.files || []);
  renderSelectedFeatures(data.selected_features || []);
}

function endpointToWorkflow(endpoint) {
  return String(endpoint || "").split("/").filter(Boolean).pop() || "";
}

async function submitWorkflow(endpoint, payload) {
  const workflow = endpointToWorkflow(endpoint);
  const asyncEndpoint = `/api/v1/workflows/${workflow}`;
  const response = await fetch(asyncEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, run_async: true }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }

  if (data.result) {
    updateProgress(workflowTitle(workflow), 100, "Completed");
    return data.result;
  }

  if (!data.job || !data.job.job_id) {
    throw new Error("Job information missing from server response.");
  }

  updateProgress(
    workflowTitle(workflow),
    Number(data.job.progress) || 0,
    data.job.detail || "Queued",
  );
  return pollJob(data.job.job_id, workflow);
}

async function pollJob(jobId, workflow) {
  for (;;) {
    await delay(700);
    const response = await fetch(`/api/v1/jobs/${jobId}`);
    const data = await response.json();
    if (!response.ok || !data.ok || !data.job) {
      throw new Error(data.error || `Job polling failed: ${response.status}`);
    }

    const job = data.job;
    updateProgress(
      workflowTitle(workflow),
      Number(job.progress) || 0,
      job.detail || job.status || "Running",
    );

    if (job.status === "completed") {
      return job.result || {};
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.detail || "Workflow execution failed");
    }
  }
}

async function submitForm(form) {
  const endpoint = form.getAttribute("data-endpoint");
  const button = form.querySelector('button[type="submit"]');
  const originalText = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "Running...";
  }

  clearStatus();
  setProgressVisible(true);
  updateProgress(workflowTitle(endpointToWorkflow(endpoint)), 0, "Submitting job");

  try {
    const data = await submitWorkflow(endpoint, collectFormPayload(form));
    saveKnownPaths(endpoint, data);
    updateProgress(workflowTitle(endpointToWorkflow(endpoint)), 100, data.summary || "Completed");
    setStatus("success", data.title, data.summary);
    renderMainResult(data);
    await loadDataPreview().catch(() => {});
    await loadFeaturePreview().catch(() => {});
  } catch (error) {
    setStatus("error", "Execution failed", error.message || String(error));
    updateProgress(workflowTitle(endpointToWorkflow(endpoint)), 100, error.message || String(error));
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
    const label = document.createElement("strong");
    label.textContent = `Label ${row.label}`;

    const count = document.createElement("span");
    count.textContent = `${row.count} cases`;

    chip.appendChild(label);
    chip.appendChild(count);
    labelGroups.appendChild(chip);
  });
  if ((data.label_groups || []).length === 0) {
    labelGroups.innerHTML = '<div class="soft-empty">No label grouping information available.</div>';
  }
  renderTable(table, data.rows || []);
}

function renderFeatureGroups(container, groups) {
  container.innerHTML = "";
  (groups || []).forEach((row) => {
    const item = document.createElement("details");
    item.className = "group-row group-accordion";

    const summary = document.createElement("summary");
    const left = document.createElement("div");
    left.className = "group-summary";

    const title = document.createElement("strong");
    title.textContent = row.group;
    const desc = document.createElement("p");
    desc.textContent = row.examples || "";
    left.appendChild(title);
    left.appendChild(desc);

    const count = document.createElement("span");
    count.className = "group-count";
    count.textContent = `${row.count}`;

    summary.appendChild(left);
    summary.appendChild(count);
    item.appendChild(summary);

    const featureList = document.createElement("div");
    featureList.className = "group-feature-list";
    (row.features || []).forEach((featureName) => {
      const chip = document.createElement("span");
      chip.className = "group-feature-chip";
      chip.textContent = featureName;
      featureList.appendChild(chip);
    });
    item.appendChild(featureList);
    container.appendChild(item);
  });
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
  if ((data.group_rows || []).length === 0) {
    groupList.innerHTML = '<div class="soft-empty">No feature grouping information available.</div>';
  } else {
    renderFeatureGroups(groupList, data.group_rows || []);
  }
  renderTable(table, data.feature_rows || []);
}

async function loadModelOptions() {
  const select = getModelSelect();
  if (!select) {
    return;
  }

  const response = await fetch("/api/inspect/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_name: getCurrentTaskName() }),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Failed to load trained models");
  }
  renderModelOptions(data.models || []);
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
        if (preview === "models") {
          await loadModelOptions();
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
    input.addEventListener("change", () => markFieldDirty(input));
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

function bindModelSelect() {
  const select = getModelSelect();
  const modelInput = getModelInput();
  if (!select || !modelInput) {
    return;
  }
  select.addEventListener("change", () => {
    if (select.value) {
      modelInput.value = select.value;
      markFieldDirty(modelInput);
      window.localStorage.setItem(taskScopedKey(STORAGE_KEYS.trainedModel), select.value);
    }
  });
}

function bindTaskInputs() {
  document.querySelectorAll("[data-task-name]").forEach((input) => {
    input.addEventListener("change", async () => {
      const normalized = normalizeTaskName(input.value);
      input.value = normalized;
      window.localStorage.setItem(STORAGE_KEYS.taskName, normalized);
      clearStatus();
      await fetchConfig();
      await loadDataPreview().catch(() => {});
      await loadFeaturePreview().catch(() => {});
      await loadModelOptions().catch(() => {});
    });
  });
}

function setupResizableSplit(container) {
  if (!container || container.dataset.splitReady === "true") {
    return;
  }

  const panes = Array.from(container.children).filter((node) => node.nodeType === Node.ELEMENT_NODE);
  if (panes.length !== 2) {
    return;
  }

  const direction = container.dataset.splitDirection || "horizontal";
  const minRatio = Number(container.dataset.splitMin) || 0.2;
  const maxRatio = Number(container.dataset.splitMax) || 0.8;
  const defaultRatio = clamp(Number(container.dataset.splitDefault) || 0.5, minRatio, maxRatio);
  const storageKey = `${STORAGE_KEYS.layoutPrefix}${container.dataset.splitId || container.className}`;
  const persistedRatio = clamp(Number(window.localStorage.getItem(storageKey)) || defaultRatio, minRatio, maxRatio);

  const splitter = document.createElement("div");
  splitter.className = `splitter splitter-${direction}`;
  splitter.setAttribute("role", "separator");
  splitter.setAttribute("tabindex", "0");

  container.insertBefore(splitter, panes[1]);
  container.classList.add("is-resizable");
  container.dataset.splitReady = "true";

  const applyRatio = (ratio) => {
    const safeRatio = clamp(ratio, minRatio, maxRatio);
    container.style.setProperty("--split-ratio", String(safeRatio));
    container.style.setProperty("--split-inverse", String(1 - safeRatio));
    window.localStorage.setItem(storageKey, String(safeRatio));
  };

  applyRatio(persistedRatio);

  splitter.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= 1180) {
      return;
    }

    const rect = container.getBoundingClientRect();
    const pointerId = event.pointerId;
    splitter.setPointerCapture(pointerId);

    const handleMove = (moveEvent) => {
      const nextRatio = direction === "horizontal"
        ? (moveEvent.clientX - rect.left) / rect.width
        : (moveEvent.clientY - rect.top) / rect.height;
      applyRatio(nextRatio);
    };

    const handleUp = () => {
      splitter.removeEventListener("pointermove", handleMove);
      splitter.removeEventListener("pointerup", handleUp);
      splitter.removeEventListener("pointercancel", handleUp);
      splitter.releasePointerCapture(pointerId);
    };

    splitter.addEventListener("pointermove", handleMove);
    splitter.addEventListener("pointerup", handleUp);
    splitter.addEventListener("pointercancel", handleUp);
    event.preventDefault();
  });
}

function initializeResizableSplits() {
  document.querySelectorAll("[data-split-id]").forEach((container) => setupResizableSplit(container));
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
    initializeResizableSplits();
    await fetchConfig();
    bindInputTracking();
    bindMainForm();
    bindModelSelect();
    bindTaskInputs();
    bindPreviewRefresh();
    setProgressVisible(false);
    await loadDataPreview().catch(() => {});
    await loadFeaturePreview().catch(() => {});
    await loadModelOptions().catch(() => {});
  } catch (error) {
    setStatus("error", "Initialization Failed", error.message || String(error));
  }
});
