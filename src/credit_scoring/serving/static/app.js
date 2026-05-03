"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (value, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
const pct = (value, digits = 1) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const number = (value) => Number(value).toLocaleString("en-US");
const names = {
  gradient_boosting: "Gradient boosting",
  logistic_regression: "Logistic regression",
};
let report = null;
async function request(url, options) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(20000),
  });
  const body = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail
          .map((item) => `${item.loc.slice(1).join(" ")}: ${item.msg}`)
          .join("; ")
      : body.detail;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return body;
}
function render(data) {
  report = data;
  const m = data.model;
  $("status").textContent = m ? "Model loaded" : "No model loaded";
  $("status").dataset.state = m ? "ready" : "unavailable";
  document.querySelector(".model-strip").dataset.ready = String(Boolean(m));
  $("algorithm").textContent = m
    ? names[m.algorithm] || m.algorithm
    : "No registered model";
  $("version").textContent = m?.version || "—";
  $("trained").textContent = m
    ? new Date(m.trained_at).toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      })
    : "—";
  $("auc").textContent = fmt(m?.metrics?.roc_auc);
  $("pr").textContent = fmt(m?.metrics?.pr_auc);
  $("brier").textContent = fmt(m?.metrics?.brier);
  $("threshold").textContent = pct(m?.threshold, 0);
  $("auc-scale").style.width =
    `${Math.max(0, Math.min(100, ((m?.metrics?.roc_auc || 0.5) - 0.5) * 200))}%`;
  $("cost-label").textContent = m?.cost
    ? `${m.cost.false_negative}:${m.cost.false_positive} missed-default cost`
    : "Stored model policy";
  $("score-button").disabled = !m;
  const op = m?.operating_point;
  if (op && ["tn", "fp", "fn", "tp"].every((key) => Number.isFinite(op[key]))) {
    $("sample-count").textContent =
      `${number(op.tn + op.fp + op.fn + op.tp)} RECORDS`;
    $("outcome-description").textContent =
      `Actual repayment outcomes against decisions at the ${pct(m.threshold, 0)} threshold.`;
    $("outcomes").innerHTML = [
      [op.tn, "Repaid · approved", "✓"],
      [op.fp, "Repaid · declined", "↗"],
      [op.fn, "Defaulted · approved", "↘"],
      [op.tp, "Defaulted · declined", "✓"],
    ]
      .map(
        ([count, label, mark]) =>
          `<div class="outcome-cell"><span class="count">${number(count)}</span><span class="cell-label">${label}</span><span class="cell-corner" aria-hidden="true">${mark}</span></div>`,
      )
      .join("");
    $("recall").textContent = `${pct(op.tp / (op.tp + op.fn))} default recall`;
  } else {
    $("sample-count").textContent = "NOT RECORDED";
    $("outcome-description").textContent =
      "This model has no saved classification counts.";
    $("outcomes").innerHTML =
      '<p class="empty">Retrain with validation metadata to inspect these outcomes.</p>';
    $("recall").textContent = "Recall unavailable";
  }
  const approved = data.predictions.approve || 0,
    declined = data.predictions.decline || 0;
  $("total").textContent = number(approved + declined);
  $("approved").textContent = number(approved);
  $("declined").textContent = number(declined);
  $("approval-track").style.width =
    `${approved + declined ? (approved / (approved + declined)) * 100 : 0}%`;
  if (data.drift) {
    const flagged = Object.entries(data.drift).filter(
      ([, value]) => value > data.drift_threshold,
    );
    $("drift-title").textContent = flagged.length
      ? `${flagged.length} features above threshold`
      : "Within the PSI threshold";
    $("drift-copy").textContent = flagged.length
      ? `Latest batch: ${flagged.map(([key]) => key).join(", ")}. Threshold: ${data.drift_threshold}.`
      : `All observed features have PSI at or below ${data.drift_threshold} in the latest batch.`;
  } else {
    $("drift-title").textContent = "No batch observed";
    $("drift-copy").textContent =
      "Score a batch through the API to compare its features with the training data.";
  }
  $("cv").innerHTML = m?.cv
    ? Object.entries(m.cv)
        .map(
          ([name, metrics]) =>
            `<div class="cv-row"><span>${esc(names[name] || name)}</span><div class="cv-track" role="img" aria-label="ROC-AUC ${fmt(metrics.cv_roc_auc)}"><i style="width:${Math.max(0, Math.min(100, metrics.cv_roc_auc * 100))}%"></i></div><span class="mono">${fmt(metrics.cv_roc_auc)}</span></div>`,
        )
        .join("") +
      '<p class="small-copy">Mean ROC-AUC across training folds · scale 0–1</p>'
    : '<p class="small-copy">No cross-validation scores recorded for this model.</p>';
  $("export").disabled = false;
}
async function refresh() {
  $("refresh").disabled = true;
  try {
    render(await request("/dashboard/data"));
    $("error").hidden = true;
  } catch (error) {
    $("status").textContent = "Connection unavailable";
    $("status").dataset.state = "unavailable";
    $("error").textContent =
      `Could not refresh the model data. ${error.message} Select Refresh to try again.`;
    $("error").hidden = false;
  } finally {
    $("refresh").disabled = false;
  }
}
$("refresh").addEventListener("click", refresh);
$("export").addEventListener("click", () => {
  if (!report) return;
  const url = URL.createObjectURL(
    new Blob(
      [
        JSON.stringify(
          { ...report, exported_at: new Date().toISOString() },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    ),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `caliper-risk-${report.model?.version || "unavailable"}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const fields = [
  ["sex", "Sex code (1 male, 2 female)", 2, 1, 2],
  ["education", "Education code (0–6)", 2, 0, 6],
  ["marriage", "Marital status code (0–3)", 1, 0, 3],
  ["pay_2", "Repayment status · month 2", 2, -2, 9],
  ["pay_3", "Repayment status · month 3", -1, -2, 9],
  ["pay_4", "Repayment status · month 4", -1, -2, 9],
  ["pay_5", "Repayment status · month 5", -2, -2, 9],
  ["pay_6", "Repayment status · month 6", -2, -2, 9],
  ...[2, 3, 4, 5, 6].map((n) => [
    `bill_amt${n}`,
    `Bill · month ${n} (NT$)`,
    n === 2 ? 3102 : n === 3 ? 689 : 0,
    null,
    null,
  ]),
  ...[1, 2, 3, 4, 5, 6].map((n) => [
    `pay_amt${n}`,
    `Payment · month ${n} (NT$)`,
    n === 2 ? 689 : 0,
    0,
    null,
  ]),
];
$("extra-fields").innerHTML = fields
  .map(
    ([name, label, value, min, max]) =>
      `<label>${esc(label)}<input name="${name}" type="number" value="${value}" ${min !== null ? `min="${min}"` : ""} ${max !== null ? `max="${max}"` : ""} required></label>`,
  )
  .join("");
$("score-form").addEventListener(
  "invalid",
  (event) => {
    const details = event.target.closest("details");
    if (details) details.open = true;
  },
  true,
);
$("score-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("score-button");
  button.disabled = true;
  button.textContent = "Calculating…";
  const payload = Object.fromEntries(
    [...new FormData(event.currentTarget)].map(([key, value]) => [
      key,
      Number(value),
    ]),
  );
  try {
    const r = await request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("score-result").innerHTML =
      `<span class="eyebrow">DEFAULT PROBABILITY</span><div class="result-placeholder">${fmt(r.default_probability * 100, 1)}<span>%</span></div><span class="decision ${esc(r.decision)}">${r.decision === "approve" ? "Approve" : "Decline"}</span><p>The estimate is ${r.default_probability > r.threshold ? "above" : "at or below"} the ${pct(r.threshold)} decision threshold.</p><div class="result-note">Model ${esc(r.model_version)} · Research estimate</div>`;
    await refresh();
  } catch (error) {
    $("score-result").innerHTML =
      `<span class="eyebrow">ASSESSMENT UNAVAILABLE</span><p style="margin-top:24px">${esc(error.message)}</p><div class="result-note">Check the application values and try again.</div>`;
  } finally {
    button.disabled = !report?.model;
    button.innerHTML =
      'Calculate default probability <span aria-hidden="true">↗</span>';
  }
});
document
  .querySelectorAll(".nav-link")
  .forEach((link) =>
    link.addEventListener("click", () =>
      document
        .querySelectorAll(".nav-link")
        .forEach((item) => item.classList.toggle("active", item === link)),
    ),
  );
refresh();
