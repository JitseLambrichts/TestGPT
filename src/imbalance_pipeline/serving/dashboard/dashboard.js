"use strict";

(() => {
  const REFRESH_MS = 15_000;
  const HISTORY_MS = 6 * 60 * 60 * 1000;
  const FORECAST_ALLOWANCE_MS = 2 * 60 * 1000;
  const numberFormat = new Intl.NumberFormat("nl-BE", {
    signDisplay: "always",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  const probabilityFormat = new Intl.NumberFormat("nl-BE", {
    style: "percent",
    maximumFractionDigits: 0,
  });
  const timeFormat = new Intl.DateTimeFormat("nl-BE", {
    timeZone: "Europe/Brussels",
    hour: "2-digit",
    minute: "2-digit",
  });
  const dateTimeFormat = new Intl.DateTimeFormat("nl-BE", {
    timeZone: "Europe/Brussels",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

  function resultFor(row) {
    if (row.flip_actual === null || row.flip_actual === undefined) return "pending";
    return row.flip_actual === row.will_flip ? "correct" : "incorrect";
  }

  function formatMw(value, fallback = "Nog niet gekend") {
    if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
    return `${numberFormat.format(value)} MW`;
  }

  function formatDecision(value) {
    if (value === null || value === undefined) return "In afwachting";
    return value ? "Ja" : "Nee";
  }

  function formatState(value) {
    if (value === "positive") return "Positief";
    if (value === "negative") return "Negatief";
    return "Niet bepaald";
  }

  function isValidRow(row) {
    const requiredNumbers = [
      row.system_imbalance_mw,
      row.p10_mw,
      row.p90_mw,
      row.flip_probability,
    ];
    return Boolean(
      row &&
      typeof row.event_id === "string" &&
      typeof row.target_time === "string" &&
      !Number.isNaN(Date.parse(row.target_time)) &&
      requiredNumbers.every(Number.isFinite) &&
      row.p10_mw <= row.p90_mw &&
      row.flip_probability >= 0 &&
      row.flip_probability <= 1 &&
      typeof row.will_flip === "boolean" &&
      (row.realized_system_imbalance_mw === null ||
        Number.isFinite(row.realized_system_imbalance_mw)) &&
      (row.flip_actual === null || typeof row.flip_actual === "boolean")
    );
  }

  window.ImbalanceDashboard = Object.freeze({ resultFor, formatMw, formatDecision, formatState });

  if (typeof document === "undefined") return;

  const elements = {
    status: document.getElementById("live-status"),
    lastUpdate: document.getElementById("last-update"),
    retry: document.getElementById("retry-button"),
    model: document.getElementById("model-version"),
    expected: document.getElementById("expected-value"),
    expectedMeta: document.getElementById("expected-meta"),
    actual: document.getElementById("actual-value"),
    actualMeta: document.getElementById("actual-meta"),
    error: document.getElementById("error-value"),
    errorMeta: document.getElementById("error-meta"),
    flip: document.getElementById("flip-value"),
    flipMeta: document.getElementById("flip-meta"),
    state: document.getElementById("state-value"),
    quality: document.getElementById("quality-value"),
    canvas: document.getElementById("prediction-chart"),
    chartSummary: document.getElementById("chart-summary"),
    empty: document.getElementById("empty-state"),
    history: document.getElementById("history-body"),
    rowCount: document.getElementById("row-count"),
  };

  let chart = null;
  let timer = null;
  let inFlight = false;
  let hasSuccessfulLoad = false;

  function setStatus(kind, label) {
    elements.status.className = `status-pill is-${kind}`;
    elements.status.lastChild.textContent = label;
  }

  function scheduleRefresh() {
    window.clearTimeout(timer);
    if (document.visibilityState === "visible") {
      timer = window.setTimeout(loadDashboard, REFRESH_MS);
    }
  }

  async function loadDashboard() {
    if (inFlight || document.visibilityState === "hidden") return;
    inFlight = true;
    elements.retry.hidden = true;
    const now = Date.now();
    const params = new URLSearchParams({
      start: new Date(now - HISTORY_MS).toISOString(),
      end: new Date(now + FORECAST_ALLOWANCE_MS).toISOString(),
      limit: "500",
    });

    try {
      const response = await fetch(`/v1/dashboard?${params}`, {
        headers: { Accept: "application/json" },
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`Dashboard request failed with ${response.status}`);
      const payload = await response.json();
      if (!payload || !Array.isArray(payload.items) || !payload.items.every(isValidRow)) {
        throw new Error("Dashboard response has an invalid shape");
      }
      render(payload.items);
      hasSuccessfulLoad = true;
      setStatus("live", "Live");
      elements.lastUpdate.textContent = timeFormat.format(new Date());
    } catch (error) {
      console.error("Live dashboard refresh failed", error);
      setStatus(hasSuccessfulLoad ? "stale" : "error", hasSuccessfulLoad ? "Verouderd" : "Niet bereikbaar");
      elements.retry.hidden = false;
      if (!hasSuccessfulLoad) renderUnavailable();
    } finally {
      inFlight = false;
      scheduleRefresh();
    }
  }

  function render(rows) {
    if (rows.length === 0) {
      renderEmpty();
      return;
    }
    elements.empty.hidden = true;
    const latest = rows.at(-1);
    const latestRealized = [...rows]
      .reverse()
      .find((row) => row.realized_system_imbalance_mw !== null);

    elements.expected.textContent = formatMw(latest.system_imbalance_mw);
    elements.expectedMeta.textContent = `Voor ${dateTimeFormat.format(new Date(latest.target_time))}`;
    elements.model.textContent = latest.model_version || "Onbekend";
    elements.flip.textContent = probabilityFormat.format(latest.flip_probability);
    elements.flipMeta.textContent = `Flip voorspeld: ${formatDecision(latest.will_flip)}`;
    elements.state.textContent = formatState(latest.predicted_state);
    elements.quality.textContent = latest.prediction_quality || "Onbekend";
    elements.quality.classList.toggle("is-degraded", latest.prediction_quality === "degraded");

    if (latestRealized) {
      const error = latestRealized.system_imbalance_mw - latestRealized.realized_system_imbalance_mw;
      elements.actual.textContent = formatMw(latestRealized.realized_system_imbalance_mw);
      elements.actualMeta.textContent = `Voor ${dateTimeFormat.format(new Date(latestRealized.target_time))}`;
      elements.error.textContent = formatMw(error);
      elements.errorMeta.textContent = "Voorspeld min werkelijk";
    } else {
      elements.actual.textContent = "—";
      elements.actualMeta.textContent = "Nog niet gekend";
      elements.error.textContent = "—";
      elements.errorMeta.textContent = "In afwachting";
    }

    renderChart(rows);
    renderTable(rows);
  }

  function renderEmpty() {
    elements.empty.hidden = false;
    elements.model.textContent = "—";
    for (const element of [elements.expected, elements.actual, elements.error, elements.flip, elements.state]) {
      element.textContent = "—";
    }
    elements.quality.textContent = "—";
    elements.history.innerHTML = '<tr><td class="table-message" colspan="8">Nog geen live voorspellingen beschikbaar.</td></tr>';
    elements.rowCount.textContent = "0 resultaten";
    elements.chartSummary.textContent = "Nog geen live voorspellingen beschikbaar.";
    if (chart) {
      chart.data.datasets.forEach((dataset) => { dataset.data = []; });
      chart.update("none");
    }
  }

  function renderUnavailable() {
    elements.history.innerHTML = '<tr><td class="table-message" colspan="8">Live gegevens konden niet worden geladen. Probeer het opnieuw.</td></tr>';
  }

  const zeroLinePlugin = {
    id: "zeroLine",
    afterDraw(instance) {
      const y = instance.scales.y.getPixelForValue(0);
      const { left, right } = instance.chartArea;
      const context = instance.ctx;
      context.save();
      context.strokeStyle = "rgba(16, 35, 63, 0.52)";
      context.lineWidth = 1.4;
      context.setLineDash([5, 4]);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
      context.restore();
    },
  };

  function renderChart(rows) {
    const point = (row, key) => ({ x: Date.parse(row.target_time), y: row[key] });
    const datasets = [
      { label: "P90", data: rows.map((row) => point(row, "p90_mw")), borderColor: "transparent", pointRadius: 0, fill: false },
      { label: "P10–P90", data: rows.map((row) => point(row, "p10_mw")), borderColor: "transparent", backgroundColor: "rgba(20, 103, 210, 0.12)", pointRadius: 0, fill: "-1" },
      { label: "Voorspeld", data: rows.map((row) => point(row, "system_imbalance_mw")), borderColor: "#1467d2", backgroundColor: "#1467d2", borderWidth: 2.2, pointRadius: 0, pointHoverRadius: 4, tension: 0.16, fill: false },
      { label: "Werkelijk", data: rows.map((row) => point(row, "realized_system_imbalance_mw")), borderColor: "#087d78", backgroundColor: "#087d78", borderWidth: 2.2, pointRadius: 0, pointHoverRadius: 4, spanGaps: false, tension: 0.12, fill: false },
    ];

    if (!chart) {
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      chart = new Chart(elements.canvas, {
        type: "line",
        data: { datasets },
        plugins: [zeroLinePlugin],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          normalized: true,
          animation: reducedMotion ? false : { duration: 420 },
          interaction: { mode: "index", intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              filter: (context) => context.datasetIndex > 1,
              title: (contexts) => contexts.length ? dateTimeFormat.format(new Date(contexts[0].parsed.x)) : "",
              label: (context) => `${context.dataset.label}: ${formatMw(context.parsed.y)}`,
            },
          },
          scales: {
            x: {
              type: "linear",
              grid: { display: false },
              border: { display: false },
              ticks: { maxTicksLimit: 7, color: "#66788d", callback: (value) => timeFormat.format(new Date(value)) },
            },
            y: {
              grid: { color: "rgba(82, 100, 122, 0.12)" },
              border: { display: false },
              ticks: { color: "#66788d", callback: (value) => `${value} MW` },
            },
          },
        },
      });
    } else {
      chart.data.datasets.forEach((dataset, index) => { dataset.data = datasets[index].data; });
      chart.update("none");
    }

    const values = rows.flatMap((row) => [row.system_imbalance_mw, row.realized_system_imbalance_mw]).filter(Number.isFinite);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    elements.chartSummary.textContent = `Voorspelde en werkelijke netbalans van ${timeFormat.format(new Date(rows[0].target_time))} tot ${timeFormat.format(new Date(rows.at(-1).target_time))}. Waarden lopen van ${formatMw(minimum)} tot ${formatMw(maximum)}.`;
  }

  function renderTable(rows) {
    elements.rowCount.textContent = `${rows.length} ${rows.length === 1 ? "resultaat" : "resultaten"}`;
    elements.history.replaceChildren(...[...rows].reverse().map(tableRow));
  }

  function tableRow(row) {
    const tr = document.createElement("tr");
    const error = row.realized_system_imbalance_mw === null
      ? null
      : row.system_imbalance_mw - row.realized_system_imbalance_mw;
    const result = resultFor(row);
    const resultCopy = { correct: "Correct", incorrect: "Fout", pending: "In afwachting" }[result];
    const values = [
      dateTimeFormat.format(new Date(row.target_time)),
      formatMw(row.system_imbalance_mw),
      formatMw(row.realized_system_imbalance_mw, "—"),
      formatMw(error, "—"),
      probabilityFormat.format(row.flip_probability),
      formatDecision(row.will_flip),
      formatDecision(row.flip_actual),
    ];
    values.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if ([1, 2, 3, 4].includes(index)) td.className = "number";
      tr.append(td);
    });
    const resultCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `result-badge result-badge--${result}`;
    badge.textContent = resultCopy;
    resultCell.append(badge);
    tr.append(resultCell);
    return tr;
  }

  elements.retry.addEventListener("click", loadDashboard);
  document.addEventListener("visibilitychange", () => {
    window.clearTimeout(timer);
    if (document.visibilityState === "visible") loadDashboard();
  });
  loadDashboard();
})();
