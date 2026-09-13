(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  let currentInstrument = localStorage.getItem("predictor.instrument") || "NIFTY50";
  let sinceLines = 0;
  let pollTimer = null;
  let statusTimer = null;
  let selectedRegSymbol = null;

  function fmtRelative(iso) {
    if (!iso) return "never";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return iso;
    const diffMs = Date.now() - then;
    const mins = Math.round(diffMs / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins} min ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs} hr ago`;
    const days = Math.round(hrs / 24);
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }

  function pct(x, digits = 0) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    return `${(x * 100).toFixed(digits)}%`;
  }

  function statRow(k, v) {
    return `<div class="stat-row"><span class="k">${k}</span><span class="v">${v}</span></div>`;
  }

  function verdictBox(text, tone) {
    return `<div class="verdict ${tone}">${text}</div>`;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ---- stock picker ---------------------------------------------------------

  async function loadInstruments(preserveSelection = true) {
    const res = await fetch("/api/instruments");
    const data = await res.json();
    const list = data.instruments || [];
    const select = $("stock-select");
    select.innerHTML = list
      .map((i) => `<option value="${i.key}">${escapeHtml(i.display_name)} (${i.key})</option>`)
      .join("") + `<option value="__register__">+ Register a new stock…</option>`;

    if (preserveSelection && list.some((i) => i.key === currentInstrument)) {
      select.value = currentInstrument;
    } else if (list.length) {
      currentInstrument = list[0].key;
      select.value = currentInstrument;
    }
    return list;
  }

  function switchInstrument(key) {
    currentInstrument = key;
    localStorage.setItem("predictor.instrument", key);
    stopRunPoll();
    sinceLines = 0;
    $("console").textContent = "";
    $("console").hidden = true;
    fetchStatus();
  }

  $("stock-select").addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "__register__") {
      e.target.value = currentInstrument;
      openRegisterDialog();
    } else {
      switchInstrument(val);
    }
  });

  // ---- registration dialog ---------------------------------------------------

  async function openRegisterDialog() {
    $("reg-search").value = "";
    $("reg-results").innerHTML = "";
    $("reg-selected").hidden = true;
    $("reg-log").hidden = true;
    $("reg-log").textContent = "";
    selectedRegSymbol = null;

    const benchSelect = $("reg-benchmark");
    const instruments = await loadInstruments();
    benchSelect.innerHTML = instruments
      .map((i) => `<option value="${i.key}"${i.key === "NIFTY50" ? " selected" : ""}>${escapeHtml(i.display_name)}</option>`)
      .join("");

    $("register-dialog").showModal();
    $("reg-search").focus();
  }

  let searchDebounce = null;
  $("reg-search").addEventListener("input", (e) => {
    clearTimeout(searchDebounce);
    const q = e.target.value.trim();
    if (!q) { $("reg-results").innerHTML = ""; return; }
    searchDebounce = setTimeout(async () => {
      const res = await fetch(`/api/instruments/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      const results = data.results || [];
      $("reg-results").innerHTML = results.length
        ? results.map((r) => `<li data-symbol="${escapeHtml(r.symbol)}" data-name="${escapeHtml(r.name)}">
             <strong>${escapeHtml(r.symbol)}</strong> — ${escapeHtml(r.name)}</li>`).join("")
        : `<li class="muted reg-no-results">No match. You can also type the exact NSE symbol and press Enter.</li>`;
    }, 250);
  });

  $("reg-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !$("reg-results").children.length) {
      pickRegCandidate(e.target.value.trim().toUpperCase(), "");
    }
  });

  $("reg-results").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-symbol]");
    if (li) pickRegCandidate(li.dataset.symbol, li.dataset.name);
  });

  function pickRegCandidate(symbol, name) {
    if (!symbol) return;
    selectedRegSymbol = symbol;
    $("reg-selected-name").textContent = name || symbol;
    $("reg-selected-symbol").textContent = name ? `${symbol}.NS` : `${symbol}.NS (typed manually — will be validated)`;
    $("reg-selected").hidden = false;
  }

  $("reg-cancel").addEventListener("click", () => $("register-dialog").close());
  $("reg-close").addEventListener("click", () => $("register-dialog").close());

  $("reg-confirm").addEventListener("click", async () => {
    if (!selectedRegSymbol) return;
    const btn = $("reg-confirm");
    btn.disabled = true;
    btn.textContent = "Registering…";
    const logEl = $("reg-log");
    logEl.hidden = false;
    logEl.textContent = "validating ticker on yfinance…";

    try {
      const res = await fetch("/api/instruments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol: selectedRegSymbol,
          name: $("reg-selected-name").textContent,
          benchmark_key: $("reg-benchmark").value,
          push: $("reg-push").checked,
        }),
      });
      const data = await res.json();
      logEl.textContent = (data.log || []).join("\n") || data.error || "done";
      if (data.ok) {
        await loadInstruments(false);
        switchInstrument(data.entry.key);
        $("stock-select").value = data.entry.key;
        setTimeout(() => $("register-dialog").close(), 1500);
      }
    } catch (e) {
      logEl.textContent = "request failed: " + e;
    } finally {
      btn.disabled = false;
      btn.textContent = "Register";
    }
  });

  // ---- top bar ---------------------------------------------------------

  function renderTopbar(data) {
    $("clock").textContent = new Date(data.server_time).toLocaleTimeString();
    $("stock-heading").textContent = data.instrument ? data.instrument.display_name : "Predictor";
    document.title = `${data.instrument ? data.instrument.display_name : "Predictor"} — Control Panel`;
    const badge = $("market-badge");
    if (data.market_open) {
      badge.textContent = "🟢 NSE market is open";
      badge.className = "badge open";
    } else {
      badge.textContent = "⚪ NSE market is closed";
      badge.className = "badge closed";
    }
    $("heartbeat-dot").className = "dot ok";
  }

  // ---- cards -------------------------------------------------------------

  function renderCollector(c) {
    const el = document.querySelector("#card-collector .card-body");
    if (!c || c.error || !c.days) {
      el.innerHTML = `<p class="muted">No cloud-collected data yet for this stock ${
        c && c.ticker ? `(<code>${escapeHtml(c.ticker)}</code>)` : ""
      }. It's added to the next collector run automatically.</p>`;
      return;
    }
    const freshMins = c.last_ingested_at
      ? Math.round((Date.now() - new Date(c.last_ingested_at).getTime()) / 60000)
      : null;
    const tone = freshMins === null ? "neutral" : freshMins < 30 ? "good" : freshMins < 24 * 60 ? "neutral" : "warn";
    const freshText = freshMins === null
      ? "No timestamped data yet."
      : `Latest data point: ${fmtRelative(c.last_ingested_at)}.`;

    let bars = "";
    if (c.per_day && c.per_day.length) {
      const max = Math.max(...c.per_day.map((d) => d.rows), 1);
      bars = `<div class="bars">${c.per_day
        .map((d) => `<div class="bar" style="height:${Math.max(6, (d.rows / max) * 100)}%" title="${d.date}: ${d.rows} rows"></div>`)
        .join("")}</div>
        <div class="bars-caption"><span>${c.per_day[0].date}</span><span>${c.per_day[c.per_day.length - 1].date}</span></div>`;
    }

    el.innerHTML = `
      ${verdictBox(freshText, tone)}
      ${statRow("Trading days collected", c.days ?? 0)}
      ${statRow("Total price snapshots", (c.total_rows ?? 0).toLocaleString())}
      ${statRow("Date range", c.first_date && c.last_date ? `${c.first_date} → ${c.last_date}` : "—")}
      <p class="muted" style="margin:0.5rem 0 0">Rows collected per day (last 3 weeks):</p>
      ${bars || '<p class="muted">not enough history yet</p>'}
    `;
  }

  function renderDataset(d) {
    const el = document.querySelector("#card-dataset .card-body");
    if (!d) {
      el.innerHTML = `<p class="muted">No dataset built yet — run the predictor to build one.</p>`;
      return;
    }
    const counts = d.label_counts || {};
    el.innerHTML = `
      ${statRow("Rows (entry points)", (d.rows ?? 0).toLocaleString())}
      ${statRow("Trading days", d.days ?? 0)}
      ${statRow("Date range", d.date_range ? `${d.date_range[0].slice(0, 10)} → ${d.date_range[1].slice(0, 10)}` : "—")}
      ${statRow("Features calculated", d.feature_columns ? d.feature_columns.length : "—")}
      ${statRow("Outcomes so far", `${counts["1"] ?? 0} up / ${counts["-1"] ?? 0} down / ${counts["0"] ?? 0} no move`)}
      <p class="muted" style="margin:0.5rem 0 0">
        More trading days → a smarter model. This grows automatically every trading day.
      </p>
    `;
  }

  function renderModel(m) {
    const el = document.querySelector("#card-model .card-body");
    if (!m) {
      el.innerHTML = `<p class="muted">No model trained yet.</p>`;
      return;
    }
    const hc = (m.metrics && m.metrics.high_confidence) || {};
    const tone = m.fire_threshold == null ? "neutral" : !hc.n_fired ? "neutral" : (hc.precision ?? 0) >= 0.55 ? "good" : "warn";
    el.innerHTML = `
      ${verdictBox(m.plain_verdict, tone)}
      ${statRow("Last trained", fmtRelative(m.trained_at))}
      ${statRow("Trained through", m.train_data_end ?? "—")}
      ${statRow("Confident calls it would fire", `${hc.n_fired ?? 0} (~${hc.fires_per_day ? hc.fires_per_day.toFixed(2) : 0}/day)`)}
      <details style="margin-top:0.5rem">
        <summary class="muted">Technical detail</summary>
        ${statRow("Directional precision (all calls)", pct((m.metrics && m.metrics.directional && m.metrics.directional.precision)))}
        ${statRow("High-confidence precision", pct(hc.precision))}
        ${statRow("Fire threshold", m.fire_threshold != null ? m.fire_threshold.toFixed(3) : "not set")}
      </details>
    `;
  }

  function renderBacktest(b) {
    const el = document.querySelector("#card-backtest .card-body");
    if (!b) {
      el.innerHTML = `<p class="muted">No backtest yet.</p>`;
      return;
    }
    const h = b.headline || {};
    const tone = !h.n_trades ? "neutral" : (h.win_rate ?? 0) >= 0.5 ? "good" : "warn";
    el.innerHTML = `
      ${verdictBox(b.plain_verdict, tone)}
      ${statRow("Simulated trades", h.n_trades ?? 0)}
      ${statRow("Trades per day", h.trades_per_day ? h.trades_per_day.toFixed(2) : "0")}
      ${statRow("Win rate", pct(h.win_rate))}
      ${statRow("Total simulated return", h.total_net_ret !== undefined ? pct(h.total_net_ret, 2) : "—")}
    `;
  }

  function renderPaper(p) {
    const el = document.querySelector("#card-paper .card-body");
    if (!p || p.status) {
      el.innerHTML = `<p class="muted">${(p && p.status) || "No paper-trading log yet."}</p>`;
      return;
    }
    const fired = p.fired || {};
    const tone = !fired.n ? "neutral" : (fired.precision ?? 0) >= 0.5 ? "good" : "warn";
    const verdict = !fired.n
      ? "No confident real-world calls logged yet."
      : `Fired ${fired.n} real-world call${fired.n === 1 ? "" : "s"}, right ${pct(fired.precision)} of the time.`;
    el.innerHTML = `
      ${verdictBox(verdict, tone)}
      ${statRow("Out-of-sample entries logged", p.rows_out_of_sample ?? 0)}
      ${statRow("Date range", p.oos_date_range ? `${p.oos_date_range[0]} → ${p.oos_date_range[1]}` : "—")}
      ${statRow("Confident calls fired", fired.n ?? 0)}
      ${statRow("Fired-call accuracy", pct(fired.precision))}
    `;
  }

  function renderSchedule(s) {
    const el = document.querySelector("#card-schedule .card-body");
    if (!s || !s.scheduled) {
      el.innerHTML = `
        ${verdictBox("Not scheduled — nobody's told Windows to run this automatically yet.", "neutral")}
        <p class="muted">Use the <strong>Run Predictor Now</strong> button above whenever you want fresh results,
        or set up a Windows Scheduled Task named <code>PredictorDaily-${escapeHtml(currentInstrument)}</code>
        (see the README) for a daily automatic run of this stock.</p>
      `;
      return;
    }
    el.innerHTML = `
      ${verdictBox(`An automatic daily run is scheduled (task "${escapeHtml(s.task_name)}").`, "good")}
      ${statRow("Next run", s.next_run ?? "—")}
      ${statRow("Last run", s.last_run ?? "—")}
      ${statRow("Last result", s.last_result ?? "—")}
    `;
  }

  function renderLogTail(lines) {
    const el = $("log-tail");
    el.textContent = (lines && lines.length) ? lines.join("\n") : "(no log file yet)";
  }

  // ---- run console ---------------------------------------------------------

  function renderStages(run) {
    const list = $("stage-list");
    if (!run.stages || !run.stages.length) {
      list.hidden = true;
      return;
    }
    list.hidden = false;
    const completed = new Set(run.completed_stages || []);
    list.innerHTML = run.stages
      .map(({ label, plain }) => {
        const isDone = completed.has(label);
        const isActive = run.running && run.current_stage === label;
        const mark = isDone ? "✓" : isActive ? "●" : "○";
        const cls = isDone ? "done" : isActive ? "active" : "";
        return `<li class="${cls}"><span class="mark">${mark}</span><span>${plain}</span></li>`;
      })
      .join("");
  }

  function renderRun(run) {
    if (run.instrument && run.instrument !== currentInstrument) return; // stale response from a stock we've since left
    const btn = $("run-btn");
    const stopBtn = $("stop-btn");
    const statusLine = $("run-status-line");
    const consoleEl = $("console");

    renderStages(run);

    if (run.running) {
      btn.disabled = true;
      btn.textContent = "Running…";
      stopBtn.hidden = false;
      statusLine.textContent = `Started ${fmtRelative(run.started_at)}.`;
    } else {
      btn.disabled = false;
      btn.textContent = "▶ Run Predictor Now";
      stopBtn.hidden = true;
      if (run.finished_at) {
        const ok = run.exit_code === 0;
        statusLine.textContent = ok
          ? `Finished ${fmtRelative(run.finished_at)} — completed successfully.`
          : `Finished ${fmtRelative(run.finished_at)} — exited with a problem (code ${run.exit_code}). See the console below.`;
      } else if (run.run_id === null) {
        statusLine.textContent = "Not run from here yet this session.";
      }
    }

    if (run.lines && run.lines.length) {
      consoleEl.hidden = false;
      consoleEl.textContent += run.lines.join("\n") + "\n";
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
    sinceLines = run.total_lines ?? sinceLines;

    if (run.running) {
      scheduleRunPoll();
    } else {
      stopRunPoll();
    }
  }

  function scheduleRunPoll() {
    if (pollTimer) return;
    const forInstrument = currentInstrument;
    pollTimer = setInterval(async () => {
      if (forInstrument !== currentInstrument) { stopRunPoll(); return; }
      try {
        const res = await fetch(`/api/run/output?instrument=${encodeURIComponent(forInstrument)}&since=${sinceLines}`);
        const run = await res.json();
        renderRun(run);
        if (!run.running) {
          stopRunPoll();
          fetchStatus();
        }
      } catch (e) { /* transient - retry next tick */ }
    }, 1000);
  }

  function stopRunPoll() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  async function triggerRun() {
    const options = {
      instrument: currentInstrument,
      skipPull: $("opt-skip-pull").checked,
      skipBackfill: $("opt-skip-backfill").checked,
      tune: Number($("opt-tune").value) || 0,
      k: Number($("opt-k").value) || 0,
      holdout: Number($("opt-holdout").value) || 0,
    };
    $("console").hidden = false;
    $("console").textContent = "";
    sinceLines = 0;
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
    const data = await res.json();
    if (!data.ok) {
      $("run-status-line").textContent = data.error || "Could not start the run.";
      return;
    }
    scheduleRunPoll();
  }

  async function stopRun() {
    if (!confirm(`Stop the run in progress for ${currentInstrument}?`)) return;
    await fetch("/api/run/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instrument: currentInstrument }),
    });
  }

  // ---- status polling -------------------------------------------------------

  async function fetchStatus() {
    try {
      const res = await fetch(`/api/status?instrument=${encodeURIComponent(currentInstrument)}`);
      const data = await res.json();
      if (data.instrument && data.instrument.key !== currentInstrument) return; // stale
      renderTopbar(data);
      renderCollector(data.collector);
      renderDataset(data.dataset);
      renderModel(data.model);
      renderBacktest(data.backtest);
      renderPaper(data.paper_trading);
      renderSchedule(data.scheduled_task);
      renderLogTail(data.log_tail);
      if (!pollTimer) renderRun(data.run);
    } catch (e) {
      $("heartbeat-dot").className = "dot bad";
    }
  }

  $("run-btn").addEventListener("click", triggerRun);
  $("stop-btn").addEventListener("click", stopRun);
  $("refresh-btn").addEventListener("click", fetchStatus);

  (async () => {
    await loadInstruments();
    fetchStatus();
  })();
  statusTimer = setInterval(() => { if (!pollTimer) fetchStatus(); }, 20000);
  setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString(); }, 1000);
})();
