/**
 * RideCourt — single-page UI (vanilla JS, no build step).
 *
 * Left: dispute queue. Centre: courtroom timeline. Right: evidence + ruling.
 * Streams events via SSE while a case resolves.
 */

const API = "/api";
let map = null;
let mapLayers = [];
let currentDisputeId = null;
let eventSource = null;
let caseDataCache = {};

// ---- Category labels ----
const CAT_LABELS = {
  route_deviation: "Route Deviation",
  no_show_driver: "No-Show (Driver)",
  no_show_rider: "No-Show (Rider)",
  fare_error: "Fare / Surge",
  property_damage: "Property Damage",
  safety: "Safety",
};

const CAT_PRIORITY_CLASS = {
  safety: "priority-safety",
  route_deviation: "priority-high",
  no_show_driver: "priority-normal",
  no_show_rider: "priority-normal",
  fare_error: "priority-normal",
  property_damage: "priority-normal",
};

// ---- Outcome human labels ----
// Maps internal outcome codes to human-readable phrases.
// The raw code is shown underneath in small text so it stays traceable.
const OUTCOME_LABELS = {
  // Ryde Dispute Resolution Policy v2.1
  refund_partial_excess_distance: "Partial refund: excess distance fare (§1.3)",
  refund_full_excess_distance: "Refund of excess distance fare (§1.3)",
  no_refund_within_tolerance: "No refund: deviation within 1 km (§1.3)",
  no_refund_justified_informed: "No refund: rider informed of a justified reroute (§1.3)",
  cancellation_fee_upheld: "Cancellation fee upheld (§2.2)",
  cancellation_fee_voided: "Cancellation fee voided (§2.1, §2.3)",
  driver_no_show_credit: "Driver no-show: fee refunded and S$5 credit (§2.4)",
  driver_no_show_no_credit: "Driver no-show: fee refunded (§2.4)",
  no_driver_no_show: "Driver arrived: not a driver no-show (§2.4)",
  refund_surge_differential: "Refund of surge differential (§3.3)",
  no_surge_refund_notified: "No refund: rider was notified of the surge (§3.3)",
  no_surge_refund: "No refund: no unnotified surge increase (§3.3)",
  damage_claim_approved: "Cleaning fee approved (§4.3)",
  claim_rejected_insufficient_evidence: "Damage claim rejected: evidence not verifiable (§4.4)",
  human_review_repair: "Repair claim: human review (§6.2)",
  human_review_rider_dispute: "Rider disputes damage charge: human review (§4.4)",
  human_review_uncovered: "Not covered by policy v2.1: human review",
  escalate_safety: "Safety incident: human review (§5)",
  // Earlier policy (kept for older stored cases)
  refund_route_deviation: "Refund for route deviation",
  no_deviation: "No route deviation",
  no_refund: "No refund",
  no_refund_rider_requested: "No refund — rider requested route change",
  no_refund_road_closure: "No refund — road closure documented",
  refund_and_credit: "Refund and goodwill credit",
  no_show_not_proven: "Driver did show — no refund",
  cancellation_fee_stands: "Cancellation fee stands",
  cancellation_fee_invalid: "Cancellation fee invalid",
  refund_overcharge: "Refund for fare overcharge",
  undercharge_noted: "Undercharge noted, not collected",
  fare_correct: "Fare is correct",
  escalate: "Escalated to human reviewer",
  requested_change: "Rider requested route change",
  not_applicable: "Not applicable",
};

function humanOutcome(code) {
  if (!code) return "—";
  return OUTCOME_LABELS[code] || code;
}

function formatMoney(amount) {
  // Always show as non-negative plain money, never negative.
  const v = Math.abs(Number(amount) || 0);
  return `S$${v.toFixed(2)}`;
}

// ---- Initialization ----
document.addEventListener("DOMContentLoaded", () => {
  loadDisputes();
  loadTripsForForm();

  document.getElementById("refresh-btn").addEventListener("click", loadDisputes);
  document.getElementById("file-dispute-btn").addEventListener("click", openFileModal);
  document.getElementById("cancel-file").addEventListener("click", closeFileModal);
  document.getElementById("file-dispute-form").addEventListener("submit", fileDispute);
  document.getElementById("cancel-override").addEventListener("click", closeOverrideModal);
  document.getElementById("override-form").addEventListener("submit", submitOverride);

  // Close modals on backdrop click.
  document.getElementById("file-modal").addEventListener("click", (e) => {
    if (e.target.id === "file-modal") closeFileModal();
  });
  document.getElementById("override-modal").addEventListener("click", (e) => {
    if (e.target.id === "override-modal") closeOverrideModal();
  });
});

// ---- Queue ----
async function loadDisputes() {
  try {
    const res = await fetch(`${API}/disputes`);
    const data = await res.json();
    renderQueue(data.disputes);
  } catch (err) {
    console.error("Failed to load disputes:", err);
  }
}

function renderQueue(disputes) {
  const container = document.getElementById("dispute-queue");
  container.innerHTML = "";

  if (!disputes || disputes.length === 0) {
    container.innerHTML = '<p class="placeholder">No disputes found.</p>';
    return;
  }

  const samples = disputes.filter((d) => d.ryde_sample);
  const rest = disputes.filter((d) => !d.ryde_sample);
  const ordered = samples.concat(rest);
  if (samples.length) {
    const h = document.createElement("div");
    h.className = "queue-section";
    h.textContent = "Ryde sample cases";
    container.appendChild(h);
  }

  for (const d of ordered) {
    if (samples.length && d === rest[0]) {
      const h = document.createElement("div");
      h.className = "queue-section";
      h.textContent = "Queue (SLA order)";
      container.appendChild(h);
    }
    const item = document.createElement("div");
    item.className = "queue-item";
    if (d.dispute_id === currentDisputeId) item.classList.add("active");
    item.dataset.id = d.dispute_id;

    const priorityClass = CAT_PRIORITY_CLASS[d.category] || "priority-normal";
    const statusClass = `status-${d.status}`;

    item.innerHTML = `
      <div class="queue-item-header">
        <span class="queue-cat">${CAT_LABELS[d.category] || d.category}</span>
        <span class="priority-badge ${priorityClass}">${d.category === "safety" ? "SAFETY" : d.priority}</span>
      </div>
      <div class="queue-item-header">
        <span class="queue-id">${d.dispute_id}${d.ryde_sample ? ' <span class="ryde-badge">Ryde sample</span>' : ""}</span>
        <span class="status-badge ${statusClass}">${d.status}</span>
      </div>
      <div class="queue-claim">${escapeHtml(d.claim_text)}</div>
    `;

    item.addEventListener("click", () => selectDispute(d.dispute_id));
    container.appendChild(item);
  }
}

// ---- Select dispute ----
async function selectDispute(disputeId) {
  currentDisputeId = disputeId;
  // Update active state in the queue.
  document.querySelectorAll(".queue-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === disputeId);
  });

  // Clear courtroom and evidence. The map container is persistent (moved
  // around, not destroyed), so clearing innerHTML here is safe.
  document.getElementById("courtroom-timeline").innerHTML = "";
  document.getElementById("evidence-content").innerHTML =
    '<p class="placeholder">Loading case...</p>';
  document.getElementById("ruling-card").classList.add("hidden");
  updateStatusBadge("loading");

  // Close any existing SSE.
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Fetch the case file.
  await fetchCase(disputeId);
}

async function fetchCase(disputeId) {
  try {
    const res = await fetch(`${API}/disputes/${disputeId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    caseDataCache[disputeId] = data;

    updateStatusBadge(data.status);
    renderEvidence(data);
    renderRuling(data);

    // If the case has events, render them.
    if (data.events && data.events.length > 0) {
      renderPastEvents(data.events);
    }

    if (data.status === "new") {
      // Auto-resolve new cases when selected.
      resolveDispute(disputeId);
    } else if (data.status === "resolving") {
      // Subscribe to the live stream.
      subscribeToEvents(disputeId);
    } else if (data.status === "ruled" || data.status === "escalated") {
      // Already resolved — check if there are events we haven't shown.
      if (!data.events || data.events.length === 0) {
        // Subscribe to replay events from the log.
        subscribeToEvents(disputeId);
      }
    }
  } catch (err) {
    console.error("Failed to fetch case:", err);
    document.getElementById("evidence-content").innerHTML =
      '<p class="placeholder">Failed to load case.</p>';
  }
}

// ---- Resolve ----
async function resolveDispute(disputeId) {
  updateStatusBadge("resolving");

  try {
    const res = await fetch(`${API}/disputes/${disputeId}/resolve`, {
      method: "POST",
    });
    if (res.status === 429) {
      addEventRow("system", "Rate limit reached. Please wait and try again.");
      return;
    }
    if (!res.ok && res.status !== 200) {
      const err = await res.json();
      addEventRow("system", `Error: ${err.detail || res.statusText}`);
      return;
    }

    const data = await res.json();
    if (data.status === "resolving") {
      subscribeToEvents(disputeId);
    } else if (data.status === "ruled" || data.status === "escalated") {
      // Already resolved — refresh the case.
      await fetchCase(disputeId);
    }
  } catch (err) {
    console.error("Failed to resolve:", err);
    addEventRow("system", "Failed to start resolution.");
  }
}

// ---- SSE event stream ----
function subscribeToEvents(disputeId) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`${API}/disputes/${disputeId}/events`);

  eventSource.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    updateStatusBadge(data.status);

    // Refresh the case file to get the full ruling + evidence.
    fetchCase(disputeId);
    loadDisputes();
    eventSource.close();
    eventSource = null;
  });

  // Listen for all event types.
  const eventTypes = [
    "DISPUTE_FILED", "EVIDENCE_REQUEST", "EVIDENCE", "CLAIM",
    "REBUTTAL", "RISK_SIGNAL", "PRECEDENT", "RULING", "ESCALATION",
    "HUMAN_OVERRIDE", "GUARD_CHECK", "GUARD_REJECTED", "POLICY_EVAL",
  ];

  for (const type of eventTypes) {
    eventSource.addEventListener(type, (e) => {
      const ev = JSON.parse(e.data);
      handleEvent(ev);
    });
  }

  eventSource.onerror = () => {
    // SSE errors are normal when the stream closes; the done event handles cleanup.
    if (eventSource.readyState === EventSource.CLOSED) {
      eventSource = null;
    }
  };
}

function renderPastEvents(events) {
  const timeline = document.getElementById("courtroom-timeline");
  timeline.innerHTML = "";
  for (const ev of events) {
    handleEvent(ev, false);
  }
}

function handleEvent(ev, live = true) {
  addEventBubble(ev);

  // If this is a RULING or ESCALATION event, update the ruling card.
  if (ev.type === "RULING" || ev.type === "ESCALATION") {
    // The case file may not be ready yet; fetch it.
    if (live) {
      setTimeout(() => fetchCase(ev.case_id), 200);
    }
  }
}

// ---- Courtroom timeline ----
function addEventRow(agent, text) {
  const ev = {
    from_agent: agent,
    type: "SYSTEM",
    content: { text: text },
  };
  addEventBubble(ev);
}

function addEventBubble(ev) {
  const timeline = document.getElementById("courtroom-timeline");
  // Remove placeholder.
  const ph = timeline.querySelector(".placeholder");
  if (ph) ph.remove();

  const agent = ev.from_agent || "system";
  const type = ev.type;

  // Determine alignment and style.
  let side = "system";
  if (agent.includes("rider")) side = "rider";
  else if (agent.includes("driver")) side = "driver";
  else if (agent === "judge") side = "judge";
  else if (agent === "orchestrator") side = "orchestrator";
  else if (agent === "policy_engine") side = "policy_engine";
  else if (agent === "evidence_tools") side = "evidence_tools";
  else if (agent === "guard") side = "guard";
  else if (agent === "human_reviewer") side = "system";

  const row = document.createElement("div");
  row.className = `event-row ${side}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${side}`;
  if (type === "ESCALATION") bubble.classList.add("escalation");

  // Build label.
  const labelMap = {
    DISPUTE_FILED: "Dispute Filed",
    EVIDENCE_REQUEST: "Evidence Request",
    EVIDENCE: "Evidence Collected",
    CLAIM: "Claim",
    REBUTTAL: "Rebuttal",
    RISK_SIGNAL: "Risk Signal",
    PRECEDENT: "Precedent",
    RULING: "Ruling",
    ESCALATION: "Escalation",
    HUMAN_OVERRIDE: "Human Override",
    GUARD_CHECK: "Guard Check",
    GUARD_REJECTED: "Guard Rejected",
    POLICY_EVAL: "Policy Evaluation",
  };

  const labelText = labelMap[type] || type;
  const agentLabel = agent === "rider_advocate" ? "Rider Advocate"
    : agent === "driver_advocate" ? "Driver Advocate"
    : agent === "judge" ? "Judge"
    : agent === "orchestrator" ? "Orchestrator"
    : agent === "policy_engine" ? "Policy Engine"
    : agent === "evidence_tools" ? "Evidence Tools"
    : agent === "guard" ? "Guard"
    : agent === "human_reviewer" ? "Human Reviewer"
    : "System";

  let html = `<div class="bubble-label">${agentLabel} · ${labelText}</div>`;

  // Content.
  const content = ev.content;
  if (content) {
    if (typeof content === "string") {
      html += `<div>${escapeHtml(content)}</div>`;
    } else {
      // Render structured content.
      if (content.claim_text) {
        html += `<div>${escapeHtml(content.claim_text)}</div>`;
      }
      if (content.category) {
        html += `<div><em>Category: ${escapeHtml(content.category)}</em></div>`;
      }
      if (content.position) {
        html += `<div><strong>${escapeHtml(content.position)}</strong></div>`;
      }
      if (content.arguments) {
        for (const arg of content.arguments) {
          html += `<div>· ${escapeHtml(arg.text)}</div>`;
        }
      }
      if (content.requested_outcome) {
        html += `<div><em>Requested: ${escapeHtml(content.requested_outcome)}</em></div>`;
      }
      if (content.items) {
        const items = content.items.map((i) => i.id).join(", ");
        html += `<div>Evidence: ${escapeHtml(items)}</div>`;
      }
      if (content.outcome) {
        html += `<div>Outcome: <strong>${escapeHtml(humanOutcome(content.outcome))}</strong> <span class="outcome-code">${escapeHtml(content.outcome)}</span></div>`;
      }
      if (content.amount !== undefined && content.amount !== null) {
        html += `<div>Amount: <strong>${formatMoney(content.amount)}</strong></div>`;
      }
      if (content.confidence !== undefined && content.confidence !== null) {
        html += `<div>Confidence: ${(content.confidence * 100).toFixed(0)}%</div>`;
      }
      if (content.computation) {
        html += `<div class="computation-string">${escapeHtml(content.computation)}</div>`;
      }
      if (content.clauses) {
        html += `<div>Clauses: ${escapeHtml(content.clauses.join(", "))}</div>`;
      }
      if (content.message_to_rider) {
        html += `<div class="message-block"><div class="message-label">To Rider</div>${escapeHtml(content.message_to_rider)}</div>`;
      }
      if (content.message_to_driver) {
        html += `<div class="message-block"><div class="message-label">To Driver</div>${escapeHtml(content.message_to_driver)}</div>`;
      }
      if (content.routing_reason && type === "ESCALATION") {
        html += `<div>Sent to a human reviewer: ${escapeHtml(content.routing_reason)}</div>`;
      } else if (content.routing_reason && type === "RULING") {
        html += `<div>Ruled by the AI (final): ${escapeHtml(content.routing_reason)}</div>`;
      } else if (content.reason) {
        html += `<div>Reason: ${escapeHtml(content.reason)}</div>`;
      }
      if (content.summary) {
        html += `<details class="case-file"><summary>Case file for the reviewer</summary><div>${escapeHtml(content.summary)}</div></details>`;
      }
      // Precedent matches — render each with source label and similarity.
      if (content.matches) {
        for (const m of content.matches) {
          const simPct = Math.round((m.similarity || 0) * 100);
          let matchLine = `<div class="precedent-match">`;
          matchLine += `<span class="precedent-case">${escapeHtml(m.case_id)}</span>`;
          matchLine += ` <span class="precedent-sim">${simPct}%</span>`;
          if (m.source === "override") {
            const origId = m.case_id.replace(/^override-/, "");
            matchLine += ` <span class="precedent-source override">Reviewer override of ${escapeHtml(origId)}</span>`;
          }
          matchLine += ` <span class="precedent-summary">${escapeHtml(m.summary || "")}</span>`;
          matchLine += `</div>`;
          html += matchLine;
        }
      }
      // Learned-from-override flag on the PRECEDENT event.
      if (content.learned_from_override) {
        html += `<div class="learned-from-override">Learned from a reviewer override</div>`;
      }
    }
  }

  // Cites as chips.
  if (ev.cites && ev.cites.length > 0) {
    html += '<div class="chips">';
    for (const cite of ev.cites) {
      const isPolicy = !/^E\d+$/.test(cite);  // §x.y, precedent IDs, CLAIM
      const cls = isPolicy ? "chip policy" : "chip";
      html += `<span class="${cls}" data-cite="${escapeHtml(cite)}">${escapeHtml(cite)}</span>`;
    }
    html += "</div>";
  }

  bubble.innerHTML = html;

  // Attach click handlers to evidence chips.
  bubble.querySelectorAll(".chip[data-cite]").forEach((chip) => {
    if (/^E\d+$/.test(chip.dataset.cite)) {
      chip.addEventListener("click", () => {
        highlightEvidence(chip.dataset.cite);
      });
    }
  });

  row.appendChild(bubble);
  timeline.appendChild(row);
  timeline.scrollTop = timeline.scrollHeight;
}

// ---- Evidence panel ----
function renderEvidence(data) {
  const container = document.getElementById("evidence-content");
  container.innerHTML = "";

  if (!data.trip) {
    container.innerHTML = '<p class="placeholder">No trip data.</p>';
    return;
  }

  const trip = data.trip;

  // Map section (only if GPS trace exists).
  if (trip.gps_trace && trip.gps_trace.length > 0) {
    const mapSection = document.createElement("div");
    mapSection.className = "evidence-section";
    mapSection.innerHTML = `
      <h4>Route Map</h4>
      <div class="map-slot"></div>
      <div class="map-legend">
        <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span> Actual GPS</div>
        <div class="legend-item"><span class="legend-dot" style="background:#2563eb"></span> Optimal Route</div>
      </div>
    `;
    container.appendChild(mapSection);

    // Render the map after the DOM is updated.
    setTimeout(() => renderMap(trip), 50);
  }

  // Fare table.
  if (trip.fare) {
    const fareSection = document.createElement("div");
    fareSection.className = "evidence-section";
    const fare = trip.fare;
    // Only lines the trip record actually has: rate components when the
    // source gives them, otherwise the quote, the itemised lines and surge.
    const money = (v) => `S$${Number(v).toFixed(2)}`;
    const rows = [];
    if (fare.base > 0) rows.push(["Base fare", money(fare.base)]);
    if (fare.per_km > 0) rows.push(["Per km rate", `${money(fare.per_km)}/km`]);
    if (fare.per_min > 0) rows.push(["Per min rate", `${money(fare.per_min)}/min`]);
    if (fare.distance_km > 0) rows.push(["Distance", `${fare.distance_km.toFixed(2)} km`]);
    if (fare.duration_min > 0) rows.push(["Duration", `${fare.duration_min.toFixed(1)} min`]);
    if (fare.upfront_quote > 0) rows.push(["Upfront quote / estimate", money(fare.upfront_quote)]);
    if (fare.excess_distance_km > 0) rows.push(["Excess distance", `${fare.excess_distance_km.toFixed(2)} km`]);
    if (fare.excess_distance_fare > 0) rows.push(["Excess distance fare", money(fare.excess_distance_fare)]);
    if (fare.quote_surge_multiplier && fare.trip_start_surge_multiplier && fare.quote_surge_multiplier !== fare.trip_start_surge_multiplier) {
      rows.push(["Surge at quote", `${fare.quote_surge_multiplier.toFixed(1)}×`]);
      rows.push(["Surge at trip start", `${fare.trip_start_surge_multiplier.toFixed(1)}×`]);
    } else {
      rows.push(["Surge multiplier", `${fare.surge_multiplier.toFixed(1)}×`]);
    }
    if (fare.surgeable_subtotal > 0) rows.push(["Surgeable subtotal", money(fare.surgeable_subtotal)]);
    if (fare.promo_code) rows.push(["Promo code", escapeHtml(fare.promo_code)]);
    if (fare.discount > 0) rows.push(["Discount", `−${money(fare.discount)}`]);
    if (fare.booking_fee > 0) rows.push(["Booking fee", money(fare.booking_fee)]);
    // A cancelled trip (no-show) was charged the cancellation fee, not a fare.
    const cancelled = trip.status && trip.status !== "completed" && fare.cancellation_fee > 0 && !(fare.total_charged > 0 && fare.distance_km > 0);
    fareSection.innerHTML = `
      <h4>Fare Breakdown</h4>
      <table class="fare-table">
        <tr><th>Component</th><th>Value</th></tr>
        ${rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}
        ${cancelled
          ? `<tr class="total"><td>Cancellation fee charged</td><td>${money(fare.cancellation_fee)}</td></tr>`
          : `<tr class="total"><td>Total charged</td><td>${money(fare.total_charged)}</td></tr>`}
      </table>
    `;
    container.appendChild(fareSection);
  }

  // Chat excerpts.
  if (data.chat && data.chat.messages && data.chat.messages.length > 0) {
    const chatSection = document.createElement("div");
    chatSection.className = "evidence-section";
    let chatHtml = "<h4>Chat Excerpts</h4>";
    for (const msg of data.chat.messages.slice(0, 15)) {
      chatHtml += `<div class="chat-excerpt"><strong>${escapeHtml(msg.sender)}</strong> [${escapeHtml(msg.ts)}]: ${highlightQuotes(msg.text)}</div>`;
    }
    chatSection.innerHTML = chatHtml;
    container.appendChild(chatSection);
  }

  // Evidence items.
  if (data.evidence_pack && data.evidence_pack.items) {
    const evSection = document.createElement("div");
    evSection.className = "evidence-section";
    let evHtml = "<h4>Evidence Pack</h4>";
    for (const item of data.evidence_pack.items) {
      const numData = {};
      if (item.data) {
        for (const [k, v] of Object.entries(item.data)) {
          if (typeof v === "number" && !Array.isArray(v)) {
            numData[k] = v;
          }
        }
      }
      evHtml += `
        <div class="evidence-item" id="evidence-${escapeHtml(item.id)}">
          <span class="eid">${escapeHtml(item.id)}</span>
          <span class="chip" style="font-size:9px">${escapeHtml(item.kind)}</span>
          <div class="esummary">${escapeHtml(item.summary)}</div>
          ${Object.keys(numData).length > 0 ? `<div class="edata">${escapeHtml(Object.entries(numData).map(([k,v]) => `${k}=${v}`).join(", "))}</div>` : ""}
        </div>
      `;
    }
    evSection.innerHTML = evHtml;
    container.appendChild(evSection);
  }
}

function highlightQuotes(text) {
  // Highlight quoted text in chat messages.
  const quoted = text.replace(/"([^"]+)"/g, '<span class="quoted">"$1"</span>');
  return escapeHtml(text).replace(/&quot;([^&]+)&quot;/g, '<span class="quoted">"$1"</span>');
}

function highlightEvidence(eid) {
  // Collapse the ruling card so the evidence area has room.
  const card = document.getElementById("ruling-card");
  if (card && !card.classList.contains("collapsed")) {
    card.classList.add("collapsed");
    const toggle = card.querySelector(".ruling-toggle");
    if (toggle) toggle.textContent = "Show details \u22BE";
  }

  // Scroll to and flash the evidence item.
  const el = document.getElementById(`evidence-${eid}`);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.style.transition = "background 0.3s";
    el.style.background = "var(--accent-light)";
    setTimeout(() => { el.style.background = ""; }, 1000);
  }
}

// ---- Leaflet map ----
// The map is created once and reused across case switches.  Removing and
// re-creating the map on every case switch causes a Leaflet error
// ("Cannot read properties of undefined (reading '_leaflet_pos')") when the
// container is hidden or removed during the transition.
//
// Approach: a persistent #map-container div is created in JS and kept
// alive.  When the evidence panel needs the map, the container is moved
// into the map section.  Old layers are cleared (not the map) and
// invalidateSize() is called after the panel is visible.

let mapContainer = null; // persistent DOM element holding the Leaflet map

function ensureMapContainer() {
  if (!mapContainer) {
    mapContainer = document.createElement("div");
    mapContainer.id = "map";
  }
  return mapContainer;
}

function ensureMap() {
  ensureMapContainer();
  if (map) return map;
  if (!mapContainer.parentElement) return null;

  map = L.map(mapContainer, { center: [1.3521, 103.9450], zoom: 13 });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map);

  return map;
}

function clearMapLayers() {
  if (!map) return;
  for (const layer of mapLayers) {
    if (map.hasLayer(layer)) {
      map.removeLayer(layer);
    }
  }
  mapLayers = [];
}

function renderMap(trip) {
  ensureMapContainer();

  // Move the persistent container into the map section (it may have been
  // removed from the DOM if the evidence panel was cleared).
  const mapSection = document.querySelector(".evidence-section .map-slot");
  if (mapSection && mapContainer.parentElement !== mapSection) {
    mapSection.appendChild(mapContainer);
  }

  // Create the map now that the container is in the DOM.
  ensureMap();
  if (!map) return;

  // Clear previous layers (don't remove the map itself).
  clearMapLayers();

  // Actual GPS trace (red).
  if (trip.gps_trace && trip.gps_trace.length > 0) {
    const tracePoints = trip.gps_trace.map((p) => [p.lat, p.lon]);
    const traceLine = L.polyline(tracePoints, {
      color: "#ef4444",
      weight: 3,
      opacity: 0.8,
    }).addTo(map);
    mapLayers.push(traceLine);

    // Start and end markers.
    const startMarker = L.circleMarker(tracePoints[0], {
      color: "#ef4444", fillColor: "#ef4444", fillOpacity: 1, radius: 5,
    }).addTo(map).bindTooltip("Pickup");
    const endMarker = L.circleMarker(tracePoints[tracePoints.length - 1], {
      color: "#ef4444", fillColor: "#ef4444", fillOpacity: 1, radius: 5,
    }).addTo(map).bindTooltip("Dropoff");
    mapLayers.push(startMarker, endMarker);
  }

  // Optimal route (blue).
  if (trip.optimal_route && trip.optimal_route.points && trip.optimal_route.points.length > 0) {
    const optimalPoints = trip.optimal_route.points.map((p) => [p.lat, p.lon]);
    const optimalLine = L.polyline(optimalPoints, {
      color: "#2563eb",
      weight: 2,
      opacity: 0.6,
      dashArray: "5, 5",
    }).addTo(map);
    mapLayers.push(optimalLine);
  }

  // Fit bounds to show both routes.
  if (mapLayers.length > 0) {
    const group = L.featureGroup(mapLayers);
    map.fitBounds(group.getBounds(), { padding: [20, 20] });
  }

  // Invalidate size after the panel is visible — the container may have
  // been moved or resized when switching cases.
  setTimeout(() => { if (map) map.invalidateSize(); }, 100);
}

// ---- Ruling card ----
function renderRuling(data) {
  const card = document.getElementById("ruling-card");
  const body = card.querySelector(".ruling-body");

  if (!data.ruling && !data.escalated) {
    card.classList.add("hidden");
    return;
  }

  card.classList.remove("hidden");

  if (data.escalated && !data.ruling) {
    // Escalation without a ruling (e.g. safety case).
    body.innerHTML = `
      <div class="ruling-row"><span class="label">Status</span><span class="value" style="color:var(--escalate)">Escalated</span></div>
      <div class="message-block"><div class="message-label">Reason</div>${escapeHtml(routingReason(data))}</div>
      ${actionsHtml(data.policy_result)}
      <div class="ruling-actions">
        <button class="btn btn-override" onclick="openOverrideModal()">Override</button>
      </div>
    `;
    return;
  }

  const ruling = data.ruling;
  const policy = data.policy_result;
  // Routing (§6) uses the confidence the policy engine computes from the four
  // §6.3 factors; the Judge model's own confidence is shown in the details.
  const engineConf = policy && policy.confidence !== undefined && policy.confidence !== null
    ? policy.confidence : ruling.confidence;
  const confidencePct = Math.round((engineConf || 0) * 100);
  const confidenceColor = confidencePct >= 80 ? "var(--success)" : "var(--warning)";
  const sentToHuman = data.escalated || (policy && policy.routing === "human");

  // --- Summary (always visible, click to toggle) ---
  let summaryHtml = `<div class="ruling-summary">`;

  // Routing banner: who decides, and the §6 rule that decided it.
  if (sentToHuman) {
    summaryHtml += `<div class="ruling-banner escalate">Sent to a human reviewer: ${escapeHtml(routingReason(data))}</div>`;
  } else if (policy && policy.routing === "ai_final") {
    summaryHtml += `<div class="ruling-banner ai-final">Ruled by the AI (final): ${escapeHtml(policy.routing_reason || "")}</div>`;
  }

  // Outcome and amount — human words with the code underneath in small text.
  const outcomeLabel = sentToHuman ? "Recommended" : "Outcome";
  summaryHtml += `<div class="ruling-row"><span class="label">${outcomeLabel}</span><span class="value">${escapeHtml(humanOutcome(ruling.outcome))}<span class="outcome-code">${escapeHtml(ruling.outcome)}</span></span></div>`;
  summaryHtml += `<div class="ruling-row"><span class="label">Amount</span><span class="value">${formatMoney(ruling.amount)}</span></div>`;

  // Confidence bar.
  summaryHtml += `<div class="ruling-row"><span class="label">Confidence (§6.3)</span><span class="value">${confidencePct}%</span></div>`;
  summaryHtml += `<div class="confidence-bar"><div class="confidence-fill" style="width:${confidencePct}%;background:${confidenceColor}"></div></div>`;

  // Credibility marker (§6.3 party credibility).
  const credibility = policy && policy.confidence_factors ? policy.confidence_factors.party_credibility : undefined;
  if (ruling.bad_faith_mentioned || (credibility !== undefined && credibility < 1)) {
    summaryHtml += `<div class="ruling-marker bad-faith">Credibility lowered (§6.3)</div>`;
  }

  // Precedent consistency.
  if (ruling.precedent_consistency && ruling.precedent_consistency !== "no_precedents") {
    const pc = ruling.precedent_consistency;
    const label = pc === "consistent" ? "Consistent with precedents" :
                  pc === "inconsistent" ? "Inconsistent with precedents" :
                  escapeHtml(pc);
    const cls = pc === "inconsistent" ? "ruling-marker inconsistent" : "ruling-marker consistent";
    summaryHtml += `<div class="${cls}">${label}</div>`;
  }

  // Learned from a reviewer override — when the top precedent came from an
  // override, say so explicitly on the ruling card.
  if (data.precedent_matches && data.precedent_matches.length > 0) {
    const topPrec = data.precedent_matches[0];
    if (topPrec.source === "override") {
      summaryHtml += `<div class="ruling-marker learned-from-override">Learned from a reviewer override</div>`;
    }
  }

  summaryHtml += `<span class="ruling-toggle">Show details \u22BE</span>`;
  summaryHtml += `</div>`;

  // --- Details (collapsible) ---
  let detailsHtml = `<div class="ruling-details">`;

  // Computation string from policy engine.
  if (policy && policy.computation) {
    detailsHtml += `<div class="ruling-row"><span class="label">Computation</span></div>`;
    detailsHtml += `<div class="computation-string">${escapeHtml(policy.computation)}</div>`;
  }

  // The four §6.3 confidence factors, each with the inputs it read.
  if (policy && policy.confidence_factors) {
    const f = policy.confidence_factors;
    const rows = [
      ["Evidence completeness", f.evidence_completeness, f.evidence_reason],
      ["Policy clarity", f.policy_clarity, f.policy_clarity_reason],
      ["Precedent alignment", f.precedent_alignment, f.precedent_reason],
      ["Party credibility", f.party_credibility, f.credibility_reason],
    ].filter((r) => r[1] !== undefined);
    if (rows.length) {
      detailsHtml += `<div class="ruling-row"><span class="label">Confidence factors (§6.3)</span></div>`;
      detailsHtml += `<div class="confidence-factors">${rows.map((r) =>
        `<div class="factor"><span class="factor-name">${r[0]}</span> <span class="factor-score">${Math.round(r[1] * 100)}%</span><div class="factor-reason">${escapeHtml(r[2] || "")}</div></div>`
      ).join("")}</div>`;
      if (ruling.confidence !== undefined && ruling.confidence !== null) {
        detailsHtml += `<div class="factor-reason">Judge model's own confidence: ${Math.round(ruling.confidence * 100)}% (§6 routing uses the engine's confidence above; a Judge below 70% also sends the case to a human)</div>`;
      }
    }
  }
  detailsHtml += actionsHtml(policy);

  // Cited clauses — render all cites as chips (E# as evidence chips,
  // P-# as policy chips), matching the courtroom's chip styling.
  if (ruling.cites && ruling.cites.length > 0) {
    detailsHtml += `<div class="ruling-row"><span class="label">Citations</span></div>`;
    detailsHtml += `<div class="clauses">${ruling.cites.map((c) => {
      const isPolicy = !/^E\d+$/.test(c);  // §x.y, precedent IDs, CLAIM
      const cls = isPolicy ? "chip policy" : "chip";
      const clickable = !isPolicy ? ` onclick="highlightEvidence('${escapeHtml(c)}')" style="cursor:pointer"` : "";
      return `<span class="${cls}"${clickable}>${escapeHtml(c)}</span>`;
    }).join("")}</div>`;
  }

  // Judge's reasoning — displayed in full with citation chips.
  if (ruling.reasoning) {
    detailsHtml += `<div class="ruling-row"><span class="label">Reasoning</span></div>`;
    detailsHtml += `<div class="reasoning-block">${escapeHtml(ruling.reasoning)}</div>`;
  }

  // Messages.
  if (ruling.message_to_rider) {
    detailsHtml += `<div class="message-block"><div class="message-label">Message to Rider</div>${escapeHtml(ruling.message_to_rider)}</div>`;
  }
  if (ruling.message_to_driver) {
    detailsHtml += `<div class="message-block"><div class="message-label">Message to Driver</div>${escapeHtml(ruling.message_to_driver)}</div>`;
  }

  // Escalate / Override / Run Again buttons.
  detailsHtml += `<div class="ruling-actions">`;
  detailsHtml += `<button class="btn btn-escalate" onclick="escalateCase('${escapeHtml(data.dispute.dispute_id)}')">Escalate</button>`;
  detailsHtml += `<button class="btn btn-override" onclick="openOverrideModal()">Override</button>`;
  if (sentToHuman && policy && policy.outcome) {
    summaryHtml = summaryHtml.replace(`<span class="ruling-toggle">`,
      `<button class="btn btn-approve" onclick="event.stopPropagation(); approveRecommended('${escapeHtml(data.dispute.dispute_id)}')">Approve recommended ruling</button><span class="ruling-toggle">`);
  }
  if (data.status === "ruled" || data.status === "escalated") {
    detailsHtml += `<button class="btn btn-rerun" onclick="rerunCase('${escapeHtml(data.dispute.dispute_id)}')">Run Again</button>`;
  }
  detailsHtml += `</div>`;

  detailsHtml += `</div>`;

  body.innerHTML = summaryHtml + detailsHtml;

  // Default: collapsed so evidence is visible alongside the ruling.
  card.classList.add("collapsed");

  // Toggle handler on the summary header.
  body.querySelector(".ruling-summary").addEventListener("click", () => {
    card.classList.toggle("collapsed");
    const toggle = body.querySelector(".ruling-toggle");
    if (toggle) {
      toggle.textContent = card.classList.contains("collapsed")
        ? "Show details \u22BE"
        : "Hide details \u22BD";
    }
  });
}

// Reason a case went to a human: the §6 routing reason when there is one.
function routingReason(data) {
  const policy = data.policy_result;
  if (policy && policy.routing === "human" && policy.routing_reason) return policy.routing_reason;
  const code = data.escalation_reason || "";
  if (code === "policy_routing_human" && policy && policy.routing_reason) return policy.routing_reason;
  return code || "N/A";
}

// Actions on the ruling (refund, account flag §2.5, interim measure §5.3, ...).
function actionsHtml(policy) {
  if (!policy || !policy.actions || policy.actions.length === 0) return "";
  return `<div class="ruling-row"><span class="label">Actions</span></div><ul class="ruling-actions-list">${policy.actions.map((a) =>
    `<li>${escapeHtml(a.description || a.action || "")}</li>`).join("")}</ul>`;
}

// A reviewer approves the engine's recommended ruling as it stands.
async function approveRecommended(disputeId) {
  const data = caseDataCache[disputeId];
  const policy = data && data.policy_result;
  if (!policy) return;
  const reason = `Reviewer approved the recommended ruling (${policy.routing_reason || "human review"}).`;
  try {
    const res = await fetch(`${API}/disputes/${disputeId}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome: policy.outcome, amount: policy.amount, reason }),
    });
    if (res.ok) {
      await fetchCase(disputeId);
      loadDisputes();
    }
  } catch (err) {
    console.error("Failed to approve the recommended ruling:", err);
  }
}

// ---- Escalate ----
async function escalateCase(disputeId) {
  if (!confirm("Escalate this case to a human reviewer?")) return;
  try {
    // Use the override endpoint with an escalation outcome.
    const res = await fetch(`${API}/disputes/${disputeId}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        outcome: "escalated",
        amount: 0,
        reason: "Manual escalation by reviewer",
      }),
    });
    if (res.ok) {
      await fetchCase(disputeId);
      loadDisputes();
    }
  } catch (err) {
    console.error("Failed to escalate:", err);
  }
}

// ---- Run Again (re-resolve a case, counts toward the cost guard) ----
async function rerunCase(disputeId) {
  if (!confirm("Re-run this case live? This consumes a resolve from the daily budget.")) return;
  try {
    const res = await fetch(`${API}/disputes/${disputeId}/resolve?force=true`, {
      method: "POST",
    });
    if (res.status === 429) {
      const err = await res.json();
      addEventRow("system", err.detail || "Daily resolve limit reached.");
      return;
    }
    if (!res.ok) {
      const err = await res.json();
      addEventRow("system", `Error: ${err.detail || res.statusText}`);
      return;
    }
    const data = await res.json();
    if (data.status === "resolving") {
      // Clear the courtroom and subscribe to the new live run.
      document.getElementById("courtroom-timeline").innerHTML = "";
      document.getElementById("ruling-card").classList.add("hidden");
      updateStatusBadge("resolving");
      subscribeToEvents(disputeId);
    } else {
      await fetchCase(disputeId);
    }
  } catch (err) {
    console.error("Failed to re-run case:", err);
    addEventRow("system", "Failed to start re-run.");
  }
}

// ---- Override modal ----
function openOverrideModal() {
  document.getElementById("override-modal").classList.remove("hidden");
}

function closeOverrideModal() {
  document.getElementById("override-modal").classList.add("hidden");
}

async function submitOverride(e) {
  e.preventDefault();
  if (!currentDisputeId) return;

  const outcome = document.getElementById("override-outcome").value;
  const amount = parseFloat(document.getElementById("override-amount").value);
  const reason = document.getElementById("override-reason").value;

  try {
    const res = await fetch(`${API}/disputes/${currentDisputeId}/override`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome, amount, reason }),
    });
    if (res.ok) {
      closeOverrideModal();
      await fetchCase(currentDisputeId);
      loadDisputes();
    }
  } catch (err) {
    console.error("Failed to submit override:", err);
  }
}

// ---- File dispute modal ----
async function loadTripsForForm() {
  try {
    const res = await fetch(`${API}/disputes`);
    const data = await res.json();
    // We need trips — load from the first dispute's case file.
    // Actually, we should have a trips endpoint, but we can use the case file.
    // For simplicity, populate from the disputes list.
    const select = document.getElementById("form-trip");
    select.innerHTML = "";
    for (const d of data.disputes) {
      const opt = document.createElement("option");
      opt.value = d.trip_id;
      opt.textContent = `${d.trip_id} (${d.dispute_id})`;
      select.appendChild(opt);
    }
  } catch (err) {
    console.error("Failed to load trips:", err);
  }
}

function openFileModal() {
  document.getElementById("file-modal").classList.remove("hidden");
}

function closeFileModal() {
  document.getElementById("file-modal").classList.add("hidden");
}

async function fileDispute(e) {
  e.preventDefault();
  const tripId = document.getElementById("form-trip").value;
  const category = document.getElementById("form-category").value;
  const claimText = document.getElementById("form-claim").value;

  try {
    const res = await fetch(`${API}/disputes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId, category, claim_text: claimText }),
    });
    if (res.ok) {
      closeFileModal();
      document.getElementById("file-dispute-form").reset();
      loadDisputes();
    }
  } catch (err) {
    console.error("Failed to file dispute:", err);
  }
}

// ---- Helpers ----
function updateStatusBadge(status) {
  const badge = document.getElementById("case-status-badge");
  const labels = {
    new: "New",
    resolving: "Resolving...",
    ruled: "Ruled",
    escalated: "Escalated",
    overridden: "Overridden",
    error: "Error",
    loading: "Loading...",
  };
  badge.textContent = labels[status] || status;
  badge.className = `status-badge status-${status}`;
}

function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}
