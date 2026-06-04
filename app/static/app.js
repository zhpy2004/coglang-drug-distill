const log = document.getElementById("log");
const form = document.getElementById("form");
const qIn = document.getElementById("q");

async function refreshDomain() {
  const d = await (await fetch("/domain")).json();
  document.getElementById("domain").textContent =
    `${d.name} — ${d.nodes} nodes / ${d.edges} edges`;
}

function addTurn(html) {
  const div = document.createElement("div");
  div.className = "turn";
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function esc(s) {
  return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

async function execConfirmed(coglang, div) {
  const r = await (await fetch("/execute", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ coglang }),
  })).json();
  div.querySelector(".confirm").remove();
  div.innerHTML += `<div class="result">${esc(r.result || r.error || "")}</div>`;
  refreshDomain();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = qIn.value.trim();
  if (!q) return;
  qIn.value = "";
  const div = addTurn(`<div class="you">${esc(q)}</div><div class="coglang">…</div>`);
  const r = await (await fetch("/ask", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
  })).json();
  div.querySelector(".coglang").textContent = r.coglang || r.error || "(no output)";
  if (r.needs_confirm) {
    const c = document.createElement("div");
    c.className = "confirm";
    c.innerHTML = `⚠️ This will modify the graph. <button>Execute</button> <button class="cancel">Cancel</button>`;
    c.querySelector("button").onclick = () => execConfirmed(r.coglang, div);
    c.querySelector(".cancel").onclick = () => c.remove();
    div.appendChild(c);
  } else {
    div.innerHTML += `<div class="result">${esc(r.result || "")}</div>`;
  }
});

document.getElementById("reset").onclick = async () => {
  await fetch("/reset", { method: "POST" });
  refreshDomain();
  addTurn(`<div class="result">(graph reset)</div>`);
};

refreshDomain();
