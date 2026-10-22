// ==============================================================================
// Getnet Multi-Agent System - Front-end Orchestration Logic
// Direction: Arthur Di Lorenzo (Visual Design Lead / FDE)
// ==============================================================================

const chatStream = document.getElementById("chat-stream");
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const userIdInput = document.getElementById("user-id-input");
const sendBtn = document.getElementById("send-btn");

// Menu Dropdown Elements
const scenariosMenuBtn = document.getElementById("scenarios-menu-btn");
const quickShortcutBtn = document.getElementById("quick-scenario-shortcut-btn");
const scenariosDropdown = document.getElementById("scenarios-dropdown");
const scenariosList = document.getElementById("scenarios-list");
const menuWrapper = document.querySelector(".menu-dropdown-wrapper");

// Toggles
const themeToggle = document.getElementById("theme-toggle");
const mobileToggle = document.getElementById("mobile-toggle");
const mobileToggleText = document.getElementById("mobile-toggle-text");
const chatViewport = document.getElementById("chat-viewport");

// Trace Elements
const traceLatency = document.getElementById("trace-latency");
const metricAgent = document.getElementById("metric-agent");
const metricTools = document.getElementById("metric-tools");
const metricSources = document.getElementById("metric-sources");
const timelineSteps = document.getElementById("timeline-steps");
const timelineStepCount = document.getElementById("timeline-step-count");
const rawJsonViewer = document.getElementById("raw-json-viewer");

// Inicialização
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initMobileToggle();
  initDropdown();
  fetchScenarios();
  setupTextareaAutoResize();
});

// Tema Claro / Escuro
function initTheme() {
  const savedTheme = localStorage.getItem("getnet_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("getnet_theme", next);
    });
  }
}

// Simulador Mobile
function initMobileToggle() {
  if (mobileToggle) {
    mobileToggle.addEventListener("click", () => {
      chatViewport.classList.toggle("mobile-mode");
      const isMobile = chatViewport.classList.contains("mobile-mode");
      mobileToggle.classList.toggle("active", isMobile);
      if (mobileToggleText) {
        mobileToggleText.textContent = isMobile ? "Visualização Desktop" : "Simular Celular";
      }
    });
  }
}

// Dropdown de Cenários do Menu Superior
function initDropdown() {
  const toggleDropdown = (e) => {
    e.stopPropagation();
    if (menuWrapper) {
      menuWrapper.classList.toggle("open");
    }
  };

  if (scenariosMenuBtn) {
    scenariosMenuBtn.addEventListener("click", toggleDropdown);
  }

  if (quickShortcutBtn) {
    quickShortcutBtn.addEventListener("click", toggleDropdown);
  }

  // Fecha ao clicar fora
  document.addEventListener("click", (e) => {
    if (menuWrapper && !menuWrapper.contains(e.target)) {
      menuWrapper.classList.remove("open");
    }
  });
}

// Lista dos 10 cenários oficiais em Português
const DEFAULT_SCENARIOS = [
  { id: 1, title: "Get Clássica vs Smart", message: "Qual é a diferença entre a Get Clássica e a Get Smart?", user_id: "cliente1988", category: "Produtos (RAG)" },
  { id: 2, title: "Previsão do Tempo", message: "Qual é a previsão do tempo em Porto Alegre para amanhã?", user_id: "cliente1988", category: "Web Search" },
  { id: 3, title: "Depósito de Vendas", message: "Quando cai o dinheiro das minhas vendas de ontem?", user_id: "cliente1988", category: "Suporte / Vendas" },
  { id: 4, title: "Conta Bancária para Pix", message: "Preciso ter conta bancária para receber minhas vendas via Pix?", user_id: "cliente1988", category: "Pix (RAG)" },
  { id: 5, title: "Máquina sem Conexão", message: "Minha maquininha não conecta na internet, o que eu faço?", user_id: "cliente1988", category: "Diagnóstico POS" },
  { id: 6, title: "Antecipação de Recebíveis", message: "Como funciona a antecipação de recebíveis com a Getnet?", user_id: "cliente1988", category: "Crédito (RAG)" },
  { id: 7, title: "Cotação do Euro", message: "Qual é a cotação do euro hoje?", user_id: "cliente1988", category: "Câmbio / Web" },
  { id: 8, title: "Transação Recusada", message: "Minha maquininha está mostrando um erro de transação recusada.", user_id: "cliente1988", category: "Diagnóstico POS" },
  { id: 9, title: "Parcelamento Crediário", message: "Em quantas parcelas posso dividir uma venda no crediário?", user_id: "cliente1988", category: "Crediário (RAG)" },
  { id: 10, title: "Vender via WhatsApp", message: "Posso vender pelo WhatsApp usando o Link de Pagamento?", user_id: "cliente1988", category: "Link de Pagamento" }
];

function renderScenarioCards(scenarios) {
  scenariosList.innerHTML = "";
  scenarios.forEach((sc) => {
    const card = document.createElement("div");
    card.className = "dropdown-scenario-card";
    card.innerHTML = `
      <div class="scenario-card-left">
        <span class="scenario-card-title">${sc.id}. ${sc.title}</span>
        <span class="scenario-card-msg">${sc.message}</span>
      </div>
      <span class="scenario-cat-badge">${sc.category}</span>
    `;
    card.addEventListener("click", () => {
      if (menuWrapper) menuWrapper.classList.remove("open");
      messageInput.value = sc.message;
      userIdInput.value = sc.user_id;
      
      // Ajusta altura do textarea e foca para envio manual
      messageInput.style.height = "auto";
      messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + "px";
      messageInput.focus();

      // Microinteração de UX: pulso suave no botão de envio
      if (sendBtn) {
        sendBtn.classList.remove("pulse-attention");
        void sendBtn.offsetWidth; // Força reflow para reiniciar animação
        sendBtn.classList.add("pulse-attention");
      }
    });
    scenariosList.appendChild(card);
  });
}

// Busca lista oficial dos 10 cenários
async function fetchScenarios() {
  // Renderiza imediatamente a lista em português
  renderScenarioCards(DEFAULT_SCENARIOS);
  
  try {
    const res = await fetch("/api/scenarios");
    if (!res.ok) return;
    const scenarios = await res.json();
    if (Array.isArray(scenarios) && scenarios.length > 0) {
      renderScenarioCards(scenarios);
    }
  } catch (err) {
    console.warn("Usando cenários padrão em português:", err);
  }
}

// Auto resize textarea
function setupTextareaAutoResize() {
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + "px";
  });
}

// Submissão do chat
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  const userId = userIdInput.value.trim() || "cliente1988";
  if (!text) return;
  
  sendMessage(text, userId);
  messageInput.value = "";
  messageInput.style.height = "auto";
});

async function sendMessage(message, userId) {
  appendUserMessage(message);
  setLoadingState(true);
  const loadingBubble = appendLoadingBubble();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message, user_id: userId })
    });

    if (!response.ok) {
      throw new Error(`Erro na API (${response.status})`);
    }

    const data = await response.json();
    loadingBubble.remove();

    appendAssistantMessage(data);
    updateTraceInspector(data);

  } catch (err) {
    loadingBubble.remove();
    appendErrorMessage(`Falha na comunicação com o orquestrador: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg-bubble user";
  row.innerHTML = `
    <div class="msg-body">
      <div class="msg-text">
        <p>${escapeHtml(text)}</p>
      </div>
      <div class="msg-time-status">
        <span class="msg-time">${getCurrentTime()}</span>
        <span class="msg-ticks">✓✓</span>
      </div>
    </div>
  `;
  chatStream.appendChild(row);
  scrollToBottom();
}

function appendAssistantMessage(data) {
  const row = document.createElement("div");
  row.className = "msg-bubble assistant";
  
  const badgeClass = getBadgeClass(data.agent_used);
  const formattedContent = formatMarkdown(data.response);
  
  const toolsHtml = (data.tools_called && data.tools_called.length > 0)
    ? `<span class="chip-tag">🛠️ ${data.tools_called.join(", ")}</span>`
    : "";
    
  const latencyHtml = `<span class="chip-tag">⚡ ${data.execution_time_ms} ms</span>`;

  row.innerHTML = `
    <div class="msg-body">
      <div class="msg-meta">
        <span class="agent-tag ${badgeClass}">${data.agent_used}</span>
        <span class="msg-timestamp">${getCurrentTime()}</span>
      </div>
      <div class="msg-text">
        ${formattedContent}
      </div>
      <div class="msg-footer">
        ${latencyHtml}
        ${toolsHtml}
      </div>
    </div>
  `;
  chatStream.appendChild(row);
  scrollToBottom();
}

function appendLoadingBubble() {
  const row = document.createElement("div");
  row.className = "msg-bubble assistant loading-bubble";
  row.innerHTML = `
    <div class="msg-body">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
      <div class="loading-hint">Consultando agentes Getnet...</div>
    </div>
  `;
  chatStream.appendChild(row);
  scrollToBottom();
  return row;
}

function appendErrorMessage(msg) {
  const row = document.createElement("div");
  row.className = "msg-bubble assistant error-bubble";
  row.innerHTML = `
    <div class="msg-body">
      <div class="msg-meta">
        <span class="agent-tag guardrail">Bloqueio / Erro</span>
        <span class="msg-timestamp">${getCurrentTime()}</span>
      </div>
      <div class="msg-text">
        <p style="color: #EF4444;">${escapeHtml(msg)}</p>
      </div>
    </div>
  `;
  chatStream.appendChild(row);
  scrollToBottom();
}

function updateTraceInspector(data) {
  traceLatency.textContent = `${data.execution_time_ms} ms`;
  metricAgent.textContent = data.agent_used || "—";
  metricTools.textContent = (data.tools_called && data.tools_called.length > 0) ? data.tools_called.join(", ") : "Nenhuma";
  metricSources.textContent = (data.sources && data.sources.length > 0) ? data.sources.join(", ") : "Nenhuma";

  if (timelineStepCount) {
    timelineStepCount.textContent = `${data.trace ? data.trace.length : 0} passos`;
  }

  // Timeline
  timelineSteps.innerHTML = "";
  if (data.trace && data.trace.length > 0) {
    data.trace.forEach((st) => {
      const card = document.createElement("div");
      let typeClass = "specialist";
      if (st.agent.includes("GUARD")) typeClass = "guardrail";
      if (st.agent.includes("ROUTER")) typeClass = "router";
      
      card.className = `trace-step-item ${typeClass}`;
      card.innerHTML = `
        <div class="step-meta">
          <span class="step-agent-name">[${st.step_id}] ${st.agent}</span>
          <span class="step-duration">${st.latency_ms} ms</span>
        </div>
        <div class="step-action-desc">${st.action}</div>
        ${st.details ? renderStepDetails(st.details) : ""}
      `;
      timelineSteps.appendChild(card);
    });
  }

  // Raw JSON
  rawJsonViewer.textContent = JSON.stringify(data, null, 2);
}

function renderStepDetails(details) {
  if (!details || typeof details !== "object") return "";
  const entries = Object.entries(details);
  if (entries.length === 0) return "";
  
  return `<div class="step-details-chips">` + entries.map(([k, v]) => {
    let valStr = "";
    if (v === null || v === undefined) {
      valStr = "null";
    } else if (typeof v === "number" && k.toLowerCase().includes("confidence")) {
      valStr = `${Math.round(v * 100)}%`;
    } else if (Array.isArray(v)) {
      valStr = v.length > 0 ? v.join(", ") : "nenhum";
    } else if (typeof v === "object") {
      valStr = JSON.stringify(v);
    } else {
      valStr = String(v);
    }
    const cleanKey = k.replace(/_/g, " ").toUpperCase();
    return `<span class="step-detail-chip"><strong class="chip-k">${escapeHtml(cleanKey)}:</strong> <span class="chip-v">${escapeHtml(valStr)}</span></span>`;
  }).join("") + `</div>`;
}

function getBadgeClass(agent) {
  if (!agent) return "orchestrator";
  if (agent.includes("CHAIN")) return "escalation";
  if (agent.includes("KNOWLEDGE")) return "knowledge";
  if (agent.includes("CUSTOMER")) return "support";
  if (agent.includes("SEARCH")) return "search";
  if (agent.includes("ESCALATION")) return "escalation";
  if (agent.includes("GUARD")) return "guardrail";
  return "orchestrator";
}

function setLoadingState(loading) {
  sendBtn.disabled = loading;
  if (loading) {
    sendBtn.innerHTML = `<span class="btn-spinner"></span>`;
  } else {
    sendBtn.innerHTML = `
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
    `;
  }
}

function scrollToBottom() {
  chatStream.scrollTop = chatStream.scrollHeight;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  
  // Negrito e Código Inline
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`(.*?)`/g, "<code>$1</code>");

  // Divide em blocos separados por quebras de linha duplas
  const rawBlocks = html.split(/\n\s*\n/);

  const formattedBlocks = rawBlocks.map(block => {
    const trimmed = block.trim();
    if (!trimmed) return "";
    const lines = trimmed.split(/\n/);
    
    // Lista com marcadores (bullet list)
    const isBulletList = lines.every(l => /^[•\-\*]\s+/.test(l.trim()));
    if (isBulletList) {
      const items = lines.map(l => `<li>${l.trim().replace(/^[•\-\*]\s+/, "")}</li>`).join("");
      return `<ul>${items}</ul>`;
    }

    // Lista numerada ordenada (ordered list)
    const isNumberedList = lines.every(l => /^[0-9]+\.\s+/.test(l.trim()));
    if (isNumberedList) {
      const items = lines.map(l => `<li>${l.trim().replace(/^[0-9]+\.\s+/, "")}</li>`).join("");
      return `<ol>${items}</ol>`;
    }

    // Parágrafo regular
    return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
  });

  return formattedBlocks.filter(Boolean).join("");
}
