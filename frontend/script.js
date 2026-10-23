/**
 * Orient BlackSwan History Catalogue RAG Bot - Frontend Logic
 * ============================================================
 * Vanilla JavaScript managing:
 * - Real-time system health checks via GET /api/health
 * - Form interactions, character counting, and sample queries
 * - Submitting queries to POST /api/query
 * - Rendering answers, physical page citations, and verbatim excerpts
 */

document.addEventListener("DOMContentLoaded", () => {
  // Element references
  const queryForm = document.getElementById("rag-query-form");
  const questionInput = document.getElementById("question-input");
  const charCounter = document.getElementById("char-counter");
  const submitBtn = document.getElementById("submit-btn");
  const chipButtons = document.querySelectorAll(".chip-btn");

  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const refreshStatusBtn = document.getElementById("refresh-status-btn");

  const responseCard = document.getElementById("response-card");
  const responseStatusBadge = document.getElementById("response-status-badge");
  const echoQuestionText = document.getElementById("echo-question-text");
  const answerContent = document.getElementById("answer-content");
  const citationsContainer = document.getElementById("citations-container");
  const citationsList = document.getElementById("citations-list");

  // Diagnostics elements
  const diagFastapi = document.getElementById("diag-fastapi");
  const diagFastapiVal = document.getElementById("diag-fastapi-val");
  const diagPdf = document.getElementById("diag-pdf");
  const diagPdfVal = document.getElementById("diag-pdf-val");
  const diagChroma = document.getElementById("diag-chroma");
  const diagChromaVal = document.getElementById("diag-chroma-val");
  const diagGemini = document.getElementById("diag-gemini");
  const diagGeminiVal = document.getElementById("diag-gemini-val");

  /**
   * Fetch system health status from FastAPI /api/health endpoint
   */
  async function checkHealth() {
    statusDot.className = "status-dot checking";
    statusText.textContent = "Checking backend...";

    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();

      // Update Header Status
      statusDot.className = "status-dot online";
      statusText.textContent = `Online (v${data.version})`;

      // Update Diagnostics
      setDiagState(diagFastapi, diagFastapiVal, true, `Connected (v${data.version})`);

      const pdfReady = data.checks?.catalogue_pdf_present;
      setDiagState(
        diagPdf,
        diagPdfVal,
        pdfReady,
        pdfReady ? "Present in data/" : "Missing (run ingest)"
      );

      const chromaReady = data.checks?.vector_database_indexed;
      setDiagState(
        diagChroma,
        diagChromaVal,
        chromaReady,
        chromaReady ? "Indexed" : "Pending Ingestion"
      );

      const geminiReady = data.checks?.gemini_api_key_configured;
      setDiagState(
        diagGemini,
        diagGeminiVal,
        geminiReady,
        geminiReady ? "API Key Configured" : "Missing in .env"
      );
    } catch (err) {
      statusDot.className = "status-dot offline";
      statusText.textContent = "Backend Offline";

      setDiagState(diagFastapi, diagFastapiVal, false, "Offline (Start FastAPI)");
      setDiagState(diagPdf, diagPdfVal, false, "Unknown");
      setDiagState(diagChroma, diagChromaVal, false, "Unknown");
      setDiagState(diagGemini, diagGeminiVal, false, "Unknown");
    }
  }

  function setDiagState(itemEl, valEl, isReady, message) {
    if (!itemEl || !valEl) return;
    itemEl.classList.remove("ready", "pending", "error");
    itemEl.classList.add(isReady ? "ready" : "pending");
    valEl.textContent = message;
  }

  // Initial health check & refresh button handler
  checkHealth();
  refreshStatusBtn.addEventListener("click", checkHealth);

  /**
   * Character count listener
   */
  function updateCharCount() {
    const len = questionInput.value.length;
    charCounter.textContent = `${len} / 500`;
  }
  questionInput.addEventListener("input", updateCharCount);

  /**
   * Sample query chips handler
   */
  chipButtons.forEach((chip) => {
    chip.addEventListener("click", () => {
      const sampleText = chip.getAttribute("data-query");
      if (sampleText) {
        questionInput.value = sampleText;
        updateCharCount();
        questionInput.focus();
      }
    });
  });

  /**
   * Keyboard shortcut: Ctrl + Enter or Cmd + Enter to submit
   */
  questionInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      queryForm.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  /**
   * Handle form submission and API call to /api/query
   */
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = questionInput.value.trim();

    if (!query) {
      questionInput.focus();
      return;
    }

    // Set UI to loading state
    submitBtn.disabled = true;
    submitBtn.classList.add("loading");
    submitBtn.querySelector(".btn-text").textContent = "Searching...";

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const result = await response.json();
      displayResult(result);
    } catch (error) {
      displayError(query, error.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove("loading");
      submitBtn.querySelector(".btn-text").textContent = "Query Catalogue";
    }
  });

  /**
   * Render successful or "Not found" response from backend
   */
  function displayResult(result) {
    echoQuestionText.textContent = result.question;
    responseCard.classList.remove("hidden");

    const isNotFound = result.answer.trim().toLowerCase() === "not found" || result.found === false;

    if (isNotFound) {
      responseStatusBadge.className = "badge badge-notfound";
      responseStatusBadge.textContent = "Document Evidence Missing";

      answerContent.className = "answer-content not-found";
      answerContent.textContent = "Not found";

      citationsContainer.style.display = "none";
    } else {
      responseStatusBadge.className = "badge badge-success";
      responseStatusBadge.textContent = "Grounded Answer";

      answerContent.className = "answer-content";
      answerContent.textContent = result.answer;

      // Render citations
      citationsList.innerHTML = "";
      if (result.citations && result.citations.length > 0) {
        citationsContainer.style.display = "block";
        result.citations.forEach((cit) => {
          const citCard = document.createElement("div");
          citCard.className = "citation-item";

          const metaDiv = document.createElement("div");
          metaDiv.className = "citation-meta";

          const badge = document.createElement("span");
          badge.className = "page-badge";
          badge.textContent = `PDF Page ${cit.page}`;

          metaDiv.appendChild(badge);

          const excerptP = document.createElement("blockquote");
          excerptP.className = "citation-excerpt";
          excerptP.textContent = `“${cit.excerpt}”`;

          citCard.appendChild(metaDiv);
          citCard.appendChild(excerptP);
          citationsList.appendChild(citCard);
        });
      } else {
        citationsContainer.style.display = "none";
      }
    }

    // Smooth scroll down to response
    responseCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /**
   * Render error notification
   */
  function displayError(query, errorMessage) {
    echoQuestionText.textContent = query;
    responseCard.classList.remove("hidden");

    responseStatusBadge.className = "badge badge-notfound";
    responseStatusBadge.textContent = "API Error";

    answerContent.className = "answer-content not-found";
    answerContent.textContent = `Unable to complete query: ${errorMessage}`;

    citationsContainer.style.display = "none";
    responseCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
