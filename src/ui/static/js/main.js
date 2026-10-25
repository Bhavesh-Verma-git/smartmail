/**
 * SmartMail Testing UI - Main JavaScript
 * Handles CSV pre-upload validation preview, confirmation modals, draft editing, and interactive controls.
 */

// Automatically attach CSRF token to state-changing fetch requests
(function() {
  const originalFetch = window.fetch;
  window.fetch = function(url, options = {}) {
    const method = (options.method || "GET").toUpperCase();
    if (["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
      const csrfMeta = document.querySelector('meta[name="csrf-token"]');
      const token = csrfMeta ? csrfMeta.getAttribute("content") : "";
      if (token) {
        if (!options.headers) {
          options.headers = {};
        }
        if (options.headers instanceof Headers) {
          if (!options.headers.has("X-CSRFToken")) {
            options.headers.set("X-CSRFToken", token);
          }
        } else if (Array.isArray(options.headers)) {
          if (!options.headers.some(h => h[0].toLowerCase() === "x-csrftoken")) {
            options.headers.push(["X-CSRFToken", token]);
          }
        } else {
          if (!options.headers["X-CSRFToken"]) {
            options.headers["X-CSRFToken"] = token;
          }
        }
      }
    }
    return originalFetch(url, options);
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  initMobileSidebar();
  initDropzone();
  initModeSelectors();
  initTabs();
  initModals();
});

function initMobileSidebar() {
  const menuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  if (!menuBtn || !sidebar || !backdrop) return;

  menuBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    backdrop.classList.toggle("active");
  });

  backdrop.addEventListener("click", () => {
    sidebar.classList.remove("open");
    backdrop.classList.remove("active");
  });
}

// ---------------------------------------------------------------------------
// CSV File Dropzone & Validation Preview
// ---------------------------------------------------------------------------
function initDropzone() {
  const dropzone = document.getElementById("csvDropzone");
  const fileInput = document.getElementById("csvFileInput");
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(fileInput.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFileSelected(fileInput.files[0]);
    }
  });
}

function handleFileSelected(file) {
  const fileLabel = document.getElementById("selectedFileName");
  if (fileLabel) {
    fileLabel.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }

  // Request pre-upload validation preview
  const formData = new FormData();
  formData.append("csv_file", file);

  const previewContainer = document.getElementById("previewContainer");
  const previewLoading = document.getElementById("previewLoading");
  if (previewLoading) previewLoading.style.display = "block";
  if (previewContainer) previewContainer.style.display = "none";

  fetch("/import/preview", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      if (previewLoading) previewLoading.style.display = "none";
      if (!data.success) {
        alert("Failed to parse CSV: " + (data.error || "Unknown error"));
        return;
      }
      renderPreview(data);
    })
    .catch((err) => {
      if (previewLoading) previewLoading.style.display = "none";
      alert("Error inspecting CSV: " + err.message);
    });
}

function renderPreview(data) {
  const previewContainer = document.getElementById("previewContainer");
  const previewTbody = document.getElementById("previewTbody");
  const validCountEl = document.getElementById("previewValidCount");
  const errorCountEl = document.getElementById("previewErrorCount");
  const confirmBtn = document.getElementById("openConfirmBtn");

  if (!previewContainer || !previewTbody) return;

  validCountEl.textContent = data.valid_count;
  errorCountEl.textContent = data.error_count;

  previewTbody.innerHTML = "";

  // Render valid contacts
  data.contacts.forEach((c) => {
    const tr = document.createElement("tr");
    const domainNote = c.is_free_provider ? '<span class="badge badge-neutral" style="font-size:0.7rem;margin-left:4px;">Free Provider</span>' : '';
    tr.innerHTML = `
      <td>${c.row_number}</td>
      <td><strong class="cell-email" title="${escapeHtml(c.email)}">${escapeHtml(c.email)}</strong>${domainNote}</td>
      <td><div class="cell-company">${escapeHtml(c.company_name || "—")}</div></td>
      <td>${escapeHtml(c.contact_name || "—")}</td>
      <td><div class="cell-subtext">${escapeHtml(c.role_or_job_url || "—")}</div></td>
      <td><span class="badge badge-proceed"><span class="badge-dot"></span>Valid</span></td>
    `;
    previewTbody.appendChild(tr);
  });

  // Render invalid / errored rows
  data.errors.forEach((e) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${e.row_number}</td>
      <td><span style="color:#f87171;">${escapeHtml(e.raw_row.email || e.raw_row[0] || "Empty")}</span></td>
      <td>${escapeHtml(e.raw_row.company_name || "—")}</td>
      <td>${escapeHtml(e.raw_row.contact_name || "—")}</td>
      <td>${escapeHtml(e.raw_row.role_or_job_url || "—")}</td>
      <td><span class="badge badge-error" title="${escapeHtml(e.reason)}">${escapeHtml(e.reason)}</span></td>
    `;
    previewTbody.appendChild(tr);
  });

  previewContainer.style.display = "block";
  if (confirmBtn) {
    confirmBtn.disabled = data.valid_count === 0;
  }
}

// ---------------------------------------------------------------------------
// Workflow Mode Selectors
// ---------------------------------------------------------------------------
function initModeSelectors() {
  const cards = document.querySelectorAll(".mode-card");
  const modeInput = document.getElementById("selectedWorkflowMode");

  cards.forEach((card) => {
    card.addEventListener("click", () => {
      cards.forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      const mode = card.getAttribute("data-mode");
      if (modeInput) modeInput.value = mode;

    });
  });
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const parent = btn.closest(".tabs-wrapper") || document;
      parent.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      parent.querySelectorAll(".tab-pane").forEach((p) => (p.style.display = "none"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const target = document.getElementById(targetId);
      if (target) target.style.display = "block";
    });
  });
}

// ---------------------------------------------------------------------------
// Modals
// ---------------------------------------------------------------------------
function initModals() {
  // Modal openers
  document.querySelectorAll("[data-open-modal]").forEach((trigger) => {
    trigger.addEventListener("click", () => {
      const modalId = trigger.getAttribute("data-open-modal");
      openModal(modalId);
    });
  });

  // Modal closers
  document.querySelectorAll("[data-close-modal]").forEach((trigger) => {
    trigger.addEventListener("click", () => {
      const modal = trigger.closest(".modal-overlay");
      if (modal) modal.classList.remove("open");
    });
  });
}

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.add("open");
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.remove("open");
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
