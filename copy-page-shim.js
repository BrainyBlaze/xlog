/*
 * Copy-page shim for the exported Mintlify site.
 *
 * Mintlify renders the contextual "Copy page" button in static exports, but the
 * exported bundle does not include hosted Markdown endpoints. The deploy
 * workflow generates those .md files from source pages, and this shim makes the
 * contextual button copy from the local Markdown export.
 */
(function () {
  "use strict";

  function markdownHref() {
    var explicit = document.querySelector('link[rel="alternate"][type="text/markdown"]');
    if (explicit && explicit.getAttribute("href")) return explicit.getAttribute("href");

    var path = window.location.pathname || "/";
    if (path === "/" || path === "") return "/index.md";
    path = path.replace(/\/$/, "");
    if (path.endsWith(".html")) path = path.slice(0, -5);
    if (path.endsWith(".md")) return path;
    return path + ".md";
  }

  function fallbackClipboard(text) {
    var textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "readonly");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
      return Promise.resolve();
    } finally {
      document.body.removeChild(textarea);
    }
  }

  function writeClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(function () {
        return fallbackClipboard(text);
      });
    }
    return fallbackClipboard(text);
  }

  function setButtonState(button, label) {
    if (!button) return;
    var span = button.querySelector("span");
    button.setAttribute("data-xlog-copy-state", label.toLowerCase());
    if (span) span.textContent = label;
  }

  function restoreButton(button) {
    window.setTimeout(function () {
      setButtonState(button, "Copy page");
    }, 1600);
  }

  function copyPage(button) {
    var href = markdownHref();
    if (button) button.setAttribute("data-xlog-copy-source", href);
    setButtonState(button, "Copying...");

    fetch(href, { headers: { Accept: "text/markdown,text/plain;q=0.9,*/*;q=0.1" } })
      .then(function (response) {
        if (!response.ok) throw new Error("Markdown export unavailable: " + response.status);
        return response.text();
      })
      .then(function (text) {
        return writeClipboard(text).then(function () {
          setButtonState(button, "Copied");
          restoreButton(button);
        });
      })
      .catch(function () {
        setButtonState(button, "Copy failed");
        restoreButton(button);
      });
  }

  window.addEventListener(
    "click",
    function (ev) {
      if (!ev.target || !ev.target.closest) return;
      var button = ev.target.closest('button[aria-label="Copy page"]');
      if (!button) return;
      ev.preventDefault();
      ev.stopImmediatePropagation();
      copyPage(button);
    },
    true
  );
})();
