/*
 * Self-hosted search shim for the exported Mintlify site.
 *
 * Static exports have no Mintlify cloud search backend, so the built-in
 * search modal shows a login prompt. This shim intercepts the search
 * triggers (search bar, Ctrl/Cmd-K) and opens a Pagefind-powered modal
 * instead. The Pagefind index is built over the exported HTML at deploy
 * time (see .github/workflows/docs.yml).
 *
 * Kept outside docs/ on purpose: Mintlify inlines any CSS it finds
 * in its content directory into every page (see the rustdoc custom-CSS
 * incident); these assets are copied into the bundle after export.
 */
(function () {
  "use strict";

  var overlay = null;
  var uiLoaded = false;

  function isDark() {
    return document.documentElement.classList.contains("dark");
  }

  function buildOverlay() {
    overlay = document.createElement("div");
    overlay.id = "xlog-search-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Search documentation");
    overlay.innerHTML =
      '<div id="xlog-search-backdrop"></div>' +
      '<div id="xlog-search-panel"><div id="xlog-pagefind"></div>' +
      '<div id="xlog-search-hint">Esc to close</div></div>';
    document.body.appendChild(overlay);

    overlay.querySelector("#xlog-search-backdrop").addEventListener("click", closeOverlay);
  }

  function focusInput() {
    var input = overlay && overlay.querySelector(".pagefind-ui__search-input");
    if (input) {
      input.focus();
      input.select();
    }
  }

  function mountPagefind() {
    if (uiLoaded) return Promise.resolve();
    return new Promise(function (resolve, reject) {
      var css = document.createElement("link");
      css.rel = "stylesheet";
      css.href = "/pagefind/pagefind-ui.css";
      document.head.appendChild(css);
      var script = document.createElement("script");
      script.src = "/pagefind/pagefind-ui.js";
      script.onload = function () {
        /* global PagefindUI */
        new PagefindUI({
          element: "#xlog-pagefind",
          showSubResults: true,
          showImages: false,
          autofocus: true,
        });
        uiLoaded = true;
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  function openOverlay() {
    if (!overlay) buildOverlay();
    overlay.classList.toggle("xlog-dark", isDark());
    overlay.classList.add("open");
    document.documentElement.style.overflow = "hidden";
    mountPagefind().then(function () {
      requestAnimationFrame(focusInput);
    });
  }

  function closeOverlay() {
    if (!overlay) return;
    overlay.classList.remove("open");
    document.documentElement.style.overflow = "";
  }

  function isOpen() {
    return overlay && overlay.classList.contains("open");
  }

  function matchesTrigger(target) {
    if (!target || !target.closest) return false;
    return !!target.closest(
      '#search-bar-entry, #search-bar-entry-mobile, [aria-label="Open search"]'
    );
  }

  window.addEventListener(
    "click",
    function (ev) {
      if (matchesTrigger(ev.target)) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        openOverlay();
      }
    },
    true
  );

  window.addEventListener(
    "keydown",
    function (ev) {
      var key = (ev.key || "").toLowerCase();
      if ((ev.ctrlKey || ev.metaKey) && key === "k") {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        if (isOpen()) closeOverlay();
        else openOverlay();
        return;
      }
      if (key === "escape" && isOpen()) {
        ev.preventDefault();
        ev.stopImmediatePropagation();
        closeOverlay();
      }
    },
    true
  );
})();
