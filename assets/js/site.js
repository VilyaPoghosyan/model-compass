/* Shared behaviour: outbound click counting (never blocks navigation). */
(function () {
  function count(path) {
    try {
      if (window.goatcounter && typeof window.goatcounter.count === "function") {
        window.goatcounter.count({ path: path, event: true });
      }
    } catch (e) { /* analytics must never break a click */ }
  }
  document.addEventListener("click", function (ev) {
    var a = ev.target && ev.target.closest ? ev.target.closest("a[data-out]") : null;
    if (a && a.dataset.out) count("out/" + a.dataset.out);
  }, true);
  window.compassCount = count;
})();
