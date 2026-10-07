/* Outbound click counting (never blocks navigation) + TOC state. */
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

  /* Text pages: aria-current on the TOC link of the section in view. */
  var toc = document.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    var links = {};
    toc.querySelectorAll('a[href^="#"]').forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var sections = Object.keys(links).map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });
      var first = sections.filter(function (sec) { return visible[sec.id]; })[0];
      if (!first) return;
      sections.forEach(function (sec) {
        if (sec === first) links[sec.id].setAttribute("aria-current", "true");
        else links[sec.id].removeAttribute("aria-current");
      });
    }, { rootMargin: "-10% 0px -60% 0px" });
    sections.forEach(function (sec) { io.observe(sec); });
  }
})();
