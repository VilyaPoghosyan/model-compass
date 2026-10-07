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

  /* Video outputs (templates/_media.html.j2, compare cards): one clip plays at a time. A mouse
     hover previews it, the play button (aria-pressed) pins it; nothing loads until it plays and
     lazy posters (data-poster) are set near the viewport. Reduced motion: no hover preview. */
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var playing = null;
  function vidOf(node) { var w = node && node.closest ? node.closest(".vid") : null; return w ? w.querySelector("video") : null; }
  function playVid(v) {
    if (playing && playing !== v) playing.pause();
    playing = v;
    if (v.dataset.poster) { v.poster = v.dataset.poster; v.removeAttribute("data-poster"); }
    var pr = v.play();
    if (pr && pr.catch) pr.catch(function () { /* autoplay refused: the button still works */ });
  }
  function mark(ev, on) {
    var v = ev.target;
    if (!v || v.tagName !== "VIDEO") return;
    if (!on) { v._pinned = false; v._hover = false; }
    var w = v.closest(".vid"), b = w && w.querySelector(".vid__play");
    if (w) w.classList.toggle("is-playing", on);
    if (b) b.setAttribute("aria-pressed", String(on));
  }
  document.addEventListener("play", function (ev) { mark(ev, true); }, true);
  document.addEventListener("pause", function (ev) { mark(ev, false); }, true);
  document.addEventListener("click", function (ev) {
    var b = ev.target && ev.target.closest ? ev.target.closest(".vid__play") : null;
    var v = b && vidOf(b);
    if (!v) return;
    if (v.paused || v._hover) { v._hover = false; v._pinned = true; playVid(v); } else v.pause();
  });
  function hover(ev, enter) {
    if (ev.pointerType !== "mouse" || (enter && calm)) return;
    var w = ev.target && ev.target.closest ? ev.target.closest(".vid") : null;
    if (!w || (ev.relatedTarget && w.contains(ev.relatedTarget))) return;
    var v = w.querySelector("video");
    if (enter && v.paused) { v._hover = true; playVid(v); }
    else if (!enter && v._hover && !v._pinned) v.pause();
  }
  document.addEventListener("pointerover", function (ev) { hover(ev, true); });
  document.addEventListener("pointerout", function (ev) { hover(ev, false); });
  if ("IntersectionObserver" in window) {
    var posterIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting || !e.target.dataset.poster) return;
        e.target.poster = e.target.dataset.poster; e.target.removeAttribute("data-poster"); posterIO.unobserve(e.target);
      });
    }, { rootMargin: "300px" });
    window.compassPosters = function (root) { (root || document).querySelectorAll("video[data-poster]").forEach(function (v) { posterIO.observe(v); }); };
  } else {
    window.compassPosters = function (root) { (root || document).querySelectorAll("video[data-poster]").forEach(function (v) { v.poster = v.dataset.poster; }); };
  }
  window.compassPosters();

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
