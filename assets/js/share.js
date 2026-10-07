/* Share controls + share analytics. Markup: templates/_share.html.j2. Native share sheet on touch
   devices, otherwise X, LinkedIn, WhatsApp and Copy link. GoatCounter events (docs/site.md 2a):
   share/open/<task>/<from>, share/<channel>/<task>/<from>, share/arrive/<task>/<channel>. */
(function (root) {
  "use strict";
  var MAX_Q = 200;   // longest request carried in a shared URL
  var QUOTE = 80;    // longest request quoted in the share text
  var CHANNELS = ["x", "linkedin", "whatsapp", "copy", "native"];

  /* Text only: no control characters, single spaces, length-limited. */
  function clean(s, max) {
    return String(s == null ? "" : s).replace(/[\u0000-\u001f\u007f]+/g, " ").replace(/\s+/g, " ").trim().slice(0, max || MAX_Q);
  }
  function quote(s) {
    var t = clean(s);
    if (t.length <= QUOTE) return t;
    return t.slice(0, QUOTE - 1).replace(/\s+\S*$/, "").replace(/[\s.,;:!?-]+$/, "") + "…";
  }
  function shareUrl(origin, base, path, q, channel) {
    var u = new URL((base || "") + "/" + String(path || "").replace(/^\/+/, ""), origin);
    var text = clean(q);
    if (text) u.searchParams.set("q", text);
    u.searchParams.set("ref", "share-" + channel);
    return u.toString();
  }
  function channelHref(channel, text, link) {
    var e = encodeURIComponent;
    if (channel === "x") return "https://x.com/intent/tweet?text=" + e(text) + "&url=" + e(link);
    if (channel === "linkedin") return "https://www.linkedin.com/sharing/share-offsite/?url=" + e(link);
    if (channel === "whatsapp") return "https://wa.me/?text=" + e(text + " " + link);
    return link;
  }
  /* "?ref=share-x" -> "x"; anything else -> "". */
  function arrivalChannel(search) {
    var m = /^share-([a-z]+)$/.exec(new URLSearchParams(search || "").get("ref") || "");
    return m && CHANNELS.indexOf(m[1]) >= 0 ? m[1] : "";
  }

  root.compassShare = { clean: clean, quote: quote, shareUrl: shareUrl, channelHref: channelHref, arrivalChannel: arrivalChannel, MAX_Q: MAX_Q };
  if (typeof document === "undefined") return;

  var base = document.documentElement.dataset.base || "";
  function count(path) { if (root.compassCount) root.compassCount(path); }
  function tag(box) { return (box.dataset.shareTask || "none") + "/" + (box.dataset.shareFrom || "page"); }
  function urlFor(box, channel) { return shareUrl(location.origin, base, box.dataset.sharePath, box.dataset.shareQ, channel); }
  function useNative() {
    return typeof navigator.share === "function" && root.matchMedia && root.matchMedia("(pointer: coarse)").matches;
  }
  function refresh(box) {
    box.querySelectorAll("a[data-share-channel]").forEach(function (a) {
      a.href = channelHref(a.dataset.shareChannel, box.dataset.shareText || document.title, urlFor(box, a.dataset.shareChannel));
    });
  }
  function setOpen(box, open) {
    var btn = box.querySelector("[data-share-toggle]");
    var panel = box.querySelector(".share__panel");
    if (open) { refresh(box); count("share/open/" + tag(box)); }
    panel.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
  }
  function copyText(text) {
    var fallback = function () {
      var prev = document.activeElement;  // ta.select() moves focus; give it back afterwards
      var ta = document.createElement("textarea");
      ta.value = text; ta.setAttribute("readonly", ""); ta.style.cssText = "position:fixed;top:0;opacity:0";
      document.body.appendChild(ta); ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      ta.remove();
      if (prev && typeof prev.focus === "function") prev.focus();
      return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
    };
    if (navigator.clipboard && root.isSecureContext) return navigator.clipboard.writeText(text).catch(fallback);
    return fallback();
  }
  function copied(box, btn, link) {
    var status = box.querySelector(".share__status");
    var label = btn.dataset.label || (btn.dataset.label = btn.textContent);
    copyText(link).then(function () {
      btn.textContent = "Copied";
      if (status) status.textContent = "Link copied";
    }, function () {
      root.prompt("Copy this link", link);
    });
    clearTimeout(btn._t);
    btn._t = setTimeout(function () { btn.textContent = label; if (status) status.textContent = ""; }, 2000);
  }

  document.addEventListener("click", function (ev) {
    var t = ev.target && ev.target.closest ? ev.target.closest("[data-share-toggle],[data-share-channel]") : null;
    var box = t && t.closest("[data-share]");
    if (!box) return;
    if (t.hasAttribute("data-share-toggle")) {
      var open = t.getAttribute("aria-expanded") === "true";
      if (!open && useNative()) {
        count("share/open/" + tag(box));
        navigator.share({ title: document.title, text: box.dataset.shareText || "", url: urlFor(box, "native") })
          .then(function () { count("share/native/" + tag(box)); })
          .catch(function (e) { if (!e || e.name !== "AbortError") setOpen(box, true); });
        return;
      }
      setOpen(box, !open);
      return;
    }
    var channel = t.dataset.shareChannel;
    count("share/" + channel + "/" + tag(box));
    if (channel === "copy") { ev.preventDefault(); copied(box, t, urlFor(box, "copy")); }
    else refresh(box);  // the click still opens the (fresh) link in a new tab
  });
  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "Escape") return;
    var box = ev.target && ev.target.closest ? ev.target.closest("[data-share]") : null;
    if (!box || box.querySelector(".share__panel").hidden) return;
    setOpen(box, false);
    box.querySelector("[data-share-toggle]").focus();
  });

  /* Task page opened from a shared home result: echo the request (text only). */
  var params = new URLSearchParams(location.search);
  var asked = document.getElementById("share-asked");
  var q = clean(params.get("q"));
  if (asked && q) {
    asked.querySelector("[data-asked-text]").textContent = q;
    asked.hidden = false;
  }

  /* Arrival from a shared link: one event once GoatCounter is ready, then drop ref from the URL
     (GoatCounter's own pageview has read it by then) so a re-shared address bar is not miscounted. */
  var channel = arrivalChannel(location.search);
  if (!channel) return;
  var tries = 0;
  var send = function () {
    var gc = root.goatcounter;
    if (!(gc && typeof gc.count === "function") && tries++ < 20) { setTimeout(send, 500); return; }
    var box = document.querySelector("[data-share]");
    var task = (box && box.dataset.shareTask) || (box && box.dataset.shareFrom) || "home";
    count("share/arrive/" + task + "/" + channel);
    try {
      var u = new URL(location.href);
      u.searchParams.delete("ref");
      history.replaceState(history.state, "", u.pathname + u.search + u.hash);
    } catch (e) { /* keep the URL as it is */ }
  };
  if (document.readyState === "complete") send(); else root.addEventListener("load", send);
})(typeof window !== "undefined" ? window : globalThis);
