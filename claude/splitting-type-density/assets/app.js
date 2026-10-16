/* HTML edition: popovers, compare mode, contents, sortable tables, Lean line ranges. Vanilla JS, no build step. */
(function () {
  'use strict';
  var body = document.body;
  var params = new URLSearchParams(location.search);

  /* ---------------- compare mode ---------------- */
  var cmpBtn = document.getElementById('compare-toggle');
  function setCompare(on, persist) {
    body.classList.toggle('compare', on);
    if (cmpBtn) cmpBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    if (persist) { try { localStorage.setItem('compare', on ? '1' : '0'); } catch (e) {} }
  }
  if (cmpBtn) {
    var initial = params.get('compare');
    if (initial === null) { try { initial = localStorage.getItem('compare'); } catch (e) {} }
    setCompare(initial === '1', false);
    cmpBtn.addEventListener('click', function () {
      // keep the reading position: remember the first visible row and restore it after relayout
      var anchor = firstVisibleRow();
      var offset = anchor ? anchor.getBoundingClientRect().top : 0;
      setCompare(!body.classList.contains('compare'), true);
      hidePopover(true);
      if (anchor) window.scrollBy(0, anchor.getBoundingClientRect().top - offset);
    });
  }
  function firstVisibleRow() {
    var rows = document.querySelectorAll('.row');
    var top = headerHeight();
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i].getBoundingClientRect();
      if (r.bottom > top + 8) return rows[i];
    }
    return null;
  }
  function headerHeight() {
    var h = document.querySelector('.site-header');
    return h ? h.getBoundingClientRect().height : 0;
  }

  /* ---------------- contents ---------------- */
  var tocBtn = document.getElementById('toc-toggle');
  if (tocBtn) {
    tocBtn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var open = !body.classList.contains('toc-open');
      body.classList.toggle('toc-open', open);
      tocBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (ev) {
      if (!body.classList.contains('toc-open')) return;
      var toc = document.getElementById('toc');
      if (toc && (!toc.contains(ev.target) || ev.target.tagName === 'A')) {
        body.classList.remove('toc-open');
        tocBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }
  // highlight the current section in the contents
  var tocLinks = document.querySelectorAll('.toc a');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    tocLinks.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && byId[en.target.id]) {
          tocLinks.forEach(function (a) { a.classList.remove('current'); });
          byId[en.target.id].classList.add('current');
        }
      });
    }, { rootMargin: '-10% 0px -80% 0px' });
    Object.keys(byId).forEach(function (id) { var el = document.getElementById(id); if (el) io.observe(el); });
  }

  /* ---------------- popovers ---------------- */
  var pop = document.getElementById('pop');
  var current = null, pinned = false, hideTimer = null, showTimer = null;
  var finePointer = window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  function showPopover(btn, pin) {
    if (!pop) return;
    var id = btn.getAttribute('data-entry');
    var src = document.querySelector('.lean-panel .lp-entry[data-entry="' + id + '"]');
    if (!src) return;
    clearTimeout(hideTimer);
    if (current && current !== btn) current.classList.remove('is-open');
    current = btn;
    pinned = !!pin;
    btn.classList.add('is-open');
    pop.innerHTML = '';
    var close = document.createElement('button');
    close.className = 'pop-close'; close.type = 'button'; close.setAttribute('aria-label', 'Close'); close.textContent = '×';
    close.addEventListener('click', function () { hidePopover(true); btn.focus(); });
    pop.appendChild(close);
    var clone = src.cloneNode(true);
    clone.querySelectorAll('[id]').forEach(function (el) { el.removeAttribute('id'); });
    pop.appendChild(clone);
    pop.hidden = false;
    pop.setAttribute('aria-label', 'Lean formalization of ' + (btn.title.split(':')[0] || 'statement'));
    position(btn);
  }
  function position(btn) {
    var r = btn.getBoundingClientRect();
    var vw = document.documentElement.clientWidth, vh = window.innerHeight;
    pop.style.left = '0px'; pop.style.top = '0px';
    var pw = pop.offsetWidth, ph = pop.offsetHeight;
    var left = r.left + window.scrollX - 12;
    left = Math.max(window.scrollX + 6, Math.min(left, window.scrollX + vw - pw - 6));
    var below = vh - r.bottom, above = r.top - headerHeight();
    var top = (below >= Math.min(ph, 320) || below >= above) ? r.bottom + window.scrollY + 6 : r.top + window.scrollY - ph - 6;
    pop.style.left = left + 'px'; pop.style.top = Math.max(top, window.scrollY + headerHeight() + 4) + 'px';
  }
  function hidePopover(force) {
    if (!pop || pop.hidden) return;
    if (pinned && !force) return;
    pop.hidden = true; pinned = false;
    if (current) current.classList.remove('is-open');
    current = null;
  }
  document.addEventListener('click', function (ev) {
    var btn = ev.target.closest && ev.target.closest('.lean-badge:not(.static)');
    if (btn) {
      ev.preventDefault();
      if (current === btn && pinned) { hidePopover(true); return; }
      showPopover(btn, true);
      return;
    }
    if (pop && !pop.hidden && !pop.contains(ev.target)) hidePopover(true);
  });
  if (finePointer) {
    document.addEventListener('mouseover', function (ev) {
      var btn = ev.target.closest && ev.target.closest('.lean-badge:not(.static)');
      if (btn) {
        clearTimeout(hideTimer);
        if (pinned && current !== btn) return;
        clearTimeout(showTimer);
        showTimer = setTimeout(function () { if (!pinned) showPopover(btn, false); }, 120);
      } else if (pop && pop.contains(ev.target)) {
        clearTimeout(hideTimer);
      }
    });
    document.addEventListener('mouseout', function (ev) {
      var btn = ev.target.closest && ev.target.closest('.lean-badge:not(.static)');
      var inPop = pop && pop.contains(ev.target);
      if (btn || inPop) {
        clearTimeout(showTimer);
        var to = ev.relatedTarget;
        if (to && ((pop && pop.contains(to)) || (current && current.contains(to)))) return;
        hideTimer = setTimeout(function () { hidePopover(false); }, 260);
      }
    });
  }
  document.addEventListener('keydown', function (ev) {
    if (ev.key === 'Escape') {
      var c = current;
      hidePopover(true);
      if (c) c.focus();
      if (body.classList.contains('toc-open')) body.classList.remove('toc-open');
    }
  });
  window.addEventListener('resize', function () { if (current && pop && !pop.hidden) position(current); });

  /* ---------------- sortable tables ---------------- */
  document.querySelectorAll('table.sortable').forEach(function (table) {
    var ths = table.querySelectorAll('thead th');
    ths.forEach(function (th, col) {
      th.setAttribute('tabindex', '0');
      function sort() {
        var asc = th.getAttribute('aria-sort') !== 'ascending';
        ths.forEach(function (t) { t.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        var tbody = table.tBodies[0];
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) {
          var x = key(a.cells[col]), y = key(b.cells[col]);
          var r = (typeof x === 'number' && typeof y === 'number') ? x - y : String(x).localeCompare(String(y));
          return asc ? r : -r;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      }
      th.addEventListener('click', sort);
      th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); sort(); } });
    });
  });
  function key(cell) {
    if (!cell) return '';
    var s = cell.getAttribute('data-sort');
    if (s === null) s = cell.textContent.trim();
    var n = Number(s);
    return (s !== '' && !isNaN(n)) ? n : s.toLowerCase();
  }

  /* ---------------- declaration index filter ---------------- */
  var filt = document.getElementById('decl-filter'), only = document.getElementById('decl-mapped');
  if (filt) {
    var rows = Array.prototype.slice.call(document.querySelectorAll('#decl-table tbody tr'));
    var count = document.getElementById('decl-count');
    var apply = function () {
      var q = filt.value.trim().toLowerCase(), n = 0;
      rows.forEach(function (r) {
        var ok = (!q || r.textContent.toLowerCase().indexOf(q) >= 0) && (!only.checked || r.classList.contains('mapped'));
        r.style.display = ok ? '' : 'none';
        if (ok) n++;
      });
      count.textContent = n + ' of ' + rows.length + ' shown';
    };
    filt.addEventListener('input', apply); only.addEventListener('change', apply); apply();
  }

  /* ---------------- Lean pages: #L12-L30 ranges ---------------- */
  function highlightRange() {
    var m = /^#L(\d+)(?:-L(\d+))?$/.exec(location.hash);
    document.querySelectorAll('.ln.hl').forEach(function (el) { el.classList.remove('hl'); });
    if (!m) return;
    var a = +m[1], b = m[2] ? +m[2] : a;
    for (var i = a; i <= b; i++) { var el = document.getElementById('L' + i); if (el) el.classList.add('hl'); }
    var first = document.getElementById('L' + a);
    if (first && m[2]) setTimeout(function () { first.scrollIntoView({ block: 'start' }); window.scrollBy(0, -headerHeight() - 8); }, 0);
  }
  if (document.querySelector('.lean-src')) { highlightRange(); window.addEventListener('hashchange', highlightRange); }

  /* ---------------- ?scrollto=ID (for headless screenshots, where #fragments are unreliable) ---------------- */
  if (params.has('only')) {   // show only the rows from the one containing ID (headless screenshots)
    var o = document.getElementById(params.get('only'));
    var row = o && o.closest('.row');
    if (row) { var r = row.previousElementSibling; while (r) { var p = r.previousElementSibling; r.style.display = 'none'; r = p; }
      var tocEl = document.getElementById('toc'); if (tocEl) tocEl.style.display = 'none'; }
    if (params.has('open') && o) { var bb = o.querySelector('.lean-badge'); if (bb) setTimeout(function () { showPopover(bb, true); }, 50); }
  }
  if (params.has('scrollto')) {
    var target = document.getElementById(params.get('scrollto'));
    if (target) { target.scrollIntoView({ block: 'start' }); window.scrollBy(0, -headerHeight() - 8); }
    if (params.has('open') && target) { var b = target.querySelector('.lean-badge'); if (b) showPopover(b, true); }
  }

  /* ---------------- self-test (used by build.py with a headless browser) ---------------- */
  if (params.has('selftest')) {
    window.addEventListener('load', function () {
      var vw = document.documentElement.clientWidth;
      var offenders = [];
      document.querySelectorAll('body *').forEach(function (el) {
        var r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) return;
        if (r.right > vw + 1 || r.left < -1) {
          // ignore content inside horizontally scrollable containers
          var p = el.parentElement, scrolls = false;
          while (p && p !== document.body) {
            var ox = getComputedStyle(p).overflowX;
            if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') { scrolls = true; break; }
            p = p.parentElement;
          }
          if (!scrolls && getComputedStyle(el).position !== 'fixed') offenders.push(el.tagName.toLowerCase() + (el.className && typeof el.className === 'string' ? '.' + el.className.split(' ')[0] : '') + '@' + Math.round(r.right));
        }
      });
      var badge = document.querySelector('.lean-badge:not(.static)');
      var popOk = null;
      if (badge) { badge.click(); popOk = pop && !pop.hidden && pop.querySelector('.lp-entry') !== null; hidePopover(true); }
      var out = {
        viewport: vw, scrollWidth: document.documentElement.scrollWidth,
        horizontal_overflow: document.documentElement.scrollWidth > vw + 1,
        offenders: offenders.slice(0, 15), offender_count: offenders.length,
        font_size_px: parseFloat(getComputedStyle(document.body).fontSize),
        compare: body.classList.contains('compare'), badges: document.querySelectorAll('.lean-badge:not(.static)').length,
        popover_opens: popOk, katex_errors: document.querySelectorAll('.katex-error').length
      };
      var pre = document.createElement('pre'); pre.id = 'selftest-out'; pre.textContent = JSON.stringify(out);
      document.body.appendChild(pre);
    });
  }
})();
