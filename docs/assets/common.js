// shared bits: theme switch, number formatting, loading json

function savedTheme() {
  try { return localStorage.getItem('theme'); } catch (e) { return null; }
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('theme', theme); } catch (e) {}
  document.dispatchEvent(new CustomEvent('themechange'));
}

function currentTheme() {
  var t = document.documentElement.getAttribute('data-theme');
  if (t) return t;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

(function () {
  var t = savedTheme();
  if (t) document.documentElement.setAttribute('data-theme', t);
})();

document.addEventListener('DOMContentLoaded', function () {
  var btn = document.getElementById('theme-btn');
  if (btn) {
    btn.addEventListener('click', function () {
      setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });
  }
});

function loadJSON(path) {
  return fetch(path).then(function (r) {
    if (!r.ok) throw new Error('could not load ' + path);
    return r.json();
  });
}

function fmt(x, digits) {
  if (x === null || x === undefined || isNaN(x)) return '–';
  return Number(x).toFixed(digits === undefined ? 1 : digits);
}

function signed(x, digits) {
  if (x === null || x === undefined || isNaN(x)) return '–';
  var s = Number(x).toFixed(digits === undefined ? 1 : digits);
  return x > 0 ? '+' + s : s.replace('-', '−');
}

function pct(x, digits) {
  if (x === null || x === undefined || isNaN(x)) return '–';
  return (100 * x).toFixed(digits === undefined ? 1 : digits) + '%';
}

function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// small svg line for a career, one point per season
function sparkline(values, color) {
  if (!values || !values.length) return '<svg class="spark" viewBox="0 0 90 30"></svg>';
  var lo = Math.min(-1, Math.min.apply(null, values));
  var hi = Math.max(3, Math.max.apply(null, values));
  var n = Math.max(values.length - 1, 1);
  var pts = values.map(function (v, i) {
    var x = 3 + (84 * i) / n;
    var y = 27 - (24 * (v - lo)) / (hi - lo);
    return x.toFixed(1) + ',' + y.toFixed(1);
  });
  var zero = 27 - (24 * (0 - lo)) / (hi - lo);
  return '<svg class="spark" viewBox="0 0 90 30" aria-hidden="true">' +
    '<line x1="0" x2="90" y1="' + zero.toFixed(1) + '" y2="' + zero.toFixed(1) + '" stroke="var(--border)" stroke-width="1"/>' +
    '<polyline points="' + pts.join(' ') + '" fill="none" stroke="' + color + '" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>' +
    '</svg>';
}
