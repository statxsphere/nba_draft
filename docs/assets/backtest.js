// backtest charts, drawn with observable plot

var bt = null;
var target = 'peak';

var MODELS = [
  { key: 'model', label: 'Model (stats only)', color: '--orange', dash: null },
  { key: 'draft_order', label: 'Actual draft order', color: '--foreground', dash: null },
  { key: 'model_pick', label: 'Model + draft slot', color: '--teal', dash: '4,3' },
  { key: 'v1_average', label: '2021 averaged regression', color: '--grey-line', dash: '2,3' }
];

function colors() {
  return {
    orange: cssVar('--orange'), teal: cssVar('--teal'), fg: cssVar('--foreground'), muted: cssVar('--muted-foreground'),
    grey: cssVar('--grey-line'), border: cssVar('--border'), card: cssVar('--card')
  };
}

function widthOf(id) {
  return Math.max(280, document.getElementById(id).clientWidth);
}

function put(id, chart) {
  var el = document.getElementById(id);
  el.innerHTML = '';
  el.appendChild(chart);
}

function baseStyle(c) {
  return { background: 'transparent', color: c.muted, fontFamily: 'Geist Mono, monospace', fontSize: '11px', overflow: 'visible' };
}

function headline() {
  var t = bt[target];
  var m = t.models;
  function card(label, a, b, note) {
    return '<div class="card stat"><div class="stat-label">' + label + '</div><div class="stat-value">' + a +
      (b !== null ? ' <span class="vs">vs ' + b + '</span>' : '') + '</div><div class="stat-note">' + note + '</div></div>';
  }
  var html = card('Average rank correlation', fmt(m.model.mean_spearman, 3), fmt(m.draft_order.mean_spearman, 3), 'model vs draft order, ' + t.classes + ' classes') +
    card('Classes where the model wins', t.model_beats_draft, null, 'out of ' + t.classes + ', on rank correlation') +
    card('Top 10 hits a class', fmt(m.model.mean_top10, 1), fmt(m.draft_order.mean_top10, 1), 'model vs picks 1 to 10') +
    card('Ranges that held', pct(t.coverage, 0), null, 'real outcome inside the 80% range');
  document.getElementById('headline').innerHTML = html;
  document.getElementById('target-note').textContent = target === 'peak' ?
    'Best three-season run of VORP in the first six years. Classes ' + t.years[0] + ' to ' + t.years[t.years.length - 1] + '.' :
    'Average VORP over the first two seasons. Classes ' + t.years[0] + ' to ' + t.years[t.years.length - 1] + '.';
}

function spearmanChart(c) {
  var t = bt[target];
  var rows = [];
  MODELS.forEach(function (mm) {
    t.models[mm.key].spearman.forEach(function (v, i) { rows.push({ year: t.years[i], value: v, model: mm.label }); });
  });
  var marks = [Plot.ruleY([0], { stroke: c.border })];
  MODELS.forEach(function (mm) {
    var data = rows.filter(function (r) { return r.model === mm.label; });
    var col = cssVar(mm.color);
    marks.push(Plot.line(data, { x: 'year', y: 'value', stroke: col, strokeWidth: mm.key === 'model' ? 2.6 : 1.8, strokeDasharray: mm.dash }));
    marks.push(Plot.dot(data, { x: 'year', y: 'value', fill: col, r: mm.key === 'model' ? 3.5 : 2.5 }));
  });
  marks.push(Plot.tip(rows, Plot.pointer({ x: 'year', y: 'value', title: function (d) { return d.model + '\n' + d.year + ': ' + d.value.toFixed(3); } })));
  put('chart-spearman', Plot.plot({
    width: widthOf('chart-spearman'), height: 300, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Draft class', tickFormat: 'd', ticks: Math.min(t.years.length, 12) },
    y: { label: 'Spearman', grid: true, domain: [Math.min(0, d3.min(rows, function (r) { return r.value; })), 1] },
    marks: marks
  }));
  document.getElementById('legend-spearman').innerHTML = MODELS.map(function (mm) {
    return '<span><i style="background:' + cssVar(mm.color) + '"></i>' + mm.label + '</span>';
  }).join('');
}

function hitsChart(c) {
  var t = bt[target];
  var rows = [];
  t.years.forEach(function (y, i) {
    rows.push({ year: String(y), who: 'Model', hits: t.models.model.top10[i] });
    rows.push({ year: String(y), who: 'Draft order', hits: t.models.draft_order.top10[i] });
  });
  put('chart-hits', Plot.plot({
    width: widthOf('chart-hits'), height: 260, marginLeft: 30, style: baseStyle(c),
    fx: { type: 'band', label: null, tickFormat: function (d) { return "'" + d.slice(2); } },
    x: { axis: null, domain: ['Model', 'Draft order'] },
    y: { label: 'Hits out of 10', grid: true, domain: [0, 10] },
    color: { domain: ['Model', 'Draft order'], range: [c.orange, c.grey] },
    marks: [
      Plot.barY(rows, { fx: 'year', x: 'who', y: 'hits', fill: 'who', rx: 2, tip: true, title: function (d) { return d.who + ', ' + d.year + ': ' + d.hits + ' of 10'; } }),
      Plot.ruleY([0], { stroke: c.border })
    ]
  }));
  document.getElementById('legend-hits').innerHTML = '<span><i style="background:' + c.orange + '"></i>Model</span><span><i style="background:' + c.grey + '"></i>Draft order (picks 1 to 10)</span>';
}

function calibChart(c) {
  var data = bt[target].calibration;
  var lo = d3.min(data, function (d) { return Math.min(d.pred, d.actual, d.low); });
  var hi = d3.max(data, function (d) { return Math.max(d.pred, d.actual, d.high); });
  put('chart-calib', Plot.plot({
    width: widthOf('chart-calib'), height: 260, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Projected VORP a season', domain: [lo, hi], grid: true },
    y: { label: 'Actual VORP a season', domain: [lo, hi], grid: true },
    marks: [
      Plot.line([[lo, lo], [hi, hi]], { stroke: c.muted, strokeDasharray: '4,4' }),
      Plot.ruleX(data, { x: 'pred', y1: 'low', y2: 'high', stroke: c.grey, strokeWidth: 3, strokeOpacity: 0.5, strokeLinecap: 'round' }),
      Plot.dot(data, { x: 'pred', y: 'actual', fill: c.orange, r: 5, stroke: c.card, strokeWidth: 1.5,
        tip: true, title: function (d) { return 'projected ' + d.pred.toFixed(2) + '\nactual ' + d.actual.toFixed(2) + '\n' + d.n + ' players'; } })
    ]
  }));
}

function scatterChart(c) {
  var data = bt[target].scatter;
  put('chart-scatter', Plot.plot({
    width: widthOf('chart-scatter'), height: 280, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Projected VORP a season', grid: true },
    y: { label: 'Actual VORP a season', grid: true },
    marks: [
      Plot.ruleY([0], { stroke: c.border }),
      Plot.dot(data, { x: 'pred', y: 'actual', r: 2.6, fill: function (d) { return d.pick <= 30 ? c.orange : c.teal; }, fillOpacity: 0.6 }),
      Plot.tip(data, Plot.pointer({ x: 'pred', y: 'actual', title: function (d) {
        return d.name + ' (' + d.year + ', pick ' + d.pick + ')\nprojected ' + d.pred.toFixed(2) + ', actual ' + d.actual.toFixed(2);
      } }))
    ]
  }));
}

function coverageChart(c) {
  var t = bt[target];
  var rows = t.years.map(function (y, i) { return { year: String(y), share: t.coverage_by_year[i] }; });
  put('chart-coverage', Plot.plot({
    width: widthOf('chart-coverage'), height: 260, marginLeft: 40, style: baseStyle(c),
    x: { type: 'band', label: null, tickFormat: function (d) { return "'" + d.slice(2); } },
    y: { label: 'Inside the range', domain: [0, 1], grid: true, tickFormat: '.0%' },
    marks: [
      Plot.barY(rows, { x: 'year', y: 'share', fill: c.teal, fillOpacity: 0.8, rx: 2, tip: true,
        title: function (d) { return d.year + ': ' + Math.round(100 * d.share) + '%'; } }),
      Plot.ruleY([0.8], { stroke: c.orange, strokeDasharray: '4,3' }),
      Plot.text([{ x: rows[0].year, y: 0.8 }], { x: 'x', y: 'y', text: function () { return 'target 80%'; }, dy: -8, dx: -10, textAnchor: 'start', fill: c.orange })
    ]
  }));
}

function importanceChart(c) {
  var data = bt[target].importance;
  put('chart-importance', Plot.plot({
    width: widthOf('chart-importance'), height: 30 + 24 * data.length, marginLeft: widthOf('chart-importance') < 500 ? 150 : 210, style: baseStyle(c),
    x: { label: 'Mean |SHAP|, class percentile points', grid: true },
    y: { label: null, domain: data.map(function (d) { return d.label; }) },
    marks: [
      Plot.barX(data, { y: 'label', x: 'shap', fill: c.orange, fillOpacity: 0.85, rx: 2, tip: true,
        title: function (d) { return d.label + ': ' + d.shap.toFixed(3); } }),
      Plot.ruleX([0], { stroke: c.border })
    ]
  }));
}

function yearTable() {
  var t = bt[target];
  var head = '<thead><tr><th>Class</th><th>Model</th><th>+ slot</th><th>2021 avg</th><th>Draft order</th><th>Top 10 model</th><th>Top 10 draft</th><th>1st round model</th><th>1st round draft</th></tr></thead>';
  var body = t.years.map(function (y, i) {
    var m = t.models.model.spearman[i], d = t.models.draft_order.spearman[i];
    var fr = t.first_round[i];
    return '<tr><td>' + y + '</td><td class="' + (m > d ? 'win' : '') + '">' + fmt(m, 3) + '</td><td>' + fmt(t.models.model_pick.spearman[i], 3) +
      '</td><td>' + fmt(t.models.v1_average.spearman[i], 3) + '</td><td class="' + (d >= m ? 'win' : '') + '">' + fmt(d, 3) + '</td><td>' +
      t.models.model.top10[i] + '</td><td>' + t.models.draft_order.top10[i] + '</td><td>' + fmt(fr.model, 3) + '</td><td>' + fmt(fr.draft_order, 3) + '</td></tr>';
  }).join('');
  var m = t.models;
  var foot = '<tr><td><b>Average</b></td><td>' + fmt(m.model.mean_spearman, 3) + '</td><td>' + fmt(m.model_pick.mean_spearman, 3) + '</td><td>' +
    fmt(m.v1_average.mean_spearman, 3) + '</td><td>' + fmt(m.draft_order.mean_spearman, 3) + '</td><td>' + fmt(m.model.mean_top10, 1) + '</td><td>' +
    fmt(m.draft_order.mean_top10, 1) + '</td><td>' + fmt(d3.mean(t.first_round, function (r) { return r.model; }), 3) + '</td><td>' +
    fmt(d3.mean(t.first_round, function (r) { return r.draft_order; }), 3) + '</td></tr>';
  document.getElementById('year-table').innerHTML = head + '<tbody>' + body + foot + '</tbody>';
}

function drawAll() {
  if (!bt) return;
  var c = colors();
  headline();
  spearmanChart(c);
  hitsChart(c);
  calibChart(c);
  scatterChart(c);
  coverageChart(c);
  importanceChart(c);
  yearTable();
}

document.addEventListener('DOMContentLoaded', function () {
  loadJSON('data/backtest.json').then(function (data) {
    bt = data;
    drawAll();
  }).catch(function (e) {
    document.getElementById('headline').innerHTML = '<div class="card stat">' + esc(e.message) + '</div>';
  });
  document.getElementById('target').addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
    document.querySelectorAll('#target button').forEach(function (x) { x.classList.toggle('on', x === b); });
    target = b.dataset.v;
    drawAll();
  });
  document.addEventListener('themechange', drawAll);
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', drawAll);
  }
  var timer = null;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(drawAll, 150);
  });
});
