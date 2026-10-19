// backtest charts, drawn with observable plot

var bt = null;
var target = 'peak';

var MODELS = [
  { key: 'model', label: 'Model (stats only)', color: '--orange', dash: null },
  { key: 'draft_order', label: 'Actual draft order', color: '--foreground', dash: null },
  { key: 'model_pick', label: 'Model + draft slot', color: '--teal', dash: '4,3' },
  { key: 'v3', label: 'V3 settings', color: '--grey-line', dash: '2,3' }
];

// everything else that was tried, shown in the kept or dropped table
var HEADS = [
  { key: 'model_gbm', label: 'LightGBM on its own' },
  { key: 'model_ridge', label: 'Ridge on its own' },
  { key: 'model_lambdarank', label: 'LightGBM lambdarank head on its own' },
  { key: 'model_rank_blend', label: 'Blend with the lambdarank head added' },
  { key: 'model_hurdle', label: 'Two-part model: reaches the league × impact' },
  { key: 'v1_average', label: 'My 2021 averaged regression' }
];

function colors() {
  return {
    orange: cssVar('--orange'), teal: cssVar('--teal'), fg: cssVar('--foreground'), muted: cssVar('--muted-foreground'),
    grey: cssVar('--grey-line'), border: cssVar('--border'), card: cssVar('--card')
  };
}

function T() { return bt.targets[target]; }

function widthOf(id) {
  return Math.max(260, document.getElementById(id).clientWidth);
}

function put(id, chart) {
  var el = document.getElementById(id);
  el.innerHTML = '';
  el.appendChild(chart);
}

function baseStyle(c) {
  return { background: 'transparent', color: c.muted, fontFamily: 'Geist Mono, monospace', fontSize: '11px', overflow: 'visible' };
}

function unit() {
  return T().rating_scale ? 'VORP a season' : 'points per 100 possessions';
}

function buttons() {
  var keys = Object.keys(bt.targets);
  document.getElementById('target').innerHTML = keys.map(function (k) {
    return '<button data-v="' + k + '" class="' + (k === target ? 'on' : '') + '">' + esc(bt.targets[k].name) + '</button>';
  }).join('');
}

function headline() {
  var t = T();
  var m = t.models;
  function card(label, a, b, note) {
    return '<div class="card stat"><div class="stat-label">' + label + '</div><div class="stat-value">' + a +
      (b !== null ? ' <span class="vs">vs ' + b + '</span>' : '') + '</div><div class="stat-note">' + note + '</div></div>';
  }
  var html = card('Stats only vs draft order', fmt(m.model.mean_spearman, 3), fmt(m.draft_order.mean_spearman, 3),
      'average rank correlation, ' + t.classes + ' classes; wins ' + t.model_beats_draft) +
    card('Stats + draft slot', fmt(m.model_pick.mean_spearman, 3), fmt(m.draft_order.mean_spearman, 3),
      'beats the draft order in ' + t.pick_beats_draft + ' of ' + t.classes + ' classes') +
    card('Top 10 hits a class', fmt(m.model.mean_top10, 1), fmt(m.draft_order.mean_top10, 1), 'stats only vs picks 1 to 10') +
    card('80% ranges that held', pct(m.model.coverage, 0), null, 'real outcome inside the range, stats only');
  document.getElementById('headline').innerHTML = html;
  var notes = {
    peak: 'Best three-season run of VORP in the first six years. ',
    imm: 'Average VORP over the first two seasons. ',
    bpm: 'Best three-season run of BPM in seasons two to six, minutes pooled and pulled toward replacement. ',
    rapm: 'Best three-season window of my own plus-minus in seasons two to six. ',
    imm_rapm: 'My own plus-minus over the first two seasons. '
  };
  document.getElementById('target-note').textContent = (notes[target] || '') + 'Classes ' + t.years[0] + ' to ' + t.years[t.years.length - 1] + '.';
}

function beforeTable() {
  var t = T(), m = t.models;
  var pub = bt.v3_published[target];
  var head = '<thead><tr><th>Version</th><th>Spearman</th><th>Top 10</th><th>NDCG@10</th><th>80% coverage</th><th>Range width</th></tr></thead>';
  var rows = [];
  if (pub) {
    rows.push('<tr><td>V3 as published <span class="muted">(old bust coding, tuned on this backtest)</span></td><td>' + fmt(pub.model, 3) + '</td><td>' + fmt(pub.top10, 1) +
      '</td><td>–</td><td>' + pct(pub.coverage, 0) + '</td><td>–</td></tr>');
    rows.push('<tr><td>Draft order, old bust coding</td><td>' + fmt(pub.draft_order, 3) + '</td><td>' + fmt(pub.draft_top10, 1) + '</td><td>–</td><td>–</td><td>–</td></tr>');
  }
  function row(label, x, cls) {
    return '<tr class="' + (cls || '') + '"><td>' + label + '</td><td>' + fmt(x.mean_spearman, 3) + '</td><td>' + fmt(x.mean_top10, 1) + '</td><td>' +
      fmt(x.mean_ndcg10, 3) + '</td><td>' + (x.coverage === null ? '–' : pct(x.coverage, 0)) + '</td><td>' + (x.width === null ? '–' : fmt(x.width, 2)) + '</td></tr>';
  }
  rows.push(row('V3 settings and inputs, new target', m.v3));
  rows.push(row('<b>V4, stats only</b>', m.model));
  rows.push(row('<b>V4, stats + draft slot</b>', m.model_pick));
  rows.push(row('Draft order <span class="muted">(range from the pick alone)</span>', m.draft_order));
  document.getElementById('before-table').innerHTML = head + '<tbody>' + rows.join('') + '</tbody>';
}

function triedTable() {
  var head = '<thead><tr><th>Idea</th><th>Change in Spearman</th><th>Classes better</th><th>Kept?</th></tr></thead>';
  var rows = [];
  var names = {
    '+ seasons 5/3/2 newest': 'Last three college seasons, weighted 5/3/2 toward the newest',
    '+ seasons 2/3/5 oldest': 'Last three seasons, weighted toward the oldest (Pelton style)',
    '+ schedule': 'Stats adjusted for opponent strength (ridge team ratings)',
    '+ shrinkage': 'Small samples shrunk toward position averages',
    '+ age adjusted': 'Stats adjusted for age, stat by stat',
    '+ position relative': 'Stats relative to the position and season average',
    '+ combine flags': 'Flags for skipping the combine'
  };
  var keep = { '+ seasons 5/3/2 newest': true };
  bt.ablation.forEach(function (r) {
    if (!names[r.run]) return;
    rows.push('<tr><td>' + names[r.run] + '</td><td>' + signed(r.mean_change, 3) + '</td><td>' + r.classes_better + ' of ' + r.classes +
      '</td><td>' + (keep[r.run] ? 'kept' : 'dropped') + '</td></tr>');
  });
  bt.ablation.forEach(function (r) {
    if (r.run !== 'chosen + shot types') return;
    rows.push('<tr><td>Shot types from play by play <span class="muted">(classes from 2014 on, where they exist)</span></td><td>' + signed(r.change_2014_on, 3) +
      '</td><td>' + r.better_2014_on + ' of ' + r.classes_2014_on + '</td><td>kept</td></tr>');
  });
  // the heads, against the final blend, over every target's classes
  HEADS.forEach(function (h) {
    var diffs = [], better = 0, n = 0;
    Object.keys(bt.targets).forEach(function (k) {
      var m = bt.targets[k].models;
      if (!m[h.key]) return;
      m[h.key].spearman.forEach(function (v, i) {
        var d = v - m.model.spearman[i];
        diffs.push(d);
        n += 1;
        if (d > 0) better += 1;
      });
    });
    if (!n) return;
    rows.push('<tr><td>' + h.label + ' <span class="muted">(vs the blend)</span></td><td>' + signed(d3.mean(diffs), 3) + '</td><td>' + better + ' of ' + n +
      '</td><td>' + (h.key === 'model_gbm' || h.key === 'model_ridge' ? 'kept, blended' : h.key === 'v1_average' ? 'old baseline' : 'dropped') + '</td></tr>');
  });
  document.getElementById('tried-table').innerHTML = head + '<tbody>' + rows.join('') + '</tbody>';
}

function auditTable() {
  var head = '<thead><tr><th>Target</th><th>Draft order Spearman</th><th>Never played, median place</th><th>500+ minutes, median place</th></tr></thead>';
  var body = bt.audit.map(function (r) {
    return '<tr><td>' + esc(r.target) + '</td><td>' + fmt(r.draft_order_spearman, 3) + '</td><td>' + fmt(r.never_played_rank, 2) + '</td><td>' + fmt(r.played_500_rank, 2) + '</td></tr>';
  }).join('');
  document.getElementById('audit-table').innerHTML = head + '<tbody>' + body + '</tbody>';
}

function spearmanChart(c) {
  var t = T();
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
  var w = widthOf('chart-spearman');
  put('chart-spearman', Plot.plot({
    width: w, height: 300, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Draft class', tickFormat: function (d) { return "'" + String(d).slice(2); }, ticks: w < 500 ? 6 : Math.min(t.years.length, 12) },
    y: { label: 'Spearman', grid: true, domain: [Math.min(0, d3.min(rows, function (r) { return r.value; })), 1] },
    marks: marks
  }));
  document.getElementById('legend-spearman').innerHTML = MODELS.map(function (mm) {
    return '<span><i style="background:' + cssVar(mm.color) + '"></i>' + mm.label + '</span>';
  }).join('');
}

function coverageChart(c) {
  var t = T();
  var who = [['model', 'Stats only', c.orange], ['model_pick', 'Stats + slot', c.teal], ['draft_order', 'Pick alone', c.grey]];
  var rows = [];
  t.years.forEach(function (y, i) {
    who.forEach(function (w) {
      var cov = t.models[w[0]].coverage_by_year;
      if (cov) rows.push({ year: String(y), who: w[1], share: cov[i] });
    });
  });
  var w = widthOf('chart-coverage');
  put('chart-coverage', Plot.plot({
    width: w, height: 260, marginLeft: 40, style: baseStyle(c),
    fx: { type: 'band', label: null, tickFormat: function (d) { return t.years.length > 12 && w < 500 && Number(d) % 2 ? '' : "'" + d.slice(2); } },
    x: { axis: null, domain: who.map(function (x) { return x[1]; }) },
    y: { label: 'Inside the range', domain: [0, 1], grid: true, tickFormat: '.0%' },
    color: { domain: who.map(function (x) { return x[1]; }), range: who.map(function (x) { return x[2]; }) },
    marks: [
      Plot.barY(rows, { fx: 'year', x: 'who', y: 'share', fill: 'who', rx: 1, tip: true,
        title: function (d) { return d.who + ', ' + d.year + ': ' + Math.round(100 * d.share) + '%'; } }),
      Plot.ruleY([0.8], { stroke: c.fg, strokeDasharray: '4,3' })
    ]
  }));
  document.getElementById('legend-coverage').innerHTML = who.map(function (x) {
    return '<span><i style="background:' + x[2] + '"></i>' + x[1] + '</span>';
  }).join('') + '<span><i style="background:' + c.fg + '"></i>target 80%</span>';
}

function hitsChart(c) {
  var t = T();
  var rows = [];
  t.years.forEach(function (y, i) {
    rows.push({ year: String(y), who: 'Model', hits: t.models.model.top10[i] });
    rows.push({ year: String(y), who: 'Draft order', hits: t.models.draft_order.top10[i] });
  });
  var w = widthOf('chart-hits');
  put('chart-hits', Plot.plot({
    width: w, height: 260, marginLeft: 30, style: baseStyle(c),
    fx: { type: 'band', label: null, tickFormat: function (d) { return t.years.length > 12 && w < 500 && Number(d) % 2 ? '' : "'" + d.slice(2); } },
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
  var data = T().calibration;
  var lo = d3.min(data, function (d) { return Math.min(d.pred, d.actual, d.low); });
  var hi = d3.max(data, function (d) { return Math.max(d.pred, d.actual, d.high); });
  put('chart-calib', Plot.plot({
    width: widthOf('chart-calib'), height: 260, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Projected, ' + unit(), domain: [lo, hi], grid: true },
    y: { label: 'Actual', domain: [lo, hi], grid: true },
    marks: [
      Plot.line([[lo, lo], [hi, hi]], { stroke: c.muted, strokeDasharray: '4,4' }),
      Plot.ruleX(data, { x: 'pred', y1: 'low', y2: 'high', stroke: c.grey, strokeWidth: 3, strokeOpacity: 0.5, strokeLinecap: 'round' }),
      Plot.dot(data, { x: 'pred', y: 'actual', fill: c.orange, r: 5, stroke: c.card, strokeWidth: 1.5,
        tip: true, title: function (d) { return 'projected ' + d.pred.toFixed(2) + '\nactual ' + d.actual.toFixed(2) + '\n' + d.n + ' players'; } })
    ]
  }));
}

function scatterChart(c) {
  var data = T().scatter;
  put('chart-scatter', Plot.plot({
    width: widthOf('chart-scatter'), height: 280, marginLeft: 40, style: baseStyle(c),
    x: { label: 'Projected, ' + unit(), grid: true },
    y: { label: 'Actual', grid: true },
    marks: [
      Plot.ruleY([T().floor], { stroke: c.border }),
      Plot.dot(data, { x: 'pred', y: 'actual', r: 2.6, fill: function (d) { return d.pick <= 30 ? c.orange : c.teal; }, fillOpacity: 0.6 }),
      Plot.tip(data, Plot.pointer({ x: 'pred', y: 'actual', title: function (d) {
        return d.name + ' (' + d.year + ', pick ' + d.pick + ')\nprojected ' + d.pred.toFixed(2) + ', actual ' + d.actual.toFixed(2);
      } }))
    ]
  }));
}

function importanceChart(c) {
  var data = T().importance;
  var w = widthOf('chart-importance');
  put('chart-importance', Plot.plot({
    width: w, height: 30 + 24 * data.length, marginLeft: w < 500 ? 150 : 210, style: baseStyle(c),
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
  var t = T(), m = t.models;
  var head = '<thead><tr><th>Class</th><th>Stats only</th><th>+ slot</th><th>V3 settings</th><th>Draft order</th><th>Top 10 model</th><th>Top 10 draft</th><th>NDCG model</th><th>NDCG draft</th><th>Range held</th></tr></thead>';
  var body = t.years.map(function (y, i) {
    var a = m.model.spearman[i], d = m.draft_order.spearman[i];
    return '<tr><td>' + y + '</td><td class="' + (a > d ? 'win' : '') + '">' + fmt(a, 3) + '</td><td>' + fmt(m.model_pick.spearman[i], 3) +
      '</td><td>' + fmt(m.v3.spearman[i], 3) + '</td><td class="' + (d >= a ? 'win' : '') + '">' + fmt(d, 3) + '</td><td>' +
      m.model.top10[i] + '</td><td>' + m.draft_order.top10[i] + '</td><td>' + fmt(m.model.ndcg10[i], 3) + '</td><td>' + fmt(m.draft_order.ndcg10[i], 3) +
      '</td><td>' + pct(m.model.coverage_by_year[i], 0) + '</td></tr>';
  }).join('');
  var foot = '<tr><td><b>Average</b></td><td>' + fmt(m.model.mean_spearman, 3) + '</td><td>' + fmt(m.model_pick.mean_spearman, 3) + '</td><td>' +
    fmt(m.v3.mean_spearman, 3) + '</td><td>' + fmt(m.draft_order.mean_spearman, 3) + '</td><td>' + fmt(m.model.mean_top10, 1) + '</td><td>' +
    fmt(m.draft_order.mean_top10, 1) + '</td><td>' + fmt(m.model.mean_ndcg10, 3) + '</td><td>' + fmt(m.draft_order.mean_ndcg10, 3) + '</td><td>' +
    pct(m.model.coverage, 0) + '</td></tr>';
  document.getElementById('year-table').innerHTML = head + '<tbody>' + body + foot + '</tbody>';
}

function drawAll() {
  if (!bt) return;
  var c = colors();
  buttons();
  headline();
  beforeTable();
  spearmanChart(c);
  coverageChart(c);
  hitsChart(c);
  calibChart(c);
  scatterChart(c);
  importanceChart(c);
  yearTable();
}

document.addEventListener('DOMContentLoaded', function () {
  loadJSON('data/backtest.json').then(function (data) {
    bt = data;
    drawAll();
    triedTable();
    auditTable();
  }).catch(function (e) {
    document.getElementById('headline').innerHTML = '<div class="card stat">' + esc(e.message) + '</div>';
  });
  document.getElementById('target').addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
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
