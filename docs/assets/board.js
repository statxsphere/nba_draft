// the draft board: one class at a time, sortable, with a drawer per player

var state = {
  meta: null,
  backtest: null,
  year: null,
  players: [],
  rankBy: 'peak',
  round: 'all',
  bg: 'all',
  search: '',
  sort: 'rank',
  asc: true,
  cache: {}
};

var LO = -4, HI = 10;  // the part of the -10..10 scale that players actually use

function pos(v) {
  var x = Math.max(LO, Math.min(HI, v));
  return (100 * (x - LO)) / (HI - LO);
}

function rangeBar(p, key, actual) {
  var r = p.pred[key];
  var html = '<div class="rangebar ' + key + '"><div class="axis"></div>' +
    '<div class="zero" style="left:' + pos(0) + '%"></div>' +
    '<div class="band" style="left:' + pos(r.rating_low) + '%;width:' + Math.max(pos(r.rating_high) - pos(r.rating_low), 1) + '%"></div>' +
    '<div class="mid" style="left:' + pos(r.rating) + '%"></div>';
  if (actual !== null && actual !== undefined) html += '<div class="act" style="left:' + pos(actual) + '%"></div>';
  return html + '</div>';
}

function actualRating(p, key) {
  var a = p.actual;
  var scale = state.meta.scales[key];
  var v = key === 'peak' ? (a.peak_complete ? a.peak : a.peak_so_far) : (a.imm_complete ? a.imm : a.imm_so_far);
  if (v === null || v === undefined) return null;
  return Math.max(-10, Math.min(10, (10 * v) / scale));
}

function bgChip(bg) {
  if (bg === 'college') return '';
  var cls = bg === 'international' ? 'intl' : bg === 'high school' ? 'hs' : 'gl';
  var label = bg === 'international' ? 'Intl' : bg === 'high school' ? 'HS' : 'G League';
  return ' <span class="chip ' + cls + '">' + label + '</span>';
}

function rankOf(p) {
  return state.rankBy === 'peak' ? p.model_rank : p.imm_rank;
}

function sortValue(p, k) {
  if (k === 'pick') return p.pick;
  if (k === 'rank') return rankOf(p);
  if (k === 'delta') return p.pick - rankOf(p);
  if (k === 'name') return p.name.toLowerCase();
  if (k === 'age') return p.age === null ? 99 : p.age;
  if (k === 'peak') return p.pred.peak.rating;
  if (k === 'imm') return p.pred.imm.rating;
  if (k === 'actual') { var a = actualRating(p, state.rankBy); return a === null ? -99 : a; }
  return 0;
}

function visiblePlayers() {
  var q = state.search.trim().toLowerCase();
  var list = state.players.filter(function (p) {
    if (state.round !== 'all' && String(p.round) !== state.round) return false;
    if (state.bg !== 'all' && p.background !== state.bg) return false;
    if (q && (p.name + ' ' + p.from + ' ' + p.team).toLowerCase().indexOf(q) === -1) return false;
    return true;
  });
  var k = state.sort, dir = state.asc ? 1 : -1;
  list.sort(function (a, b) {
    var x = sortValue(a, k), y = sortValue(b, k);
    if (x < y) return -dir;
    if (x > y) return dir;
    return a.pick - b.pick;
  });
  return list;
}

function renderTable() {
  var tbody = document.querySelector('#board tbody');
  var list = visiblePlayers();
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty">No players match those filters.</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(function (p) {
    var rank = rankOf(p);
    var d = p.pick - rank;
    var dcls = d > 0 ? 'up' : d < 0 ? 'down' : '';
    var act = actualRating(p, state.rankBy);
    var actText = act === null ? '<span class="muted">–</span>' : fmt(act, 1);
    var sub = [p.position, p.from].filter(Boolean).join(' · ');
    var comp = p.ceiling ? '<div class="small">' + esc(p.ceiling) + '</div><div class="psub">' + esc(p.floor) + '</div>' : '<span class="muted">–</span>';
    return '<tr data-id="' + p.id + '">' +
      '<td class="num"><span class="pick-badge">' + p.pick + '</span></td>' +
      '<td class="num"><span class="pick-badge rank-badge">' + rank + '</span></td>' +
      '<td class="num hide-sm"><span class="delta ' + dcls + '">' + (d === 0 ? '0' : signed(d, 0)) + '</span></td>' +
      '<td><div class="pname">' + esc(p.name) + bgChip(p.background) + '</div><div class="psub">' + esc(sub) + '</div></td>' +
      '<td class="num hide-md">' + fmt(p.age, 1) + '</td>' +
      '<td><div class="cell-score"><span class="score">' + fmt(p.pred.peak.rating, 1) + '</span>' + rangeBar(p, 'peak', state.rankBy === 'peak' ? act : actualRating(p, 'peak')) + '</div></td>' +
      '<td class="hide-sm"><div class="cell-score"><span class="score">' + fmt(p.pred.imm.rating, 1) + '</span>' + rangeBar(p, 'imm', actualRating(p, 'imm')) + '</div></td>' +
      '<td class="num hide-md">' + actText + '</td>' +
      '<td class="hide-md">' + comp + '</td>' +
      '</tr>';
  }).join('');
  document.querySelectorAll('#board th.sortable').forEach(function (th) {
    var on = th.dataset.k === state.sort;
    th.innerHTML = th.textContent.replace(/[▲▼]/g, '') + (on ? '<span class="arrow">' + (state.asc ? '▲' : '▼') + '</span>' : '');
  });
}

function classMetric(key, model, field) {
  var bt = state.backtest.targets[key];
  var i = bt.years.indexOf(state.year);
  if (i === -1) return null;
  return bt.models[model][field][i];
}

function renderStats() {
  var c = state.meta.classes.filter(function (x) { return x.year === state.year; })[0];
  var cards = [];
  var peakModel = classMetric('peak', 'model', 'spearman');
  var peakDraft = classMetric('peak', 'draft_order', 'spearman');
  var immModel = classMetric('imm', 'model', 'spearman');
  var immDraft = classMetric('imm', 'draft_order', 'spearman');

  function card(label, value, note) {
    return '<div class="card stat"><div class="stat-label">' + label + '</div><div class="stat-value">' + value + '</div><div class="stat-note">' + note + '</div></div>';
  }
  function vs(a, b) {
    return fmt(a, 2) + ' <span class="vs">vs ' + fmt(b, 2) + '</span>';
  }
  cards.push(card('Players drafted', c.players, c.outcomes === 'complete' ? 'six full NBA seasons in the books' :
    c.outcomes === 'partial' ? 'careers still in progress' : 'no NBA games yet'));
  if (peakModel !== null) {
    cards.push(card('Peak: model vs draft order', vs(peakModel, peakDraft), 'rank correlation with the real peak'));
  } else {
    cards.push(card('Peak: model vs draft order', '<span class="stat-text">Pending</span>', 'needs six seasons to grade'));
  }
  if (immModel !== null) {
    cards.push(card('Immediate: model vs draft order', vs(immModel, immDraft), 'rank correlation with the first two seasons'));
  } else {
    cards.push(card('Immediate: model vs draft order', '<span class="stat-text">Pending</span>', 'needs two seasons to grade'));
  }
  var hits = classMetric(state.rankBy, 'model', 'top10');
  var draftHits = classMetric(state.rankBy, 'draft_order', 'top10');
  if (hits !== null) {
    cards.push(card('Top 10 hits (' + (state.rankBy === 'peak' ? 'peak' : 'immediate') + ')', hits + ' <span class="vs">vs ' + draftHits + '</span>', 'of the real top 10, model vs picks 1 to 10'));
  } else {
    cards.push(card('Projections made', '<span class="stat-text">Out of sample</span>', 'the model never saw this class'));
  }
  document.getElementById('class-stats').innerHTML = cards.join('');

  var status = c.outcomes === 'complete' ? 'Their careers have run long enough to grade, so the dots show what really happened.' :
    c.outcomes === 'partial' ? 'They are still early in their careers, so the dots show their numbers so far.' :
    'Nobody here has played an NBA game in the data yet, so these are pure projections.';
  document.getElementById('board-title').innerHTML = 'The <span class="mono">' + state.year + '</span> class';
  document.getElementById('board-lede').textContent = 'Every pick ranked by what the model expects from them, next to where teams actually took them. ' + status;
  document.getElementById('actual-head').firstChild.textContent = c.outcomes === 'complete' ? 'Actual' : 'So far';
}

function renderYears() {
  var html = state.meta.classes.slice().reverse().map(function (c) {
    var color = c.outcomes === 'complete' ? 'var(--teal)' : c.outcomes === 'partial' ? 'var(--orange)' : 'var(--grey-line)';
    return '<button data-y="' + c.year + '" class="' + (c.year === state.year ? 'on' : '') + '" role="tab">' + c.year +
      '<span class="dot" style="background:' + color + '"></span></button>';
  }).join('');
  var el = document.getElementById('years');
  el.innerHTML = html;
  var on = el.querySelector('.on');
  if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function loadYear(year, openId) {
  state.year = year;
  renderYears();
  function done(players) {
    state.players = players;
    renderStats();
    renderTable();
    if (openId) openPlayer(openId, false);
  }
  if (state.cache[year]) return done(state.cache[year]);
  loadJSON('data/board_' + year + '.json').then(function (players) {
    state.cache[year] = players;
    done(players);
  }).catch(function (e) {
    document.querySelector('#board tbody').innerHTML = '<tr><td colspan="9" class="empty">' + esc(e.message) + '</td></tr>';
  });
}

// ---------- the drawer ----------

function ratingCard(p, key) {
  var r = p.pred[key];
  var title = key === 'peak' ? 'Peak Impact' : 'Immediate Impact';
  var act = actualRating(p, key);
  var a = p.actual;
  var complete = key === 'peak' ? a.peak_complete : a.imm_complete;
  var actLine = act === null ? '<span class="muted">No NBA seasons yet.</span>' :
    (complete ? 'Actual: ' : 'So far: ') + '<b>' + fmt(act, 1) + '</b>' +
    ' <span class="muted">(' + fmt(key === 'peak' ? (complete ? a.peak : a.peak_so_far) : (complete ? a.imm : a.imm_so_far), 2) + ' VORP a season)</span>';
  return '<div class="card rating-card"><div class="stat-label">' + title + ' Rating</div>' +
    '<div class="big" style="color:' + (key === 'peak' ? 'var(--orange)' : 'var(--teal)') + '">' + fmt(r.rating, 1) + '</div>' +
    '<div class="range">80% range ' + fmt(r.rating_low, 1) + ' to ' + fmt(r.rating_high, 1) + ' · ' + fmt(r.vorp, 2) + ' VORP/season</div>' +
    rangeBar(p, key, act) +
    '<div class="actual">' + actLine + '</div></div>';
}

function perPossCard(p, key, title) {
  // the per possession projections, in points per 100 against an average player
  var r = p.pred[key];
  if (!r) return '';
  var a = p.actual;
  var act = key === 'bpm' ? (a.bpm !== null && a.bpm !== undefined ? a.bpm : a.bpm_so_far) : a.rapm;
  var done = key === 'bpm' ? a.bpm !== null && a.bpm !== undefined : act !== null && act !== undefined;
  var actLine = act === null || act === undefined ? '<span class="muted">Not graded yet.</span>' :
    (done ? 'Actual: ' : 'So far: ') + '<b>' + signed(act, 1) + '</b>';
  return '<div class="card rating-card"><div class="stat-label">' + title + '</div>' +
    '<div class="big">' + signed(r.rating, 1) + '</div>' +
    '<div class="range">80% range ' + signed(r.rating_low, 1) + ' to ' + signed(r.rating_high, 1) + '</div>' +
    '<div class="actual">' + actLine + '</div></div>';
}

function places(x) {
  var n = Math.max(1, Math.abs(Math.round(x)));
  return n + (n === 1 ? ' place' : ' places');
}

function driverList(p, key) {
  var list = p.pred[key].drivers;
  if (!list.length) return '<p class="muted small">No drivers saved for this one.</p>';
  var maxAbs = Math.max.apply(null, list.map(function (d) { return Math.abs(d.pctl); }).concat([2]));
  var rows = list.map(function (d) {
    var w = (50 * Math.abs(d.pctl)) / maxAbs;
    var color = d.pctl >= 0 ? 'var(--teal)' : 'var(--orange)';
    var bar = d.pctl >= 0 ? 'left:50%;width:' + w + '%' : 'right:50%;width:' + w + '%';
    return '<div class="driver"><div class="what"><b>' + esc(d.label) + '</b> <span>' + esc(d.value) + '</span></div>' +
      '<div class="bar"><span class="mid"></span><i style="' + bar + ';background:' + color + '"></i></div>' +
      '<div class="eff">' + signed(d.pctl, Math.abs(d.pctl) < 1 ? 1 : 0) + '</div></div>';
  }).join('');
  var up = list.filter(function (d) { return d.pctl > 0; })[0];
  var down = list.filter(function (d) { return d.pctl < 0; })[0];
  var words = [];
  if (up) words.push('The biggest push up is ' + up.label.toLowerCase() + ' (' + up.value + '), which moves him about ' + places(up.pctl) + ' up a 100-player class.');
  if (down) words.push('The biggest drag is ' + down.label.toLowerCase() + ' (' + down.value + '), about ' + places(down.pctl) + ' down.');
  return '<div class="drivers">' + rows + '</div><p class="plain">' + esc(words.join(' ')) + '</p>';
}

function collegeTable(p) {
  var c = p.college;
  if (!c) {
    var why = p.background === 'college' ? 'His college box scores weren\'t in the data, so the model works from age and measurements only.' :
      'He came from ' + (p.background === 'international' ? 'overseas' : p.background === 'high school' ? 'high school' : 'the G League / OTE') +
      ', so there are no college stats. The model works from age and measurements and knows his path.';
    return '<p class="muted small">' + why + '</p>';
  }
  return '<div class="table-scroll"><table class="boxscore"><thead><tr><th>Season</th><th>G</th><th>MPG</th><th>PTS</th><th>REB</th><th>AST</th><th>STL</th><th>BLK</th><th>TS%</th><th>3P%</th><th>USG</th></tr></thead>' +
    '<tbody><tr><td>' + esc(c.season) + '</td><td>' + c.g + '</td><td>' + fmt(c.mpg, 1) + '</td><td>' + fmt(c.pts40, 1) + '</td><td>' + fmt(c.trb40, 1) +
    '</td><td>' + fmt(c.ast40, 1) + '</td><td>' + fmt(c.stl40, 1) + '</td><td>' + fmt(c.blk40, 1) + '</td><td>' + pct(c.ts, 1) + '</td><td>' + pct(c.fg3_pct, 1) +
    '</td><td>' + fmt(c.usg, 1) + '</td></tr></tbody></table></div><p class="plain">' + esc(c.team) + ', counting stats per 40 minutes.</p>';
}

function careerTable(p) {
  if (!p.seasons.length) return '<p class="muted small">No NBA seasons in the data yet (it runs through ' + state.meta.last_nba_season + ').</p>';
  var rows = p.seasons.map(function (s) {
    return '<tr><td>' + s.season + '</td><td>' + s.g + '</td><td>' + s.mp + '</td><td>' + fmt(s.bpm, 1) + '</td><td>' + fmt(s.ws, 1) + '</td><td>' + fmt(s.vorp, 1) + '</td></tr>';
  }).join('');
  return '<div class="table-scroll"><table class="boxscore"><thead><tr><th>Season</th><th>G</th><th>MP</th><th>BPM</th><th>WS</th><th>VORP</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}

function compList(p) {
  if (!p.comps.length) return '<p class="muted small">Not enough history with matching numbers to find comps.</p>';
  return '<div class="comps">' + p.comps.map(function (c) {
    var tag = c.name === p.ceiling ? '<span class="tag ceil">ceiling</span>' : c.name === p.floor ? '<span class="tag floor">floor</span>' : '';
    return '<div class="card comp"><div><div class="cname">' + esc(c.name) + tag + '</div><div class="cmeta">' + c.year + ' draft, pick ' + c.pick + ' · ' + Math.round(100 * c.sim) + '% similar</div></div>' +
      '<div class="spark-wrap">' + sparkline(c.line, 'var(--orange)') + '</div>' +
      '<div class="cnums">peak ' + fmt(c.peak, 1) + '<br><span class="muted">imm ' + fmt(c.imm, 1) + '</span></div></div>';
  }).join('') + '</div>';
}

function openPlayer(id, push) {
  var p = state.players.filter(function (x) { return x.id === id; })[0];
  if (!p) return;
  var facts = [
    ['Pick', p.pick + ' (' + p.team + ')'], ['From', p.from], ['Position', p.position || '–'], ['Age', fmt(p.age, 1)],
    ['Height', p.height || '–'], ['Wingspan', p.wingspan || '–'], ['Weight', p.weight ? p.weight + ' lb' : '–']
  ].map(function (f) { return '<span>' + f[0] + ' <b>' + esc(f[1]) + '</b></span>'; }).join('');
  var key = state.rankBy;
  var other = key === 'peak' ? 'imm' : 'peak';
  var html = '<div class="drawer-head"><div><p class="eyebrow">' + state.year + ' draft · model #' + p.model_rank + ' for peak, #' + p.imm_rank + ' for immediate</p>' +
    '<h2>' + esc(p.name) + bgChip(p.background) + '</h2><div class="facts">' + facts + '</div></div>' +
    '<button class="icon-btn close" id="drawer-close" aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button></div>' +
    '<div class="drawer-body">' +
    '<div class="ratings">' + ratingCard(p, 'peak') + ratingCard(p, 'imm') + '</div>' +
    '<div class="ratings small-cards">' + perPossCard(p, 'bpm', 'Peak BPM, per 100') + perPossCard(p, 'rapm', 'Peak plus-minus, per 100') + '</div>' +
    '<p class="muted small per-note">Per possession, so minutes do not count. 0 is an average player and −2 is replacement level, which is also the floor.</p>' +
    '<section><div class="section-title"><h3>What drives the ' + (key === 'peak' ? 'peak' : 'immediate') + ' projection</h3><span>SHAP, in class percentile points</span></div>' + driverList(p, key) + '</section>' +
    '<section><div class="section-title"><h3>And the ' + (other === 'peak' ? 'peak' : 'immediate') + ' one</h3></div>' + driverList(p, other) + '</section>' +
    '<section><div class="section-title"><h3>Pre-draft season</h3></div>' + collegeTable(p) + '</section>' +
    '<section><div class="section-title"><h3>Closest comps</h3><span>cosine similarity, earlier classes only</span></div>' + compList(p) + '</section>' +
    '<section><div class="section-title"><h3>NBA career</h3><span>first ten seasons</span></div>' + careerTable(p) + '</section>' +
    '</div>';
  var drawer = document.getElementById('drawer');
  drawer.innerHTML = html;
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  drawer.scrollTop = 0;
  document.getElementById('scrim').classList.add('open');
  document.getElementById('drawer-close').addEventListener('click', closePlayer);
  document.getElementById('drawer-close').focus();
  if (push !== false) history.replaceState(null, '', '#' + state.year + '/' + p.id);
}

function closePlayer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'true');
  document.getElementById('scrim').classList.remove('open');
  history.replaceState(null, '', '#' + state.year);
}

function segment(id, field, after) {
  var el = document.getElementById(id);
  el.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b) return;
    el.querySelectorAll('button').forEach(function (x) { x.classList.toggle('on', x === b); });
    state[field] = b.dataset.v;
    if (after) after();
    renderTable();
  });
}

function start() {
  Promise.all([loadJSON('data/meta.json'), loadJSON('data/backtest.json')]).then(function (res) {
    state.meta = res[0];
    state.backtest = res[1];
    document.getElementById('updated').textContent = 'NBA data through ' + state.meta.last_nba_season + ' · updated ' + state.meta.updated;

    var hash = location.hash.replace('#', '').split('/');
    var years = state.meta.classes.map(function (c) { return c.year; });
    var year = years.indexOf(parseInt(hash[0], 10)) !== -1 ? parseInt(hash[0], 10) : years[years.length - 1];
    loadYear(year, hash[1] ? parseInt(hash[1], 10) : null);

    document.getElementById('years').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      history.replaceState(null, '', '#' + b.dataset.y);
      loadYear(parseInt(b.dataset.y, 10));
    });
    segment('rank-by', 'rankBy', function () { renderStats(); });
    segment('round-filter', 'round');
    document.getElementById('bg-filter').addEventListener('change', function (e) { state.bg = e.target.value; renderTable(); });
    document.getElementById('search').addEventListener('input', function (e) { state.search = e.target.value; renderTable(); });
    document.querySelectorAll('#board th.sortable').forEach(function (th) {
      th.addEventListener('click', function () {
        var k = th.dataset.k;
        if (state.sort === k) state.asc = !state.asc;
        else { state.sort = k; state.asc = ['pick', 'rank', 'name', 'age'].indexOf(k) !== -1; }
        renderTable();
      });
    });
    document.querySelector('#board tbody').addEventListener('click', function (e) {
      var tr = e.target.closest('tr[data-id]');
      if (tr) openPlayer(parseInt(tr.dataset.id, 10));
    });
    document.getElementById('scrim').addEventListener('click', closePlayer);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closePlayer(); });
  }).catch(function (e) {
    document.querySelector('#board tbody').innerHTML = '<tr><td colspan="9" class="empty">' + esc(e.message) + '</td></tr>';
  });
}

document.addEventListener('DOMContentLoaded', start);
