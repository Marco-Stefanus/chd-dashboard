"""
Tampilan smartwatch untuk halaman "Simulasi smartwatch".

Seluruh animasi (hitung mundur, pengukuran detak jantung, tekanan darah, analisis, hasil)
berjalan di browser dengan HTML/CSS/JS. Python hanya mengirim data lewat `payload`:

    {
      "vitals": {"hr": 78, "sys": 128, "dia": 82},     # nilai yang "terbaca" oleh jam
      "models": [{"name","prob","thr","pred","color"}] atau null,   # hasil prediksi
      "label": "HIGH" | "LOW" | null,
      "votes": 2, "n": 3,
      "source": "live" | "notebook" | "none",
      "speed": 1.0                                       # 1 = normal, 0.35 = mode cepat
    }
"""
from __future__ import annotations

import json

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:transparent;font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;}
body{display:flex;justify-content:center;padding:6px 0 0}
.stage{display:flex;flex-direction:column;align-items:center;width:270px}

/* tali dan badan jam */
.band{width:116px;background:linear-gradient(90deg,#16303f,#24495c 50%,#16303f);position:relative}
.band.top{height:104px;border-radius:26px 26px 4px 4px;margin-bottom:-6px}
.band.bot{height:104px;border-radius:4px 4px 26px 26px;margin-top:-6px}
.band::after{content:"";position:absolute;left:50%;top:14px;bottom:14px;width:2px;transform:translateX(-50%);
  background:repeating-linear-gradient(to bottom,rgba(255,255,255,.12) 0 6px,transparent 6px 14px)}
.case{position:relative;width:252px;height:300px;border-radius:66px;padding:12px;
  background:linear-gradient(145deg,#3a3f44,#0d0f11 60%,#2a2e32);
  box-shadow:0 18px 34px rgba(15,42,61,.28),inset 0 0 0 1.5px rgba(255,255,255,.08);z-index:2}
.crown{position:absolute;right:-10px;top:86px;width:12px;height:46px;border-radius:7px;
  background:linear-gradient(90deg,#6b7076,#23272b);box-shadow:0 2px 4px rgba(0,0,0,.3)}
.side{position:absolute;right:-7px;top:152px;width:8px;height:30px;border-radius:5px;background:linear-gradient(90deg,#50555a,#1d2024)}
.screen{position:relative;width:100%;height:100%;border-radius:54px;background:#000;overflow:hidden;color:#fff}

/* layar */
.scr{position:absolute;inset:0;padding:26px 24px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;opacity:0;pointer-events:none;transition:opacity .35s ease}
.scr.on{opacity:1;pointer-events:auto}
.lbl{font-size:12.5px;font-weight:600;color:#9fb6c4;line-height:1.35}
.lbl.small{font-size:10.5px;margin-top:8px;color:#6f8797}
.big{font-size:40px;font-weight:800;line-height:1;margin:6px 0 2px;letter-spacing:-.02em}
.big small{font-size:12px;font-weight:600;color:#9fb6c4;margin-left:4px;letter-spacing:0}

/* idle */
.time{font-size:46px;font-weight:800;letter-spacing:-.03em;line-height:1}
.date{font-size:12px;color:#9fb6c4;margin:5px 0 16px;font-weight:500}
.go{border:0;border-radius:999px;padding:9px 16px;font:700 12.5px 'Plus Jakarta Sans',sans-serif;color:#fff;cursor:pointer;
  background:linear-gradient(120deg,#0B7A8A,#19A7B5);box-shadow:0 4px 14px rgba(25,167,181,.4)}
.go:hover{filter:brightness(1.1)}
.hint{font-size:10px;color:#6f8797;margin-top:9px}
.ring{position:absolute;inset:10px;border-radius:46px;border:2px solid rgba(25,167,181,.18);pointer-events:none}
.go{position:relative;z-index:2}

/* hitung mundur */
.count{font-size:64px;font-weight:800;color:#19A7B5;line-height:1;margin:6px 0}

/* detak jantung */
.heart{width:34px;height:34px;animation:beat 1s ease-in-out infinite;transform-origin:center;filter:drop-shadow(0 0 8px rgba(255,90,95,.55))}
@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.22)}30%{transform:scale(.98)}45%{transform:scale(1.12)}}
.ecg{width:168px;height:34px;overflow:hidden;margin:6px 0 4px}
.ecg g{animation:scroll 1.3s linear infinite}
@keyframes scroll{from{transform:translateX(0)}to{transform:translateX(-85px)}}
.bar{width:150px;height:5px;border-radius:5px;background:#1c2a33;margin-top:10px;overflow:hidden}
.bar i{display:block;height:100%;width:0;border-radius:5px;background:linear-gradient(90deg,#0B7A8A,#19A7B5)}

/* tekanan darah */
.cuff{width:54px;height:54px;border-radius:50%;border:5px solid #19A7B5;margin:2px 0 8px;animation:squeeze 1.7s ease-in-out infinite;
  box-shadow:0 0 14px rgba(25,167,181,.4)}
@keyframes squeeze{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(.72);opacity:.65}}
.bpgrid{display:flex;gap:22px;margin-top:4px}
.bpgrid div{text-align:center}
.bpgrid small{display:block;font-size:10px;color:#9fb6c4;font-weight:600}
.bpgrid b{font-size:32px;font-weight:800;line-height:1.1;letter-spacing:-.02em}
.unit{font-size:10.5px;color:#9fb6c4;margin-top:1px}

/* ringkasan */
.rows{width:100%;margin-top:9px;display:flex;flex-direction:column;gap:7px}
.row{display:flex;align-items:center;justify-content:space-between;background:#10181e;border-radius:12px;padding:7px 11px}
.row .n{font-size:10.5px;color:#9fb6c4;font-weight:600;text-align:left;white-space:nowrap}
.row>div:last-child{white-space:nowrap;display:flex;align-items:baseline}
.row .v{font-size:15px;font-weight:800}
.row .v small{font-size:9.5px;color:#9fb6c4;font-weight:600;margin-left:2px}
.tag{font-size:9.5px;font-weight:700;padding:2px 7px;border-radius:999px;margin-left:8px}
.tag.ok{background:rgba(46,201,138,.18);color:#2EC98A}
.tag.mid{background:rgba(255,176,32,.18);color:#FFB020}
.tag.hi{background:rgba(255,90,95,.2);color:#FF7A7E}
.tag.lo{background:rgba(96,165,250,.2);color:#7FB6FF}

/* analisis */
.spin{width:34px;height:34px;border-radius:50%;border:4px solid #1c2a33;border-top-color:#19A7B5;animation:rot .9s linear infinite;margin-bottom:10px}
@keyframes rot{to{transform:rotate(360deg)}}
.mlist{margin-top:12px;width:100%;display:flex;flex-direction:column;gap:6px}
.mrow{display:flex;justify-content:space-between;font-size:11.5px;background:#10181e;border-radius:10px;padding:6px 11px;color:#9fb6c4;font-weight:600}
.mrow .s{color:#6f8797}
.mrow.done .s{color:#2EC98A}

/* hasil */
#s-res{transition:opacity .35s ease,background .4s}
#s-res.low{background:radial-gradient(circle at 50% 18%,#0f5a40,#04150f 72%)}
#s-res.high{background:radial-gradient(circle at 50% 18%,#7a1c20,#1a0507 72%)}
#s-res.none{background:radial-gradient(circle at 50% 18%,#6a4a08,#150f02 72%)}
.ricon{width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:23px;font-weight:800;margin-bottom:7px}
.low .ricon{background:#2EC98A;color:#04150f}.high .ricon{background:#FF5A5F;color:#1a0507}.none .ricon{background:#FFB020;color:#150f02}
.rtitle{font-size:17px;font-weight:800;line-height:1.15}
.rsub{font-size:10.5px;color:#cfe0ea;margin-top:3px}
.chips{display:flex;gap:4px;flex-wrap:wrap;justify-content:center;margin:8px 0 6px}
.chips span{font-size:9px;font-weight:700;padding:2px 7px;border-radius:999px}
.rmsg{font-size:10.5px;color:#e6eef3;line-height:1.4;max-width:170px}
.again{margin-top:9px;border:1px solid rgba(255,255,255,.35);background:transparent;color:#fff;border-radius:999px;
  font:600 10.5px 'Plus Jakarta Sans',sans-serif;padding:5px 12px;cursor:pointer}
.again:hover{background:rgba(255,255,255,.12)}
.foot{font-size:8.5px;color:#9fb6c4;margin-top:7px;opacity:.85}
</style></head>
<body>
<div class="stage">
  <div class="band top"></div>
  <div class="case">
    <div class="crown"></div><div class="side"></div>
    <div class="screen">

      <div class="scr on" id="s-idle">
        <div class="ring"></div>
        <div class="time" id="clock">10:09</div>
        <div class="date" id="date">Jumat, 16 Okt</div>
        <button class="go" id="start">♥&nbsp; Cek kesehatan</button>
        <div class="hint">Ketuk untuk memulai tes</div>
      </div>

      <div class="scr" id="s-ready">
        <div class="lbl">Posisikan jam di pergelangan tangan</div>
        <div class="count" id="count">3</div>
        <div class="lbl small">Duduk tenang dan rileks</div>
      </div>

      <div class="scr" id="s-hr">
        <div class="lbl">Mengukur detak jantung</div>
        <svg class="heart" id="heart" viewBox="0 0 24 24"><path fill="#FF5A5F" d="M12 21s-7-4.6-9.2-9.2C1.1 8.2 3 4.5 6.7 4.5c2 0 3.5 1 5.3 3.2 1.8-2.2 3.3-3.2 5.3-3.2 3.7 0 5.6 3.7 3.9 7.3C19 16.4 12 21 12 21z"/></svg>
        <div class="big"><span id="hr-v">--</span><small>BPM</small></div>
        <svg class="ecg" viewBox="0 0 168 34"><g fill="none" stroke="#19A7B5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M0 20 H25 L30 20 L34 6 L40 32 L44 14 L48 20 H85 L90 20 L94 6 L100 32 L104 14 L108 20 H170 L175 20 L179 6 L185 32 L189 14 L193 20 H255"/></g></svg>
        <div class="bar"><i id="hr-p"></i></div>
      </div>

      <div class="scr" id="s-bp">
        <div class="lbl">Mengukur tekanan darah</div>
        <div class="cuff" style="margin-top:8px"></div>
        <div class="bpgrid"><div><small>Sistolik</small><b id="sys-v">--</b></div><div><small>Diastolik</small><b id="dia-v">--</b></div></div>
        <div class="unit">mmHg</div>
        <div class="bar"><i id="bp-p"></i></div>
      </div>

      <div class="scr" id="s-sum">
        <div class="lbl">Hasil pengukuran</div>
        <div class="rows">
          <div class="row"><div class="n">Jantung</div><div><span class="v" id="o-hr"></span><span class="tag" id="t-hr"></span></div></div>
          <div class="row"><div class="n">Sistolik</div><div><span class="v" id="o-sys"></span><span class="tag" id="t-sys"></span></div></div>
          <div class="row"><div class="n">Diastolik</div><div><span class="v" id="o-dia"></span><span class="tag" id="t-dia"></span></div></div>
        </div>
      </div>

      <div class="scr" id="s-ana">
        <div class="spin"></div>
        <div class="lbl">Menganalisis risiko CHD</div>
        <div class="mlist" id="mlist"></div>
      </div>

      <div class="scr none" id="s-res">
        <div class="ricon" id="r-icon">!</div>
        <div class="rtitle" id="r-title"></div>
        <div class="rsub" id="r-sub"></div>
        <div class="chips" id="r-chips"></div>
        <div class="rmsg" id="r-msg"></div>
        <button class="again" id="again">↻ Ulangi tes</button>
        <div class="foot">Risiko CHD 10 tahun · simulasi</div>
      </div>

    </div>
  </div>
  <div class="band bot"></div>
</div>

<script>
const P = __PAYLOAD__;
const D = P.speed || 1;
const $ = id => document.getElementById(id);
let running = false;

function tickClock(){
  const d = new Date();
  $('clock').textContent = d.toLocaleTimeString('id-ID',{hour:'2-digit',minute:'2-digit',hour12:false}).replace('.',':');
  $('date').textContent = d.toLocaleDateString('id-ID',{weekday:'long',day:'numeric',month:'short'});
}
tickClock(); setInterval(tickClock, 15000);

function show(id){ document.querySelectorAll('.scr').forEach(e => e.classList.toggle('on', e.id === id)); }
const wait = ms => new Promise(r => setTimeout(r, ms * D));

function converge(el, target, spread, dur, bar){
  return new Promise(res => {
    const t0 = performance.now();
    const iv = setInterval(() => {
      const p = Math.min(1, (performance.now() - t0) / (dur * D));
      const noise = (Math.random() * 2 - 1) * spread * (1 - p);
      el.textContent = Math.round(target + noise);
      if (bar) bar.style.width = (p * 100) + '%';
      if (p >= 1){ el.textContent = target; clearInterval(iv); res(); }
    }, 120);
  });
}

function band(kind, v){
  // rentang acuan umum (mis. AHA): sistolik <120 normal, 120-129 meningkat, >=130 tinggi; diastolik <80 normal
  if (kind === 'hr')  return v < 60 ? ['Rendah','lo'] : v <= 100 ? ['Normal','ok'] : ['Tinggi','hi'];
  if (kind === 'sys') return v < 120 ? ['Normal','ok'] : v < 130 ? ['Meningkat','mid'] : ['Tinggi','hi'];
  return v < 80 ? ['Normal','ok'] : ['Tinggi','hi'];
}
function fill(idV, idT, kind, v, unit){
  $(idV).innerHTML = v + '<small>' + unit + '</small>';
  const [t, c] = band(kind, v);
  const el = $(idT); el.textContent = t; el.className = 'tag ' + c;
}

async function run(){
  if (running) return; running = true;
  const V = P.vitals;

  show('s-ready');
  for (let n = 3; n > 0; n--){ $('count').textContent = n; await wait(900); }

  show('s-hr');
  $('heart').style.animationDuration = (60 / V.hr) + 's';
  await converge($('hr-v'), V.hr, 14, 5200, $('hr-p'));
  await wait(500);

  show('s-bp');
  $('sys-v').textContent = '--'; $('dia-v').textContent = '--';
  await wait(1200);
  await Promise.all([converge($('sys-v'), V.sys, 26, 4600, $('bp-p')), converge($('dia-v'), V.dia, 16, 4600)]);
  await wait(500);

  show('s-sum');
  fill('o-hr','t-hr','hr',V.hr,' bpm'); fill('o-sys','t-sys','sys',V.sys,''); fill('o-dia','t-dia','dia',V.dia,'');
  await wait(3600);

  show('s-ana');
  const list = $('mlist'); list.innerHTML = '';
  const names = (P.models && P.models.length) ? P.models.map(m => m.name) : ['XGBoost','EBM','TabNet'];
  const rows = names.map(n => { const r = document.createElement('div'); r.className = 'mrow';
    r.innerHTML = '<span>' + n + '</span><span class="s">memproses…</span>'; list.appendChild(r); return r; });
  for (const r of rows){ await wait(900); r.classList.add('done'); r.querySelector('.s').textContent = '✓ selesai'; }
  await wait(600);

  renderResult();
  show('s-res');
  running = false;
}

function renderResult(){
  const res = $('s-res'), icon = $('r-icon'), title = $('r-title'), sub = $('r-sub'), chips = $('r-chips'), msg = $('r-msg');
  chips.innerHTML = '';
  if (!P.models || !P.label){
    res.className = 'scr none'; icon.textContent = '?';
    title.textContent = 'Hasil belum tersedia';
    sub.textContent = 'Model belum terpasang';
    msg.textContent = 'Letakkan file model di folder models/ agar jam bisa memprediksi.';
    return;
  }
  const high = P.label === 'HIGH';
  res.className = 'scr ' + (high ? 'high' : 'low');
  icon.textContent = high ? '!' : '✓';
  title.innerHTML = high ? 'Risiko CHD<br>Tinggi' : 'Risiko CHD<br>Rendah';
  sub.textContent = P.votes + ' dari ' + P.n + ' model menandai CHD';
  P.models.forEach(m => {
    const s = document.createElement('span');
    s.textContent = m.name + ': ' + (m.pred ? 'CHD' : 'Aman');
    s.style.background = m.pred ? m.color : 'rgba(255,255,255,.14)';
    s.style.color = '#fff';
    chips.appendChild(s);
  });
  msg.textContent = high ? 'Disarankan berkonsultasi dengan dokter untuk pemeriksaan lanjutan.'
                         : 'Pertahankan pola hidup sehat dan pantau berkala.';
}

$('start').addEventListener('click', run);
$('again').addEventListener('click', () => { show('s-idle'); });
</script>
</body></html>
"""


def watch_html(payload: dict) -> str:
    """Kembalikan HTML lengkap smartwatch dengan data `payload` tertanam."""
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.replace("__PAYLOAD__", data)
