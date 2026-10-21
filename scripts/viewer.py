#!/usr/bin/env python3
"""本地浏览器里实时看优化：python viewer.py E:\\Download\\SIGMA85 [--port 8765]  → http://localhost:8765

零依赖（标准库 http.server）。各级脚本通过 live.py 把状态写进 <workdir>/_live.json，网页每秒轮询重画：
  layout（优化后轮廓 + 光线，叠在原结构图上，滚轮缩放/拖动平移，悬停看每面 R 与矢高框占用）
  误差曲线、RMS、玻璃表 + nd–vd 玻璃图（灰点=目录，彩点=当前各片）、日志
  「停止」按钮 → 写 _stop，优化脚本在段间检查后收尾退出
  反馈框 → 追加到 _feedback.md（带时间戳），供下一轮读取
只监听 127.0.0.1。
"""
import argparse, json, os, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
WD = None; CATALOG = None


def catalog(vendor):
    global CATALOG
    if CATALOG is None:
        try:
            sys.path.insert(0, os.path.join(HERE, '..', '..', 'patent-lens-to-excel', 'scripts'))
            import lensmath
            CATALOG = [[g['name'], g['nd'], g['vd']] for g in lensmath.load(vendor) if not lensmath.legacy(g) and g['vd'] > 15]
        except Exception:
            CATALOG = []
    return CATALOG


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, str): body = body.encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split('?')[0]
        if p == '/': return self._send(200, PAGE, 'text/html; charset=utf-8')
        if p == '/state':
            try: return self._send(200, open(os.path.join(WD, '_live.json'), 'rb').read())
            except Exception: return self._send(200, '{}')
        if p == '/catalog': return self._send(200, json.dumps(catalog(self.server.vendor)))
        if p == '/image':
            try:
                st = json.load(open(os.path.join(WD, '_live.json'), encoding='utf-8')); f = st['meta']['image']['path']
                if os.path.dirname(os.path.abspath(f)) != os.path.abspath(WD): raise PermissionError
                return self._send(200, open(f, 'rb').read(), 'image/jpeg' if f.lower().endswith(('jpg', 'jpeg')) else 'image/png')
            except Exception: return self._send(404, '')
        if p == '/feedback':
            try: return self._send(200, open(os.path.join(WD, '_feedback.md'), 'rb').read(), 'text/plain; charset=utf-8')
            except Exception: return self._send(200, '', 'text/plain; charset=utf-8')
        self._send(404, '{}')

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0); body = self.rfile.read(min(n, 100000)).decode('utf-8', 'replace')
        if self.path == '/stop':
            open(os.path.join(WD, '_stop'), 'w').write(time.strftime('%H:%M:%S')); return self._send(200, '{"ok":true}')
        if self.path == '/feedback':
            try: d = json.loads(body)
            except Exception: d = {'text': body}
            txt = str(d.get('text', '')).strip()
            if txt:
                with open(os.path.join(WD, '_feedback.md'), 'a', encoding='utf-8') as f:
                    f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')}  [{d.get('stage', '')}] {d.get('ctx', '')}\n{txt}\n")
            return self._send(200, '{"ok":true}')
        self._send(404, '{}')


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>镜头优化 · 实时</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2330;--mut:#6b7686;--card:#fff;--line:#d9dee6;--acc:#2563eb;--ok:#15803d;--warn:#b45309;--bad:#b91c1c}
@media(prefers-color-scheme:dark){:root{--bg:#12151b;--fg:#e6e9ee;--mut:#8b95a5;--card:#1b2029;--line:#2c3340;--acc:#6ea0ff;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}
header{display:flex;gap:14px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5;flex-wrap:wrap}
header h1{font-size:15px;margin:0}.pill{padding:2px 9px;border-radius:99px;background:var(--acc);color:#fff;font-weight:600}
.mut{color:var(--mut)}.grow{flex:1}button{font:inherit;padding:4px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.stop{border-color:var(--bad);color:var(--bad)}label{user-select:none;cursor:pointer}
main{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:10px;padding:10px}@media(max-width:1100px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}.card h2{font-size:12px;margin:0 0 6px;color:var(--mut);font-weight:600;letter-spacing:.04em}
#lay{width:100%;height:52vh;display:block;background:#fff;border-radius:6px;cursor:grab;touch-action:none}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:2px 5px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);font-weight:500}td:first-child,th:first-child{text-align:left}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.kv div{background:var(--bg);border-radius:6px;padding:5px 8px}.kv b{display:block;font-size:15px;font-variant-numeric:tabular-nums}
#log{height:150px;overflow:auto;font:11.5px/1.4 Consolas,monospace;white-space:pre-wrap;background:var(--bg);border-radius:6px;padding:6px}
textarea{width:100%;height:64px;font:inherit;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);padding:6px;resize:vertical}
#tip{position:fixed;pointer-events:none;background:#111c;color:#fff;padding:4px 8px;border-radius:5px;font-size:12px;display:none;z-index:9;white-space:pre}
.bar{height:7px;border-radius:4px;background:var(--line);position:relative;min-width:70px}.bar i{position:absolute;top:0;bottom:0;border-radius:4px}
svg text{font-family:inherit}
main{padding-bottom:96px}
#dock{position:fixed;left:0;right:0;bottom:0;z-index:8;background:var(--card);border-top:1px solid var(--line);padding:8px 14px;display:flex;gap:10px;align-items:flex-end;box-shadow:0 -4px 14px #0002}
#dock textarea{height:40px;min-height:40px;max-height:40vh;flex:1}#dock textarea:focus{height:110px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:1100px){.two{grid-template-columns:1fr}}
.lg{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:var(--mut);margin-top:4px}.lg i{display:inline-block;width:22px;height:0;border-top:2px solid;vertical-align:middle;margin-right:4px}
</style></head><body>
<header><h1 id="title">镜头优化</h1><span class="pill" id="stage">—</span><span id="msg"></span><span class="grow"></span>
<span class="mut" id="clock"></span>
<label><input type="checkbox" id="cImg" checked> 原图</label><label><input type="checkbox" id="cRay" checked> 光线</label>
<label><input type="checkbox" id="cNum"> 面号</label><label><input type="checkbox" id="cBox"> 贴框标红</label><button id="fit">适应窗口</button><button class="stop" id="stop">停止优化</button></header>
<main><div style="display:grid;gap:10px;min-width:0">
<div class="card"><svg id="lay"></svg></div>
<div class="two"><div class="card"><h2>MTF vs 像高（CODE V 衍射 MTF，F/全开）</h2><svg id="mtf" width="100%" height="230"></svg><div class="lg"><span><i style="border-color:#dc2626"></i>10 lp/mm S</span><span><i style="border-color:#dc2626;border-top-style:dashed"></i>10 lp/mm M</span><span><i style="border-color:#16a34a"></i>30 lp/mm S</span><span><i style="border-color:#16a34a;border-top-style:dashed"></i>30 lp/mm M</span><span id="mtfn"></span></div></div><div class="card"><h2>相对照度 / 畸变 vs 像高</h2><svg id="ri" width="100%" height="230"></svg><div class="lg"><span><i style="border-color:#2563eb"></i>相对照度 %（左轴）</span><span><i style="border-color:#ea580c"></i>畸变 %（右轴）</span></div></div></div>
<div class="card"><h2>矢高框占用（每面 边缘矢高偏移 ÷ 容差；±1 = 贴框）</h2><svg id="sag" width="100%" height="90"></svg></div>
<div class="card"><h2>日志</h2><div id="log"></div></div>
</div><div style="display:grid;gap:10px;align-content:start;min-width:0">
<div class="card"><h2>指标</h2><div class="kv" id="kv"></div><div id="spots" style="margin-top:8px"></div></div>
<div class="card"><h2>误差曲线（各级量纲不同，按级分色；对数轴）</h2><svg id="err" width="100%" height="130"></svg></div>
<div class="card"><h2>玻璃图 nd–vd（灰=目录，蓝=SLD 片，橙=其余；圈=已定牌号）</h2><svg id="gmap" width="100%" height="210"></svg></div>
<div class="card"><h2>玻璃</h2><div style="max-height:300px;overflow:auto"><table id="gt"></table></div></div>
</div></main><div id="dock"><textarea id="fb" placeholder="反馈写给 Claude（存到 _feedback.md）。点图上的面会自动填「面N：」。Ctrl+Enter 发送"></textarea><button id="send">发送</button><span class="mut" id="fbs" style="min-width:90px"></span></div><div id="tip"></div>
<script>
const $=s=>document.querySelector(s),NS='http://www.w3.org/2000/svg',FC=['#dc2626','#16a34a','#2563eb','#9333ea'];
let S=null,cat=[],vb=null,lastT='',sel='';
const el=(n,a,p)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);p&&p.appendChild(e);return e};
const P=p=>p.map(q=>q[0].toFixed(2)+','+(-q[1]).toFixed(2)).join(' ');
function fit(){if(!S||!S.geo)return;const z1=S.geo.img_z+4,h=Math.max(...S.geo.surf.map(s=>s.semi||0))+6;vb=[-8,-h,z1+8,2*h];apply()}
function apply(){$('#lay').setAttribute('viewBox',vb.join(' '))}
function drawLay(){const v=$('#lay');v.innerHTML='';if(!S.geo)return;if(!vb)fit();const g=S.geo,im=S.meta.image,px=vb[2]/v.clientWidth;
 const showImg=im&&$('#cImg').checked,over=new Set(((S.aper||{}).marg||[]).concat((S.aper||{}).chief||[]).map(q=>q[0]));
 if(showImg){el('image',{href:'/image',x:im.xi,y:-(im.yi+im.H*im.mmpx),width:im.W*im.mmpx,height:im.H*im.mmpx,opacity:.55,preserveAspectRatio:'none'},v)}
 el('line',{x1:-12,y1:0,x2:g.img_z+3,y2:0,stroke:'#9ca3af','stroke-width':px,'stroke-dasharray':`${10*px} ${3*px} ${2*px} ${3*px}`},v);
 for(const e of g.edges)el('polygon',{points:P(e.poly),fill:e.cover?'#e5e7eb':e.sld?'#93c5fd':'#e0f2fe','fill-opacity':showImg?.5:.9,stroke:'#1f2937','stroke-width':1.2*px,'stroke-linejoin':'round'},v);
 if($('#cRay').checked)for(const r of g.rays)el('polyline',{points:P(r.p),fill:'none',stroke:FC[r.f],'stroke-width':1.1*px,'stroke-linejoin':'round'},v);
 for(const s of g.surf){if(s.stop){for(const k of[1,-1])el('line',{x1:s.z,x2:s.z,y1:-k*((S.metrics||{}).y_stop||s.semi),y2:-k*(s.semi+5),stroke:'#111','stroke-width':2.5*px},v);continue}
  const r=s.eps?Math.abs(s.dz)/s.eps:0,edge=$('#cBox').checked&&r>.98,col=s.asph?'#ea580c':(edge?'#b91c1c':'transparent');
  if(s.ca)el('polyline',{points:P(s.pts.filter(q=>Math.abs(q[1])<=s.ca+.01)),fill:'none',stroke:'#0f172a','stroke-width':2.4*px,'stroke-linecap':'round'},v);
  const pl=el('polyline',{points:P(s.pts),fill:'none',stroke:sel==s.k?'#2563eb':col,'stroke-width':(sel==s.k?3:1.8)*px},v);
  if(over.has(s.k))for(const q of[s.pts[0],s.pts[s.pts.length-1]])el('circle',{cx:q[0],cy:-q[1],r:4*px,fill:'#dc2626'},v);
  const hit=el('polyline',{points:P(s.pts),fill:'none',stroke:'transparent','stroke-width':9*px,style:'cursor:pointer'},v);
  hit.onmousemove=e=>tip(e,`面 ${s.k}${s.asph?'（非球面）':''}\nR = ${s.R??'∞'}   图上 R₀ = ${s.R0??'∞'}\n边缘矢高偏移 ${s.dz??'—'} / 容差 ±${s.eps??'—'} mm\n通光口径 ${s.ca??'—'}（粗线）/ 图上画到 ${s.semi}`);hit.onmouseleave=()=>$('#tip').style.display='none';
  hit.onclick=()=>{sel=s.k;$('#fb').value+=(($('#fb').value&&!$('#fb').value.endsWith('\n'))?'\n':'')+`面${s.k}：`;$('#fb').focus();drawLay()};
  if($('#cNum').checked){const q=s.pts[s.pts.length-1];el('text',{x:q[0],y:-q[1]-1.2,'font-size':11*px,'text-anchor':'middle',fill:s.asph?'#ea580c':'#555'},v).textContent=s.k}}
 el('line',{x1:g.img_z,x2:g.img_z,y1:-22,y2:22,stroke:'#111','stroke-width':2.5*px},v)}
function tip(e,t){const d=$('#tip');d.textContent=t;d.style.display='block';d.style.left=e.clientX+14+'px';d.style.top=e.clientY+12+'px'}
function drawSag(){const v=$('#sag');v.innerHTML='';const W=v.clientWidth,Hh=90,ss=S.geo.surf.filter(s=>s.eps);if(!ss.length)return;const bw=W/ss.length,mid=Hh/2-6;
 for(const y of[-1,0,1])el('line',{x1:0,x2:W,y1:mid-y*30,y2:mid-y*30,stroke:y?'#b91c1c55':'#8886'},v);
 ss.forEach((s,i)=>{const r=Math.max(-1.6,Math.min(1.6,s.dz/s.eps)),a=Math.abs(r),c=a>.98?'#b91c1c':a>.7?'#d97706':'#2563eb';
  const b=el('rect',{x:i*bw+2,width:Math.max(2,bw-4),y:r>0?mid-r*30:mid,height:Math.max(1,a*30),fill:s.asph?'#ea580c':c,rx:2},v);
  b.onmousemove=e=>tip(e,`面 ${s.k}  偏移/容差 = ${(s.dz/s.eps).toFixed(2)}\n偏移 ${s.dz} mm，容差 ±${s.eps} mm`);b.onmouseleave=()=>$('#tip').style.display='none';
  el('text',{x:i*bw+bw/2,y:Hh-2,'font-size':9,'text-anchor':'middle',fill:'#888'},v).textContent=s.k})}
function drawErr(){const v=$('#err');v.innerHTML='';const h=(S.hist||[]).filter(p=>p.e>0);if(h.length<2)return;const W=v.clientWidth,Hh=130,pad=34;
 const st=[...new Set(h.map(p=>p.stage))],col=['#2563eb','#16a34a','#ea580c','#9333ea','#dc2626','#0891b2'];
 const lo=Math.log10(Math.min(...h.map(p=>p.e))),hi=Math.log10(Math.max(...h.map(p=>p.e)))+1e-9,X=i=>pad+i/(h.length-1)*(W-pad-6),Y=e=>8+(hi-Math.log10(e))/(hi-lo||1)*(Hh-26);
 for(let d=Math.ceil(lo);d<=hi;d++){el('line',{x1:pad,x2:W,y1:Y(10**d),y2:Y(10**d),stroke:'#8883'},v);el('text',{x:pad-4,y:Y(10**d)+3,'font-size':9,'text-anchor':'end',fill:'#888'},v).textContent='1e'+d}
 st.forEach((s,k)=>{let pts=[];h.forEach((p,i)=>{if(p.stage==s)pts.push(X(i)+','+Y(p.e))});el('polyline',{points:pts.join(' '),fill:'none',stroke:col[k%6],'stroke-width':1.6},v);
  const i0=h.findIndex(p=>p.stage==s);el('text',{x:X(i0)+2,y:Hh-3,'font-size':9,fill:col[k%6]},v).textContent=s})}
function drawG(){const v=$('#gmap');v.innerHTML='';const W=v.clientWidth,Hh=210,X=vd=>8+(96-vd)/(96-15)*(W-16),Y=nd=>Hh-8-(nd-1.42)/(2.02-1.42)*(Hh-16);
 for(const c of cat)el('circle',{cx:X(c[2]),cy:Y(c[1]),r:1.5,fill:'#9aa3af'},v);
 for(const g of S.glasses||[]){const c=el('circle',{cx:X(g.vd),cy:Y(g.nd),r:4.5,fill:g.sld?'#2563eb':'#ea580c',stroke:g.name?'#111':'none','stroke-width':1.5,opacity:.9},v);
  c.onmousemove=e=>tip(e,`片 ${g.e}  nd ${g.nd}  vd ${g.vd}\n${g.name?'牌号 '+g.name:'最近 '+g.near+'（距离 '+g.dist+'）'}`);c.onmouseleave=()=>$('#tip').style.display='none';
  el('text',{x:X(g.vd)+6,y:Y(g.nd)+3,'font-size':9,fill:'#555'},v).textContent=g.e}}
function axes(v,W,Hh,pl,pr,xmax,yt,fmt){for(const t of yt){const y=8+(1-t)*(Hh-30);el('line',{x1:pl,x2:W-pr,y1:y,y2:y,stroke:'#8883'},v);el('text',{x:pl-4,y:y+3,'font-size':9,'text-anchor':'end',fill:'#888'},v).textContent=fmt(t)}
 for(let h=0;h<=xmax+.01;h+=5){const x=pl+h/xmax*(W-pl-pr);el('line',{x1:x,x2:x,y1:8,y2:Hh-22,stroke:'#8882'},v);el('text',{x:x,y:Hh-10,'font-size':9,'text-anchor':'middle',fill:'#888'},v).textContent=h}
 el('text',{x:W-pr,y:Hh-1,'font-size':9,'text-anchor':'end',fill:'#888'},v).textContent='像高 mm'}
function drawMTF(){const v=$('#mtf'),r=$('#ri');v.innerHTML='';r.innerHTML='';const M=(S.extra||{}).mtf;if(!M||!M.length){el('text',{x:10,y:24,'font-size':12,fill:'#888'},v).textContent='（第 2 级起由 CODE V 计算；第 1 级没有）';$('#mtfn').textContent='';return}
 const W=v.clientWidth,Hh=230,pl=28,pr=8,xmax=Math.max(21.7,...M.map(d=>d.h)),X=h=>pl+h/xmax*(W-pl-pr),Y=t=>8+(1-t)*(Hh-30);
 axes(v,W,Hh,pl,pr,xmax,[0,.2,.4,.6,.8,1],t=>t.toFixed(1));
 for(const[k,c]of[['f10','#dc2626'],['f30','#16a34a']])for(const j of[0,1]){const pts=M.filter(d=>d[k]).map(d=>X(d.h)+','+Y(d[k][j])).join(' ');
  el('polyline',{points:pts,fill:'none',stroke:c,'stroke-width':1.8,'stroke-dasharray':j?'6 4':''},v);
  for(const d of M.filter(d=>d[k])){const c0=el('circle',{cx:X(d.h),cy:Y(d[k][j]),r:2.6,fill:c},v);c0.onmousemove=e=>tip(e,`像高 ${d.h} mm\n10 lp/mm  S ${d.f10[0]}  M ${d.f10[1]}\n30 lp/mm  S ${d.f30[0]}  M ${d.f30[1]}`);c0.onmouseleave=()=>$('#tip').style.display='none'}}
 $('#mtfn').textContent=M.length+' 个视场点';
 const W2=r.clientWidth,pr2=30,X2=h=>pl+h/xmax*(W2-pl-pr2),dm=Math.max(1,...M.map(d=>Math.abs(d.dist)))*1.15,YD=t=>8+(1-(t/dm+1)/2)*(Hh-30);
 axes(r,W2,Hh,pl,pr2,xmax,[0,.25,.5,.75,1],t=>(t*100).toFixed(0));
 for(const t of[-1,-.5,0,.5,1])el('text',{x:W2-pr2+4,y:YD(t*dm)+3,'font-size':9,fill:'#ea580c'},r).textContent=(t*dm).toFixed(1);
 el('polyline',{points:M.map(d=>X2(d.h)+','+Y(d.ri/100)).join(' '),fill:'none',stroke:'#2563eb','stroke-width':1.8},r);
 el('polyline',{points:M.map(d=>X2(d.h)+','+YD(d.dist)).join(' '),fill:'none',stroke:'#ea580c','stroke-width':1.8},r);
 for(const d of M){const c0=el('circle',{cx:X2(d.h),cy:Y(d.ri/100),r:2.6,fill:'#2563eb'},r);c0.onmousemove=e=>tip(e,`像高 ${d.h} mm\n相对照度 ${d.ri}%\n畸变 ${d.dist}%`);c0.onmouseleave=()=>$('#tip').style.display='none';el('circle',{cx:X2(d.h),cy:YD(d.dist),r:2.6,fill:'#ea580c'},r)}}
function drawT(){const m=S.metrics||{},kv=[['EFL',m.efl],['后截距 CT / ET',(m.back??'—')+' / '+(m.back_et??'—')],['误差',S.errf==null?'—':(+S.errf).toPrecision(5)],['更新',S.time]];
 $('#kv').innerHTML=kv.map(k=>`<div><span class=mut>${k[0]}</span><b>${k[1]??'—'}</b></div>`).join('');
 const A=S.aper||{},M0=((S.extra||{}).mtf||[])[0]||{},bad=(A.marg||[]).length,okc='style="color:var(--ok)"',badc='style="color:var(--bad)"';
 let h=`<div style="margin:0 0 8px;line-height:1.7"><div ${bad?badc:okc}>${bad?'✗ 轴上边缘光线超口径：面 '+A.marg.map(q=>q[0]+'('+q[1]+')').join(' '):'✓ 轴上无渐晕（最紧：面 '+(A.worst_marg||['—','—'])[0]+'，占用 '+(A.worst_marg||['—','—'])[1]+'）'}</div>`
  +((A.chief||[]).length?`<div ${badc}>✗ 轴外光束超口径上限：面 ${A.chief.map(q=>q[0]+'('+q[1]+')').join(' ')}</div>`:'')
  +(A.et_worst?`<div ${(A.et_bad||[]).length?badc:okc}>${(A.et_bad||[]).length?'✗ 边缘厚超加工极限：'+A.et_bad.map(q=>'片'+q[0]+' CT'+q[1]+'/ET'+q[2]+'='+q[3]).join('　'):'✓ 中心厚/边缘厚 全部合格（最极端：片 '+A.et_worst[0]+'，比 '+A.et_worst[3]+'）'}</div>`:'')
  +(()=>{const M=(S.extra||{}).mtf;if(!M||!M.length)return A.tan_pass!=null?`<div>最大视场子午通光 ${(A.tan_pass*100).toFixed(0)}%</div>`:'';const ri=M[M.length-1].ri;return `<div ${Math.abs(ri-25)>6?badc:okc}>最大视场相对照度 ${ri}%（目标 ≈25%）</div>`})()
  +`<div>畸变 %：${(A.dist||[]).map(q=>`<span ${Math.abs(q[1])>2?badc:''}>${q[0]}H ${q[1]>0?'+':''}${q[1]}</span>`).join('　')}</div>`
  +(M0.fno_axis?`<div ${Math.abs(M0.fno_axis-S.meta.fno)>.02?badc:okc}>CODE V 轴上实际 F/${M0.fno_axis.toFixed(3)}（设定 F/${S.meta.fno}）</div>`:'')+'</div>';
 h+='<table><tr><th>视场</th><th>子午 RMS µm</th><th>通光%</th></tr>'+(m.tan_rms||[]).map(r=>`<tr><td>${r[0]}H</td><td>${r[1]??'—'}</td><td>${r[2]}</td></tr>`).join('')+'</table>';
 const sp=(S.extra||{}).spots;if(sp)h+='<table style="margin-top:6px"><tr><th>CODE V 视场°</th><th>RMS 弥散斑直径 µm</th></tr>'+sp.map(r=>`<tr><td>${r[0]}</td><td>${r[1].toFixed(1)}</td></tr>`).join('')+'</table>';
 $('#spots').innerHTML=h;
 $('#gt').innerHTML='<tr><th>片</th><th>nd</th><th>vd</th><th>牌号 / 最近</th><th>距离</th></tr>'+(S.glasses||[]).map(g=>`<tr><td>${g.e}${g.sld?' <span style="color:#2563eb">SLD</span>':''}</td><td>${g.nd.toFixed(5)}</td><td>${g.vd.toFixed(2)}</td><td>${g.name?'<b>'+g.name+'</b>':g.near}</td><td style="color:${g.dist>1.5?'var(--bad)':g.dist>.8?'var(--warn)':'inherit'}">${g.name?'—':g.dist}</td></tr>`).join('')}
async function tick(){try{const r=await fetch('/state');const s=await r.json();if(s.time&&s.time+s.stage+s.msg!==lastT){lastT=s.time+s.stage+s.msg;S=s;
 $('#title').textContent=s.meta.title||'镜头优化';$('#stage').textContent=s.stage||'—';$('#msg').textContent=s.msg||'';
 drawLay();drawSag();drawMTF();drawErr();drawG();drawT();const L=$('#log'),bot=L.scrollTop+L.clientHeight>=L.scrollHeight-8;L.textContent=(s.log||[]).join('\n');if(bot)L.scrollTop=L.scrollHeight}
 $('#clock').textContent=S?('状态 '+S.time):'等待优化脚本写入 _live.json …'}catch(e){$('#clock').textContent='连接中断'}}
const lay=$('#lay');lay.addEventListener('wheel',e=>{if(!vb)return;e.preventDefault();const k=e.deltaY>0?1.15:1/1.15,r=lay.getBoundingClientRect(),mx=vb[0]+(e.clientX-r.left)/r.width*vb[2],my=vb[1]+(e.clientY-r.top)/r.height*vb[3];
 vb=[mx-(mx-vb[0])*k,my-(my-vb[1])*k,vb[2]*k,vb[3]*k];apply();drawLay()},{passive:false});
let drag=null;lay.onpointerdown=e=>{drag=[e.clientX,e.clientY,vb.slice()];lay.setPointerCapture(e.pointerId)};lay.onpointerup=()=>drag=null;
lay.onpointermove=e=>{if(!drag)return;const r=lay.getBoundingClientRect();vb[0]=drag[2][0]-(e.clientX-drag[0])/r.width*vb[2];vb[1]=drag[2][1]-(e.clientY-drag[1])/r.height*vb[3];apply()};
for(const id of['#cImg','#cRay','#cNum','#cBox'])$(id).onchange=()=>S&&drawLay();$('#fit').onclick=()=>{fit();drawLay()};
$('#stop').onclick=async()=>{if(confirm('让优化脚本在当前这一段结束后停下并保存？')){await fetch('/stop',{method:'POST'});$('#msg').textContent='已请求停止…'}};
async function send(){const t=$('#fb').value.trim();if(!t)return;await fetch('/feedback',{method:'POST',body:JSON.stringify({text:t,stage:S&&S.stage,ctx:S&&S.msg})});$('#fb').value='';$('#fbs').textContent='已保存 '+new Date().toLocaleTimeString()}
$('#send').onclick=send;$('#fb').onkeydown=e=>{if(e.ctrlKey&&e.key=='Enter')send()};
fetch('/catalog').then(r=>r.json()).then(c=>{cat=c;S&&drawG()});window.onresize=()=>S&&(drawLay(),drawSag(),drawMTF(),drawErr(),drawG());tick();setInterval(tick,1000);
</script></body></html>"""


def main():
    global WD
    ap = argparse.ArgumentParser(); ap.add_argument('workdir'); ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--vendor', default='HOYA'); ap.add_argument('--open', action='store_true', help='启动后自动开默认浏览器')
    a = ap.parse_args(); WD = os.path.abspath(a.workdir)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H); srv.vendor = a.vendor
    url = f'http://localhost:{a.port}'; print('实时查看：', url, ' 工作目录：', WD, flush=True)
    if a.open:
        import webbrowser; webbrowser.open(url)
    srv.serve_forever()


if __name__ == '__main__':
    main()
