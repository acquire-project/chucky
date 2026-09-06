// Derive tables, a comparison chart, and a local interactive view from retained
// measurements. Uses only Node.js standard-library modules; never runs the GPU.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

const here = path.dirname(fileURLToPath(import.meta.url));
const oldDir = path.join(here, '../blosc-rtx5070-20260905');
const GIB = 2 ** 30;
const median = xs => { const s = [...xs].sort((a,b) => a-b); return (s[Math.floor((s.length-1)/2)] + s[Math.floor(s.length/2)]) / 2; };
const id = r => [r.fill, r.chunk_kib, r.codec, r.shuffle, r.block_kib].join('__');
const groupId = r => `${r.fill}__${r.chunk_kib}`;
const sameGroup = (a,b) => groupId(a) === groupId(b);
const dominates = (a,b) => a.speed >= b.speed && a.fold >= b.fold && (a.speed > b.speed || a.fold > b.fold);
const dominatesMemory = (a,b,key) => a.speed >= b.speed && a.fold >= b.fold && a[key] <= b[key] && (a.speed > b.speed || a.fold > b.fold || a[key] < b[key]);
const frontier = rs => rs.filter(b => !rs.some(a => dominates(a,b))).sort((a,b) => a.fold-b.fold || b.speed-a.speed);
const fmt = (n,d=3) => Number(n).toFixed(d);
const csv = (name, rows, keys=Object.keys(rows[0])) => {
  const cell = x => { const s = String(x ?? ''); return /[,"\n]/.test(s) ? '"' + s.replaceAll('"','""') + '"' : s; };
  fs.writeFileSync(path.join(here,name), keys.join(',') + '\n' + rows.map(r => keys.map(k => cell(r[k])).join(',')).join('\n') + '\n');
};
// Historical numeric CSV has no quoted fields.
const oldLines = fs.readFileSync(path.join(oldDir,'summary.csv'),'utf8').trim().split(/\r?\n/);
const oldKeys = oldLines.shift().split(',');
const historical = oldLines.map(line => {
  const o = Object.fromEntries(line.split(',').map((v,i) => [oldKeys[i],v]));
  return {...o, chunk_kib:+o.chunk_kib, block_kib:+o.block_kib, speed:+o.throughput_median_gibs,
    lo:+o.throughput_min_gibs, hi:+o.throughput_max_gibs, fold:+o.compression_fold,
    device_gib:+o.device_gib, estimate_gib:+o.estimated_total_gib, machine:'5070 Laptop'};
});
const rawFile = path.join(here,'raw-results.jsonl.gz');
const rawText = fs.existsSync(rawFile) ? gunzipSync(fs.readFileSync(rawFile)).toString() : fs.readFileSync(path.join(here,'raw-results.jsonl'),'utf8');
const records = rawText.trim().split('\n').map(JSON.parse);
const provenance = JSON.parse(fs.readFileSync(path.join(here,'provenance.json'),'utf8'));
assert.equal(provenance.complete,true,'Wait for the full sweep before generating conclusions');
assert.equal(createHash('sha256').update(rawText).digest('hex'),provenance.raw_results_sha256);
assert.equal(records.length,800);
assert(records.every(r => r.code === 0 && !r.validation_error && r.result.status === 'pass'));
assert.equal(new Set(records.map(r => `${r.pass}__${id(r.config)}`)).size,800);
assert.equal(records.filter(r => r.warmup).length,200);
const measured = records.filter(r => !r.warmup);
assert.equal(measured.length,600);
const groups = Map.groupBy(measured,r => id(r.config));
assert.equal(groups.size,200);
const rows = [...groups.values()].map(reps => {
  assert.equal(reps.length,3);
  const config = reps[0].config;
  assert(reps.every(r => id(r.config) === id(config)));
  const results = reps.map(r => r.result);
  assert(results.every(r=>r.memory_device_overhead_bytes !== null && r.memory_device_used_bytes > 0 && r.memory_estimate_total_bytes > 0),'Memory readings unavailable');
  for (const field of ['compression_fold','memory_estimate_total_bytes','memory_estimate_pinned_bytes','total_chunks','chunks_per_epoch'])
    assert.equal(new Set(results.map(r => r[field])).size,1,`Varying ${field}: ${id(config)}`);
  const metric = f => median(results.map(f));
  const speed = metric(r => r.throughput_in_gibs);
  const lo = Math.min(...results.map(r => r.throughput_in_gibs));
  const hi = Math.max(...results.map(r => r.throughput_in_gibs));
  return {...config, machine:'5080', repeats:3, speed, lo, hi, span_pct:100*(hi-lo)/speed,
    fold:results[0].compression_fold, device_gib:metric(r => r.memory_device_used_bytes/GIB),
    device_min_gib:Math.min(...results.map(r => r.memory_device_used_bytes/GIB)),
    device_max_gib:Math.max(...results.map(r => r.memory_device_used_bytes/GIB)),
    estimate_gib:results[0].memory_estimate_total_bytes/GIB,
    pinned_gib:results[0].memory_estimate_pinned_bytes/GIB,
    overhead_mib:metric(r => r.memory_device_overhead_bytes/2**20),
    input_bytes:results[0].stages.memcpy.in_bytes, compressed_payload_bytes:results[0].stages.compress.out_bytes,
    // The reported fold includes sink writes (aligned shard footers, etc.).
    // Stage payload bytes exclude those writes. Exact sink bytes were not
    // emitted by this benchmark version; preserve the reported fold's precision.
    sink_bytes_approx:results[0].stages.compress.in_bytes/results[0].compression_fold,
    padded_bytes:results[0].stages.compress.in_bytes,
    input_per_encoded:results[0].compression_fold*results[0].stages.memcpy.in_bytes/results[0].stages.compress.in_bytes,
    padding_factor:results[0].stages.compress.in_bytes/results[0].stages.memcpy.in_bytes,
    compress_ms:metric(r => r.stages.compress.total_ms),
    compress_gibs:metric(r => r.stages.compress.in_gibs),
    h2d_ms:metric(r => r.stages.h2d.total_ms), d2h_ms:metric(r => r.stages.d2h.total_ms),
    wall_s:metric(r => r.wall_s), init_s:metric(r => r.init_s),
    chunks_per_epoch:results[0].chunks_per_epoch, total_chunks:results[0].total_chunks};
}).sort((a,b) => id(a).localeCompare(id(b)));
assert.deepEqual(new Set(rows.map(id)),new Set(historical.map(id)));
for (const r of rows) {
  const pool = rows.filter(s => s.block_kib && sameGroup(s,r));
  r.per_codec_frontier = !!r.block_kib && !pool.some(s => s.codec === r.codec && dominates(s,r));
  r.overall_frontier = !!r.block_kib && !pool.some(s => dominates(s,r));
  r.measured_memory_frontier = !!r.block_kib && !pool.some(s => dominatesMemory(s,r,'device_gib'));
  r.estimated_memory_frontier = !!r.block_kib && !pool.some(s => dominatesMemory(s,r,'estimate_gib'));
  const old = historical.find(s => id(s) === id(r));
  r.speedup_vs_5070 = r.speed/old.speed;
  r.fold_delta_vs_5070 = r.fold-old.fold;
  r.estimate_delta_mib = (r.estimate_gib-old.estimate_gib)*1024;
}
csv('summary.csv',rows);
csv('pareto-frontier.csv',rows.filter(r => r.per_codec_frontier));
csv('pareto-memory-frontier.csv',rows.filter(r => r.measured_memory_frontier || r.estimated_memory_frontier));
const memoryGroups=Map.groupBy(rows.filter(r=>r.block_kib),r=>[r.chunk_kib,r.codec,r.shuffle,r.block_kib].join('__'));
const memoryRows=[...memoryGroups.values()].map(pair=>{
  assert.equal(pair.length,2);
  assert.equal(pair[0].estimate_gib,pair[1].estimate_gib,'Estimate depends on input contents');
  const r=pair[0];
  return {chunk_kib:r.chunk_kib,codec:r.codec,shuffle:r.shuffle,block_kib:r.block_kib,
    batch_chunks:r.chunks_per_epoch,estimated_device_bytes:r.estimate_gib*GIB,
    estimated_pinned_bytes:r.pinned_gib*GIB,delta_vs_5070_mib:r.estimate_delta_mib};
});
assert.equal(memoryRows.length,96);
csv('memory-estimates.csv',memoryRows);
csv('comparison.csv',rows.map(r => {
  const old = historical.find(s => id(s) === id(r));
  return {fill:r.fill,chunk_kib:r.chunk_kib,codec:r.codec,shuffle:r.shuffle,block_kib:r.block_kib,
    speed_5080:r.speed,speed_5070:old.speed,speedup:r.speedup_vs_5070,fold_5080:r.fold,fold_5070:old.fold,
    estimate_5080_gib:r.estimate_gib,estimate_5070_gib:old.estimate_gib,estimate_delta_mib:r.estimate_delta_mib};
}));
csv('random-entropy-reference.csv',rows.filter(r=>r.fill==='rand').map(r=>({
  chunk_kib:r.chunk_kib,codec:r.codec,shuffle:r.shuffle,block_kib:r.block_kib,
  padded_compression_fold:r.fold,ingested_compression_fold:r.input_per_encoded,
  padding_factor:r.padding_factor,independent_12bit_reference_fold:4/3*r.padding_factor,
  bytes_above_12bit_reference_pct:100*((4/3)/r.input_per_encoded-1)
})));
const panels = [['xor',256],['xor',1024],['rand',256],['rand',1024]];
const memoryBudget = [];
const sinkEnvelope = [];
for (const [fill,chunk_kib] of panels) {
  const group = rows.filter(r => r.block_kib && r.fill===fill && r.chunk_kib===chunk_kib);
  for (const budget_gib of [1.5,2,2.5,3,4,6]) {
    // Explicit allocations are stable across content and repeats. Capacity only;
    // runtime headroom must be reserved in addition to this budget.
    const feasible = group.filter(r => r.estimate_gib<=budget_gib);
    for (const r of frontier(feasible)) memoryBudget.push({budget_gib,...r});
  }
  for (const sink_gibs of [0.01,0.025,0.05,0.1,0.25,0.5,1,2,4,8]) {
    const candidates = group.map(r => ({r, bound:Math.min(r.speed,sink_gibs*r.input_per_encoded)}));
    candidates.sort((a,b) => b.bound-a.bound || a.r.estimate_gib-b.r.estimate_gib);
    const {r,bound} = candidates[0];
    sinkEnvelope.push({fill,chunk_kib,sink_gibs,codec:r.codec,shuffle:r.shuffle,block_kib:r.block_kib,
      optimistic_input_gibs:bound,discard_input_gibs:r.speed,compression_fold:r.fold,input_per_encoded:r.input_per_encoded});
  }
}
csv('pareto-by-allocation-budget.csv',memoryBudget);
csv('sink-envelope.csv',sinkEnvelope);
const stats = {
  configurations:rows.length, measured_runs:measured.length,
  median_span_pct:median(rows.map(r => r.span_pct)),max_span_pct:Math.max(...rows.map(r => r.span_pct)),
  all_folds_identical_to_5070:rows.every(r => r.fold_delta_vs_5070===0),
  changed_folds:rows.filter(r => r.fold_delta_vs_5070!==0).map(r => ({id:id(r),delta:r.fold_delta_vs_5070})),
  overall_frontiers:panels.map(([fill,chunk_kib]) => ({fill,chunk_kib,points:rows.filter(r => r.fill===fill && r.chunk_kib===chunk_kib && r.overall_frontier).map(r => ({codec:r.codec,shuffle:r.shuffle,block_kib:r.block_kib,speed:r.speed,lo:r.lo,hi:r.hi,fold:r.fold,estimate_gib:r.estimate_gib}))})),
  all_per_codec_frontiers_use_bitshuffle:rows.filter(r => r.per_codec_frontier).every(r => r.shuffle==='bit'),
  measured_pass_speed_vs_warmup:[1,2,3].map(pass=>({pass,median_ratio:median(records.filter(r=>r.pass===pass).map(r=>r.result.throughput_in_gibs/records.find(w=>w.warmup && id(w.config)===id(r.config)).result.throughput_in_gibs))})),
  duration_minutes:(Date.parse(provenance.finish_utc)-Date.parse(provenance.start_utc))/60000
};
fs.writeFileSync(path.join(here,'analysis.json'),JSON.stringify(stats,null,2)+'\n');

// Standalone scientific plot. Current and historical frontiers share axes;
// current min/max bars show repetitions, not confidence intervals.
const esc = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const style = JSON.parse(fs.readFileSync(path.join(here,'../blosc-figure-style.json'),'utf8'));
const colors = style.colors;
const text = (x,y,s,extra='') => `<text x="${x}" y="${y}" ${extra}>${esc(s)}</text>`;
const line = (x1,y1,x2,y2,extra='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${extra}/>`;
function marker(x,y,shape,color,r=5,{open=false,extra='',title=''}={}) {
  const attrs=`fill="${open || shape==='raw'?'white':color}" stroke="${color}" stroke-width="1.4" ${extra}`;
  const tip=title?`<title>${esc(title)}</title>`:'';
  if(shape==='bit') return `<polygon points="${x},${y-r} ${x-r},${y+r} ${x+r},${y+r}" ${attrs}>${tip}</polygon>`;
  if(shape==='byte') return `<circle cx="${x}" cy="${y}" r="${r}" ${attrs}>${tip}</circle>`;
  if(shape==='raw') return `<polygon points="${x},${y-r} ${x-r},${y} ${x},${y+r} ${x+r},${y}" ${attrs}>${tip}</polygon>`;
  return `<rect x="${x-r}" y="${y-r}" width="${r*2}" height="${r*2}" ${attrs}>${tip}</rect>`;
}
function legend() {
  const neutral=style.muted;
  const out=['<g role="group" aria-label="Figure legend">',
    `<rect x="48" y="96" width="1404" height="96" rx="8" fill="${style.legend_fill}" stroke="${style.legend_border}"/>`];
  for(const [x,codec] of [[72,'blosc-lz4'],[212,'blosc-zstd']])
    out.push(line(x,126,x+32,126,`stroke="${colors[codec]}" stroke-width="3"`),text(x+43,131,codec.replace('blosc-','').toUpperCase(),'class="legend-label"'));
  for(const [x,shape,label] of [[390,'none','No shuffle'],[570,'byte','Byte shuffle'],[760,'bit','Bitshuffle'],[940,'raw','Raw codec control']])
    out.push(marker(x,126,shape,neutral),text(x+17,131,label,'class="legend-label"'));
  out.push(line(72,166,112,166,`stroke="${neutral}" stroke-width="2.5"`),
    marker(92,166,'bit',neutral),text(125,171,'RTX 5080 frontier','class="legend-label"'),
    line(390,166,430,166,`stroke="${neutral}" stroke-width="2.5" stroke-dasharray="7 5" opacity=".65"`),
    marker(410,166,'bit',neutral,5,{open:true,extra:'opacity=".65"'}),text(443,171,'5070 Laptop frontier','class="legend-label"'),
    marker(760,166,'bit',neutral,5,{extra:'opacity=".23"'}),text(777,171,'Dominated candidate','class="legend-label"'),
    line(1100,155,1100,177,`stroke="${neutral}" stroke-width="1.4"`),
    line(1095,155,1105,155,`stroke="${neutral}" stroke-width="1.4"`),
    line(1095,177,1105,177,`stroke="${neutral}" stroke-width="1.4"`),
    text(1117,171,'Min–max of 3 runs','class="legend-label"'),'</g>');
  return out.join('\n');
}
function chart(zoom=true) {
  const svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1220" viewBox="0 0 1500 1220" role="img" aria-labelledby="title desc">',
    '<title id="title">RTX 5080 and RTX 5070 Laptop Blosc Pareto frontiers</title>',
    '<desc id="desc">Four panels compare throughput and compression fold across codecs, filters, and block sizes. Graphical legends identify codec colors, filter shapes, machine line styles, dominated points, and timing ranges.</desc>',
    `<style>${style.css}</style>`,
    '<rect width="1500" height="1220" fill="white"/>',
    text(48,43,'Blosc throughput–compression frontier: RTX 5080 vs RTX 5070 Laptop','class="title"'),
    text(48,72,'Same 200 configurations · u16 · 100 frames · discard sink · 1 warmup + 3 measured runs','class="subtitle"'),
    legend(),
    text(48,217,'Labels: block KiB · per-codec frontier across all filters · higher and farther right is better','class="muted"')];
  panels.forEach(([fill,chunk],i) => {
    const ox=30+(i%2)*745, oy=250+Math.floor(i/2)*430;
    const left=ox+62,right=ox+677,top=oy+48,bottom=oy+353;
    const current=rows.filter(r=>r.fill===fill && r.chunk_kib===chunk);
    const old=historical.filter(r=>r.fill===fill && r.chunk_kib===chunk);
    const xmax=fill==='xor'?500:zoom?(chunk===256?1.39:1.497):1.55;
    const xmin=fill==='xor'?1:zoom?(chunk===256?1.367:1.47):1;
    const ymax=Math.ceil(Math.max(...current.map(r=>r.hi),...old.map(r=>r.hi))/2)*2;
    const transform=fill==='xor'?Math.log10:x=>x;
    const x=v=>left+(transform(v)-transform(xmin))/(transform(xmax)-transform(xmin))*(right-left);
    const y=v=>bottom-v/ymax*(bottom-top);
    svg.push(text(ox+18,oy+7,`${fill==='xor'?'Repetitive XOR':'12-bit random'} · ${chunk===256?'256 KiB':'1 MiB'} Zarr chunks`,'class="panel-title"'),
      text(ox+18,oy+28,fill==='xor'?'log ratio axis':zoom?'zoomed ratio axis':'linear ratio axis','class="muted"'));
    const xticks=fill==='xor'?[1,2,5,10,20,50,100,200,500]:zoom?(chunk===256?[1.37,1.375,1.38,1.385,1.39]:[1.475,1.48,1.485,1.49,1.495]):[1,1.1,1.2,1.3,1.4,1.5];
    for(let v=0;v<=ymax;v+=2) svg.push(line(left,y(v),right,y(v),'class="grid"'),text(left-10,y(v)+5,v,'text-anchor="end"'));
    for(const v of xticks) svg.push(line(x(v),top,x(v),bottom,'class="grid"'),text(x(v),bottom+24,`${v}×`,'text-anchor="middle"'));
    svg.push(text((left+right)/2,bottom+52,'Compression fold →','class="axis" text-anchor="middle"'),`<text transform="translate(${ox+13},${(top+bottom)/2}) rotate(-90)" class="axis" text-anchor="middle">Input throughput (GiB/s) →</text>`);
    const visible=r=>r.fold>=xmin && r.fold<=xmax;
    for(const [machine,rs] of [['old',old],['current',current]]) {
      svg.push(`<g class="${machine}">`);
      for(const codec of Object.keys(colors)) {
        const pool=rs.filter(r=>r.codec===codec), front=frontier(pool), frontIds=new Set(front.map(id));
        const color=colors[codec];
        svg.push(`<polyline points="${front.map(r=>`${x(r.fold)},${y(r.speed)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" ${machine==='old'?'stroke-dasharray="7 5"':''}/>`);
        for(const r of pool.filter(visible)) {
          const isFront=frontIds.has(id(r)), cx=x(r.fold),cy=y(r.speed);
          const tip=`${machine==='old'?'5070 Laptop':'5080'} ${r.codec}, ${r.block_kib} KiB ${r.shuffle}: ${fmt(r.speed)} GiB/s [${fmt(r.lo)}, ${fmt(r.hi)}], ${fmt(r.fold,5)}×, ${fmt(r.device_gib)} GiB observed, ${fmt(r.estimate_gib)} GiB allocated`;
          if(isFront && machine==='current') svg.push(line(cx,y(r.lo),cx,y(r.hi),`stroke="${color}"`),line(cx-4,y(r.lo),cx+4,y(r.lo),`stroke="${color}"`),line(cx-4,y(r.hi),cx+4,y(r.hi),`stroke="${color}"`));
          svg.push(marker(cx,cy,r.shuffle,color,isFront?6:4,{open:machine==='old'&&isFront,extra:`class="${isFront?'front':'candidate'}"`,title:tip}));
        }
        if(machine==='current') {
          // Place labels in separate columns for each codec and spread vertically.
          const labels=front.filter(visible).sort((a,b)=>b.speed-a.speed);
          const labelX=codec==='blosc-lz4'?left+12:right-8;
          let prev=top-24;
          labels.forEach((r,j)=>{
            const ly=Math.min(bottom-(labels.length-1-j)*21,Math.max(top+14,prev+21,y(r.speed)-12)); prev=ly;
            const anchor=codec==='blosc-lz4'?'start':'end';
            svg.push(line(x(r.fold),y(r.speed),labelX,ly-4,`stroke="${color}" opacity=".45"`),text(labelX,ly,`${r.block_kib} KiB`,`class="point-label" text-anchor="${anchor}" style="fill:${color}"`));
          });
        }
      }
      for(const r of rs.filter(r=>!r.block_kib && visible(r))) {
        const cx=x(r.fold),cy=y(r.speed),color=colors[`blosc-${r.codec}`];
        svg.push(marker(cx,cy,'raw',color,5,{title:`${machine} raw ${r.codec}: ${fmt(r.speed)} GiB/s, ${r.fold}×`}));
      }
      svg.push('</g>');
    }
  });
  svg.push(text(48,1140,'Empirical frontiers over tested settings. Different OS, driver, compiler, and source revision; speedups are whole-system comparisons.','class="note"'),
    text(48,1162,'Fold = padded chunk bytes / sink bytes; compare within each panel. Memory is not an objective here.','class="note"'),
    text(48,1184,'Lines connect tested settings only. Current timing ranges are not confidence intervals.','class="note"'),
    text(48,1206,zoom?'Random panels zoom to the frontiers; off-scale candidates and raw controls are omitted.':'All measured candidates shown. Close frontier rankings are sensitive to timing and ratio precision.','class="note"'),'</svg>');
  return svg.join('\n');
}
const svg=chart();
fs.writeFileSync(path.join(here,'pareto-comparison.svg'),svg+'\n');
fs.writeFileSync(path.join(here,'pareto-all-candidates.svg'),chart(false)+'\n');
const frontRows=rows.filter(r=>r.overall_frontier);
const table=frontRows.map(r=>`<tr><td>${r.fill}</td><td>${r.chunk_kib}</td><td>${r.codec}</td><td>${r.shuffle}</td><td>${r.block_kib}</td><td>${fmt(r.speed)}</td><td>${fmt(r.lo)}–${fmt(r.hi)}</td><td>${fmt(r.fold,5)}</td><td>${fmt(r.estimate_gib)}</td></tr>`).join('');
fs.writeFileSync(path.join(here,'pareto.html'),`<!doctype html><html lang="en"><meta charset="utf-8"><title>RTX 5080 Blosc comparison</title><style>body{font:16px system-ui;margin:24px;color:#293744}svg{width:100%;height:auto}label{margin-right:25px}table{border-collapse:collapse}th,td{text-align:right;padding:7px 12px;border-bottom:1px solid #ddd}h1{font-size:25px}.hide-old .old,.hide-candidates .candidate{display:none}</style><h1>RTX 5080 Blosc frontier and historical comparison</h1><p>Hover over points for settings, throughput ranges, ratios, and memory. <a href="README.md">Analysis and methodology</a> · <a href="summary.csv">All results</a> · <a href="pareto-all-candidates.svg">Unzoomed chart</a></p><label><input id="old" type="checkbox" checked> Historical 5070 Laptop</label><label><input id="candidates" type="checkbox" checked> Dominated candidates</label>${svg}<h2>Cross-codec frontier on this machine</h2><table><thead><tr>${['Input','Chunk KiB','Codec','Shuffle','Block KiB','GiB/s','Min–max','Fold','Allocated GiB'].map(s=>`<th>${s}</th>`).join('')}</tr></thead><tbody>${table}</tbody></table><p>Memory excludes runtime headroom. Bars are repetition ranges, not confidence intervals.</p><script>for(const id of ['old','candidates'])document.getElementById(id).addEventListener('change',e=>document.body.classList.toggle('hide-'+id,!e.target.checked));</script></html>\n`);
console.log(JSON.stringify(stats,null,2));
