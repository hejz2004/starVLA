"use strict";
const $ = id => document.getElementById(id);
const videos = [$("head"), $("left"), $("right")];
const colors = ["#8aa56f","#bad099","#7fa888","#c5b36e","#a4bd82","#6b9a7e","#c0c9a0"];
let data = null, now = 0, playing = false, currentSegment = -1, offset = 0, total = 0;
let requestVersion = 0, searchVersion = 0, listAbort, episodeAbort, activeURL, progressTimer;
let syncBusy = false;
const stageLabel = s => s.startsWith("stage_") ? "训练 · Stage " + s.slice(-2) : s === "validation" ? "验证集" : "未纳入当前索引";
const fmt = v => { v=Math.max(0,v||0); return String(Math.floor(v/60)).padStart(2,"0")+":"+(v%60).toFixed(2).padStart(5,"0"); };
const text = (id, value) => { $(id).textContent = value; };
function notice(message=""){ text("notice",message); $("notice").hidden=!message; }
async function getJSON(url, signal){
  const r=await fetch(url,{signal});
  if(!r.ok){let body;try{body=await r.json();}catch{body={};}throw Error(body.detail||"请求失败："+r.status);}
  return r.json();
}
function node(tag, cls, value){ const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=value;return n; }
async function search(reset=true){
  if(reset)offset=0;
  const version=++searchVersion;
  if(listAbort)listAbort.abort();
  listAbort=new AbortController();
  text("result-count","正在搜索…");
  try{
    const p=new URLSearchParams({q:$("search").value.trim(),stage:$("stage").value,offset,limit:30});
    const result=await getJSON("/api/episodes?"+p,listAbort.signal);
    if(version!==searchVersion)return;
    total=result.total;
    text("result-count",total.toLocaleString()+" 条轨迹");
    text("page",Math.floor(offset/30)+1);
    $("prev").disabled=offset===0;$("next").disabled=offset+30>=total;
    $("episode-list").replaceChildren();
    if(!result.items.length)$("episode-list").append(node("p","empty","没有匹配结果，试试其他任务词或 episode ID。"));
    for(const ep of result.items){
      const b=node("button","episode-item"+(data?.id===ep.id?" active":""));
      b.dataset.id=ep.id;
      const top=node("div","topline");top.append(node("span","mono","#"+String(ep.id).padStart(6,"0")),node("span","",ep.duration.toFixed(1)+" s"));
      b.append(top,node("p","",ep.task));b.title=ep.task;b.onclick=()=>loadEpisode(ep.id);
      $("episode-list").append(b);
    }
  }catch(e){if(e.name!=="AbortError"){text("result-count","搜索暂不可用");$("episode-list").replaceChildren(node("p","empty",e.message));}}
}
async function updateSummary(){
  try{
    const s=await getJSON("/api/summary");
    text("total-stat",s.total_episodes.toLocaleString());
    text("index-status",s.index.ready ? "搜索目录就绪 · "+s.train_episodes.toLocaleString()+" 条训练轨迹" : s.index.error ? "搜索索引错误："+s.index.error : "建立目录："+s.index.indexed.toLocaleString()+" / "+s.total_episodes.toLocaleString());
    if(!s.index.ready)setTimeout(updateSummary,3500);else search(false);
  }catch(e){notice("连接失败："+e.message);}
}
function pause(){
  playing=false;videos.forEach(v=>v.pause());text("play","▶ 播放");
}
function ready(v){
  return new Promise((resolve,reject)=>{
    if(v.readyState>=1)return resolve();
    const timer=setTimeout(()=>finish(Error("视频元数据加载超时")),45000);
    const ok=()=>finish(),bad=()=>finish(Error("视频无法加载，请检查远程文件或浏览器解码支持"));
    function finish(error){clearTimeout(timer);v.removeEventListener("loadedmetadata",ok);v.removeEventListener("error",bad);error?reject(error):resolve();}
    v.addEventListener("loadedmetadata",ok,{once:true});v.addEventListener("error",bad,{once:true});
  });
}
function cameraFor(v){
  const name=v.id==="head"?$("head-camera").value:v.id==="left"?"wrist_left":"wrist_right";
  return data.cameras.find(c=>c.name===name);
}
async function attachCamera(v, version){
  const camera=cameraFor(v);
  if(!camera)throw Error("缺少相机 "+v.id);
  v.dataset.start=camera.start;v.dataset.end=camera.end;
  v.src=camera.url+"#t="+camera.start.toFixed(6);
  v.playbackRate=Number($("speed").value);
  await ready(v);
  if(version!==requestVersion)return;
  v.currentTime=camera.start+now;
}
async function loadEpisode(id){
  const version=++requestVersion;
  if(episodeAbort)episodeAbort.abort();episodeAbort=new AbortController();
  pause();data=null;now=0;currentSegment=-1;
  clearTimeout(progressTimer);
  videos.forEach(v=>{v.removeAttribute("src");v.load();});
  $("play").disabled=true;$("scrubber").disabled=true;
  notice("正在读取 episode #"+id+" 的逐帧数据与 Luna 标注…");
  text("episode-title","Episode "+String(id).padStart(6,"0"));
  text("task","正在读取原始任务描述…");
  $("segment-list").replaceChildren();$("timeline").replaceChildren();
  for(const k of ["stage-badge","duration-meta","frames-meta","coverage-meta","current-instruction","current-segment","window-status","source-info"])text(k,"—");
  $("export").removeAttribute("href");
  try{
    const d=await getJSON("/api/episode/"+id,episodeAbort.signal);
    if(version!==requestVersion)return;
    data=d;
    const url=new URL(location.href);url.searchParams.set("episode",id);history.replaceState(null,"",url);
    text("task",d.task);text("stage-badge",stageLabel(d.stage));
    text("duration-meta",d.duration.toFixed(2)+" 秒");text("frames-meta",d.frames.toLocaleString()+" 帧");
    text("coverage-meta","有效标注 "+(100*d.valid_frames/d.frames).toFixed(1)+"%");
    text("segment-count",d.segments.length);text("mid-time",(d.duration/2).toFixed(1)+" s");text("end-time",d.duration.toFixed(1)+" s");
    $("scrubber").max=Math.max(0,d.duration-1/d.fps);$("scrubber").value=0;$("scrubber").disabled=false;
    $("annotation-filter").value="";renderSegments();
    text("source-info",JSON.stringify(d.source,null,2));
    document.querySelectorAll(".episode-item").forEach(el=>el.classList.toggle("active",Number(el.dataset.id)===d.id));
    if(activeURL)URL.revokeObjectURL(activeURL);
    activeURL=URL.createObjectURL(new Blob([JSON.stringify({episode_index:d.id,task:d.task,fps:d.fps,duration:d.duration,stage:d.stage,source:d.source,segments:d.segments},null,2)],{type:"application/json"}));
    $("export").href=activeURL;$("export").download="umi_"+d.id+"_luna.json";
    renderTime(0);notice();
    text("video-status","正在加载三路原始视频…");text("head-tag",$("head-camera").value.replace("_"," ").toUpperCase());
    await Promise.all(videos.map(v=>attachCamera(v,version)));
    if(version!==requestVersion)return;
    $("play").disabled=false;
    text("video-status","三视角已连接 · 按相机时间偏移同步");
  }catch(e){
    if(version!==requestVersion||e.name==="AbortError")return;
    notice(e.message);text("video-status",data?"视频加载失败；标注和曲线仍可浏览":"读取失败");
  }
}
function renderSegments(){
  if(!data)return;
  const query=$("annotation-filter").value.trim().toLowerCase();
  $("segment-list").replaceChildren();$("timeline").replaceChildren();
  data.segments.forEach((s,i)=>{
    const color=s.valid?colors[i%colors.length]:"#c9cec7";
    const t=node("button");t.style.width=((s.end-s.start)/data.duration*100)+"%";t.style.background=color;
    t.title=fmt(s.start)+" – "+fmt(s.end)+" · "+s.instruction;t.setAttribute("aria-label",t.title);t.dataset.segment=i;t.onclick=()=>selectSegment(i);$("timeline").append(t);
    if(query&&!s.instruction.toLowerCase().includes(query))return;
    const b=node("button","segment-item"+(!s.valid?" invalid":""));b.dataset.segment=i;
    const top=node("div");top.append(node("span","segment-num",String(i+1).padStart(2,"0")+(s.valid?"":" · 无效")),node("span","mono",fmt(s.start)+" → "+fmt(s.end)));
    b.append(top,node("p","",s.instruction));b.onclick=()=>selectSegment(i);$("segment-list").append(b);
  });
  if(!$("segment-list").children.length)$("segment-list").append(node("p","empty","没有匹配的标注片段。"));
  currentSegment=-1;renderTime(now);
}
function segmentAt(t){return data?data.segments.findIndex(s=>s.start<=t+1e-7&&t<s.end): -1;}
function selectSegment(i){pause();seek(data.segments[i].start);}
function seek(t){
  if(!data)return;
  t=Math.max(0,Math.min(t,data.duration-1/data.fps));
  videos.forEach(v=>{if(v.readyState>=1)v.currentTime=Number(v.dataset.start)+t;});
  renderTime(t);
}
function rowAt(t){
  let lo=0,hi=data.timestamps.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(data.timestamps[mid]<=t+1e-6)lo=mid+1;else hi=mid;}
  return Math.max(0,lo-1);
}
function renderTime(t){
  if(!data)return;
  now=Math.max(0,Math.min(t,data.duration));
  $("scrubber").value=now;text("time",fmt(now)+" / "+fmt(data.duration));
  const idx=segmentAt(now), row=rowAt(now);
  const anchor=data.training_ranges.some(([a,b])=>row>=a&&row<b&&(row-a)%data.anchor_stride===0);
  text("window-status",anchor?"✓ 当前帧为训练 anchor · t+1…t+16":"当前帧非已采样训练 anchor");
  if(idx!==currentSegment){
    currentSegment=idx;
    const s=data.segments[idx];
    text("current-instruction",s?.instruction||"此时间点无标注");
    text("current-segment",s?"片段 "+(idx+1)+" / "+data.segments.length+" · "+(s.valid?"有效标注":"无效标注")+" · "+fmt(s.start)+"–"+fmt(s.end):"—");
    document.querySelectorAll("[data-segment]").forEach(el=>el.classList.toggle("active",Number(el.dataset.segment)===idx));
    const active=$("segment-list").querySelector(".active");
    if(active&&playing){const box=$("segment-list");box.scrollTop=active.offsetTop-box.offsetTop-box.clientHeight/3;}
  }
  const a=data.signals.robot1_sensor_magnetic_encoder[row],b=data.signals.robot2_sensor_magnetic_encoder[row];
  text("frame-info","f "+data.frame_indices[row]+" · "+(a==null?"—":a.toFixed(4))+" / "+(b==null?"—":b.toFixed(4)));
  drawCharts();
}
async function togglePlay(){
  if(!data||$("play").disabled)return;
  if(playing){pause();return;}
  if(now>=data.duration-1/data.fps)seek(0);
  const version=requestVersion;playing=true;
  try{
    await Promise.all(videos.map(v=>v.play()));
    if(version!==requestVersion||!playing){pause();return;}
    text("play","Ⅱ 暂停");notice();
  }catch(e){if(version===requestVersion){pause();notice("播放未开始："+e.message);}}
}
function tick(){
  requestAnimationFrame(tick);
  if(!data||!playing||syncBusy)return;
  const main=videos[0];
  if(main.seeking)return;
  let t=main.currentTime-Number(main.dataset.start);
  const s=data.segments[currentSegment];
  if($("loop").checked&&s&&t>=s.end-0.012){seek(s.start);return;}
  if(t>=data.duration-1/data.fps){pause();seek(data.duration-1/data.fps);return;}
  if(t<0){seek(0);return;}
  videos.slice(1).forEach(v=>{
    const delta=v.currentTime-Number(v.dataset.start)-t;
    if(!v.seeking&&Math.abs(delta)>.12)v.currentTime=Number(v.dataset.start)+t;
  });
  renderTime(t);
}
function drawChart(canvas, series){
  const width=canvas.clientWidth;if(!width)return;
  const height=155,dpr=window.devicePixelRatio||1;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,width,height);
  const pad={l:40,r:9,t:10,b:20},w=width-pad.l-pad.r,h=height-pad.t-pad.b;
  let min=Infinity,max=-Infinity;
  for(const s of series)for(const v of s.values)if(v!=null&&Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}
  if(!Number.isFinite(min)){min=0;max=1;}if(max-min<1e-8)max=min+1;
  const margin=(max-min)*.08;min-=margin;max+=margin;
  c.font="9px -apple-system, sans-serif";
  for(let i=0;i<=3;i++){const y=pad.t+i*h/3;c.strokeStyle="#edf0e8";c.beginPath();c.moveTo(pad.l,y);c.lineTo(width-pad.r,y);c.stroke();c.fillStyle="#97a18f";c.fillText((max-(max-min)*i/3).toFixed(2),0,y+3);}
  for(let i=0;i<=4;i++){const x=pad.l+w*i/4;c.fillStyle="#97a18f";c.fillText((data.duration*i/4).toFixed(0)+"s",x-6,height-3);}
  for(const s of series){
    c.strokeStyle=s.color;c.lineWidth=1.2;c.beginPath();let pen=false;
    const step=Math.max(1,Math.ceil(s.values.length/900));
    for(let i=0;i<s.values.length;i+=step){const val=s.values[i];if(val==null||!Number.isFinite(val)){pen=false;continue;}const x=pad.l+data.timestamps[i]/data.duration*w,y=pad.t+(max-val)/(max-min)*h;if(pen)c.lineTo(x,y);else c.moveTo(x,y);pen=true;}c.stroke();
  }
  const x=pad.l+now/data.duration*w;c.setLineDash([3,3]);c.strokeStyle="#315d47";c.lineWidth=1;c.beginPath();c.moveTo(x,pad.t);c.lineTo(x,pad.t+h);c.stroke();c.setLineDash([]);
}
let plottedData=null,positions,grippers;
function drawCharts(){
  if(!data)return;
  if(plottedData!==data){
    plottedData=data;positions=[];grippers=[];
    const palette=[["#198677","#62ad91","#aacabd"],["#cf7849","#e3a565","#efca9f"]];
    for(let arm=1;arm<=2;arm++){
      for(let axis=0;axis<3;axis++)positions.push({values:data.signals["robot"+arm+"_finger_eef_pose"].map(v=>v?.[axis]??null),color:palette[arm-1][axis]});
      grippers.push({values:data.signals["robot"+arm+"_sensor_magnetic_encoder"],color:palette[arm-1][0]});
    }
  }
  drawChart($("position-chart"),positions);drawChart($("gripper-chart"),grippers);
}
$("search-form").onsubmit=e=>{e.preventDefault();search();const q=$("search").value.trim();if(/^\d+$/.test(q))loadEpisode(Number(q));};
$("stage").onchange=()=>search();
$("prev").onclick=()=>{offset=Math.max(0,offset-30);search(false);};
$("next").onclick=()=>{offset+=30;search(false);};
$("random").onclick=async()=>{
  try{
    const p=new URLSearchParams({q:$("search").value.trim(),stage:$("stage").value,limit:1});
    const r=await getJSON("/api/episodes?"+p);if(!r.total)return notice("当前筛选没有可探索的轨迹。");
    p.set("offset",Math.floor(Math.random()*r.total));const sample=await getJSON("/api/episodes?"+p);loadEpisode(sample.items[0].id);
  }catch(e){notice(e.message);}
};
$("play").onclick=togglePlay;
$("back-frame").onclick=()=>{pause();if(data)seek((Math.round(now*data.fps)-1)/data.fps);};
$("next-frame").onclick=()=>{pause();if(data)seek((Math.round(now*data.fps)+1)/data.fps);};
$("scrubber").oninput=()=>{pause();seek(Number($("scrubber").value));};
$("speed").onchange=()=>videos.forEach(v=>v.playbackRate=Number($("speed").value));
$("annotation-filter").oninput=renderSegments;
$("head-camera").onchange=async()=>{
  if(!data)return;
  pause();$("play").disabled=true;
  const version=requestVersion;
  try{await attachCamera(videos[0],version);if(version===requestVersion){text("head-tag",$("head-camera").value.replace("_"," ").toUpperCase());$("play").disabled=false;}}
  catch(e){if(version===requestVersion)notice(e.message);}
};
for(const id of ["position-chart","gripper-chart"])$(id).onclick=e=>{
  if(!data)return;const rect=e.target.getBoundingClientRect();pause();seek((e.clientX-rect.left-40)/(rect.width-49)*data.duration);
};
for(const v of videos){
  v.addEventListener("error",()=>{if(data&&v.getAttribute("src"))text("video-status","部分视频加载失败，可尝试重新选择 episode；标注仍可浏览。");});
  v.addEventListener("waiting",()=>{if(playing)text("video-status","视频缓冲中…");});
  v.addEventListener("playing",()=>text("video-status","三视角已连接 · 按相机时间偏移同步"));
}
document.addEventListener("keydown",e=>{
  if(["INPUT","SELECT","TEXTAREA","BUTTON","A"].includes(document.activeElement.tagName))return;
  if(e.code==="Space"){e.preventDefault();togglePlay();}
  if(e.code==="ArrowLeft"){e.preventDefault();$("back-frame").click();}
  if(e.code==="ArrowRight"){e.preventDefault();$("next-frame").click();}
});
window.addEventListener("resize",drawCharts);
const initial=Number(new URLSearchParams(location.search).get("episode")||24348);
updateSummary();loadEpisode(Number.isSafeInteger(initial)&&initial>=0?initial:24348);requestAnimationFrame(tick);
