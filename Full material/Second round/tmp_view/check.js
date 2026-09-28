
/* ------------------------------------------------------------------ deck */
const slides=[...document.querySelectorAll('.slide')];let cur=0;
function fit(){const s=Math.min(innerWidth/1300,innerHeight/740);document.getElementById('stage').style.transform=`translate(-50%,-50%) scale(${s})`;}
addEventListener('resize',fit);fit();
function show(i){cur=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,k)=>s.classList.toggle('active',k===cur));
  document.getElementById('bar').style.width=((cur+1)/slides.length*100)+'%';
  document.getElementById('num').textContent=(cur+1)+' / '+slides.length+' · '+slides[cur].dataset.title;
  const n=slides[cur].querySelector('.notes');document.getElementById('notes').innerHTML=n?('<b>Notes:</b> '+n.innerHTML):'';
  try{location.hash=cur+1}catch(e){}
  onSlide(cur);}
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(cur+1)}
  if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(cur-1)}
  if(e.key==='n'||e.key==='N'){const el=document.getElementById('notes');el.style.display=el.style.display==='block'?'none':'block'}
  if(e.key==='o'||e.key==='O'){overview()}
  if(e.key==='Home')show(0); if(e.key==='End')show(slides.length-1);});
document.getElementById('prev').onclick=()=>show(cur-1);document.getElementById('next').onclick=()=>show(cur+1);
let tx=null;addEventListener('touchstart',e=>{tx=e.touches[0].clientX});addEventListener('touchend',e=>{if(tx===null)return;const d=e.changedTouches[0].clientX-tx;if(Math.abs(d)>50)show(cur+(d<0?1:-1));tx=null});
function overview(){const t=slides.map((s,k)=>`${k+1}. ${s.dataset.title}`).join('\n');const k=prompt('Go to slide:\n'+t,cur+1);if(k)show(parseInt(k)-1);}
document.addEventListener('DOMContentLoaded',()=>{if(window.renderMathInElement)renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});});
window.addEventListener('load',()=>{if(window.renderMathInElement)renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});});

const C={blue:'#2a78d6',orange:'#eb6834',aqua:'#1baf7a',yellow:'#eda100',ink:'#0b0b0b',ink2:'#52514e',muted:'#8a8984',rule:'#dedcd6',red:'#c43b3a'};
function hidpi(cv){const r=window.devicePixelRatio||1;if(cv._s)return cv.getContext('2d');const w=cv.width,h=cv.height;cv.style.width=w+'px';cv.style.height=h+'px';cv.width=w*r;cv.height=h*r;const g=cv.getContext('2d');g.scale(r,r);cv._s=1;cv._w=w;cv._h=h;return g;}

/* ------------------------------------------------------------------ physics helpers (M = 1) */
function wallTrajectory(Rm){           // domain wall from rest at Rm; returns arrays of (tau, t, R)
  const M=1, sm=Math.sqrt(1-2*M/Rm), a0=(1-sm)/Rm;        // a(R) = 4 pi sigma R = a0 R
  const rate=R=>{const a=a0*R, al=(2*M/(a*R)+a)/2; return Math.sqrt(Math.max(al*al-1,0));};
  const beta=R=>{const a=a0*R, al=(2*M/(a*R)+a)/2; return al-a;};
  const Rend=0.03, umax=Math.sqrt(Rm-Rend), N=4000;
  const T=[0],Tt=[0],RR=[Rm]; let tau=0,t=0,frozen=false;
  for(let i=0;i<N;i++){const u0=umax*i/N,u1=umax*(i+1)/N,um=(u0+u1)/2,R=Rm-um*um;
    const d=2*um*(u1-u0)/Math.max(rate(R),1e-12); tau+=d;
    const f=1-2*M/R; if(!frozen&&f>1e-9){t+=d*beta(R)/f;} else frozen=true;
    T.push(tau);Tt.push(frozen?Infinity:t);RR.push(Rm-u1*u1);}
  return {tau:T,t:Tt,R:RR};
}
function ringMode(R){                   // model-C matter-mode frequencies (l=2, Gamma=2), fits to the verified data
  const x=1/R, sc=Math.pow(R,-1.5);
  return {gamma:(0.5147+0.41*x+0.36*x*x)*sc, wst:(1.505-0.95*x-1.2*x*x)*sc};
}

/* ------------------------------------------------------------------ slide 4: collapse animation */
const A={play:false,t:0,Rm:6,traj:null,last:null};
function aInit(){A.Rm=parseFloat(document.getElementById('aR').value);document.getElementById('aRv').textContent=A.Rm.toFixed(1);
  A.traj=wallTrajectory(A.Rm);A.t=0;aDraw();}
function interp(xs,ys,x){if(x<=xs[0])return ys[0];for(let i=1;i<xs.length;i++){if(xs[i]>=x){const s=(x-xs[i-1])/(xs[i]-xs[i-1]);return ys[i-1]+s*(ys[i]-ys[i-1]);}}return ys[ys.length-1];}
function arrow(g,x0,y0,x1,y1,col){g.strokeStyle=col;g.fillStyle=col;g.lineWidth=2;g.beginPath();g.moveTo(x0,y0);g.lineTo(x1,y1);g.stroke();
  const a=Math.atan2(y1-y0,x1-x0);g.beginPath();g.moveTo(x1,y1);g.lineTo(x1-8*Math.cos(a-0.4),y1-8*Math.sin(a-0.4));g.lineTo(x1-8*Math.cos(a+0.4),y1-8*Math.sin(a+0.4));g.closePath();g.fill();}
function aDraw(){const cv=document.getElementById('cvA'),g=hidpi(cv),W=cv._w,H=cv._h;g.clearRect(0,0,W,H);
  const Rm=A.Rm,tr=A.traj,tauEnd=tr.tau[tr.tau.length-1];
  // ---- left: wall, drawn in its own proper time tau
  const tau=Math.min(A.t,tauEnd), Rw=interp(tr.tau,tr.R,tau);
  const cx=175,cy=205,scale=150/Rm;
  g.fillStyle='#f7f6f2';g.fillRect(0,0,560,H);
  g.fillStyle=C.ink;g.font='600 17px Segoe UI, sans-serif';g.fillText('Model A: domain wall, released at its turning point',16,26);
  // horizon
  if(Rw<2){g.fillStyle='#111';g.globalAlpha=0.85;g.beginPath();g.arc(cx,cy,2*scale,0,7);g.fill();g.globalAlpha=1;
    g.fillStyle='#fff';g.font='13px Segoe UI';g.textAlign='center';g.fillText('black hole',cx,cy+4);g.textAlign='left';}
  g.setLineDash([5,5]);g.strokeStyle=C.muted;g.lineWidth=1.2;g.beginPath();g.arc(cx,cy,2*scale,0,7);g.stroke();g.setLineDash([]);
  g.fillStyle='rgba(42,120,214,0.10)';g.beginPath();g.arc(cx,cy,Rw*scale,0,7);g.fill();
  g.strokeStyle=C.ink;g.lineWidth=4;g.beginPath();g.arc(cx,cy,Math.max(Rw*scale,1),0,7);g.stroke();
  if(Rw>2.2){for(let k=0;k<8;k++){const th=k*Math.PI/4+0.2,x0=cx+(Rw*scale+22)*Math.cos(th),y0=cy+(Rw*scale+22)*Math.sin(th);
      arrow(g,x0,y0,cx+(Rw*scale+4)*Math.cos(th),cy+(Rw*scale+4)*Math.sin(th),C.orange);}}
  g.fillStyle=C.ink2;g.font='14px Segoe UI';
  g.fillText('R = '+Rw.toFixed(2)+' M',16,H-62);g.fillText('proper time of the wall  τ = '+tau.toFixed(1)+' M',16,H-42);
  const tFar=interp(tr.tau,tr.t,tau);g.fillText('distant clock  t = '+(isFinite(tFar)?tFar.toFixed(1)+' M':'∞ (frozen at r = 2M)'),16,H-22);
  g.fillStyle=C.orange;g.fillText('tension + gravity pull inward',330,H-22);
  // mini plot R(tau)
  const px=350,py=50,pw=190,ph=120;g.strokeStyle=C.rule;g.strokeRect(px,py,pw,ph);
  g.strokeStyle=C.blue;g.lineWidth=2;g.beginPath();for(let i=0;i<tr.tau.length;i+=20){const x=px+pw*tr.tau[i]/tauEnd,y=py+ph*(1-tr.R[i]/Rm);i?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();
  g.setLineDash([3,3]);g.strokeStyle=C.muted;g.beginPath();g.moveTo(px,py+ph*(1-2/Rm));g.lineTo(px+pw,py+ph*(1-2/Rm));g.stroke();g.setLineDash([]);
  g.fillStyle=C.red;g.beginPath();g.arc(px+pw*tau/tauEnd,py+ph*(1-Rw/Rm),4,0,7);g.fill();
  g.fillStyle=C.muted;g.font='12px Segoe UI';g.fillText('R(τ)/R_m',px+4,py+14);g.fillText('r = 2M',px+pw-44,py+ph*(1-2/Rm)-4);
  g.fillText('collapse in τ = '+tauEnd.toFixed(1)+' M',px,py+ph+16);
  // ---- right: model C, exterior time
  const ox=580;g.fillStyle=C.ink;g.font='600 17px Segoe UI';g.fillText('Model C: static fluid shell, same radius, perturbed (ℓ = 2)',ox+16,26);
  const md=ringMode(Rm),mode=document.getElementById('aMode').value,tc=A.t;
  let eps; if(mode==='u'){eps=Math.min(0.004*Math.exp(md.gamma*tc),0.45);} else {eps=0.12*Math.cos(md.wst*tc)*Math.exp(-0.001*tc);}
  const cx2=ox+200,cy2=205,Rpx=150;
  g.setLineDash([5,5]);g.strokeStyle=C.muted;g.lineWidth=1.2;g.beginPath();g.arc(cx2,cy2,Rpx,0,7);g.stroke();g.setLineDash([]);
  g.fillStyle='rgba(42,120,214,0.10)';g.strokeStyle=C.ink;g.lineWidth=4;g.beginPath();
  for(let k=0;k<=240;k++){const th=2*Math.PI*k/240,P2=0.5*(3*Math.cos(th)**2-1),r=Rpx*(1+eps*P2);const x=cx2+r*Math.sin(th),y=cy2-r*Math.cos(th);k?g.lineTo(x,y):g.moveTo(x,y);}
  g.closePath();g.fill();g.stroke();
  for(let k=0;k<6;k++){const th=k*Math.PI/3+0.5,x0=cx2+(Rpx-6)*Math.cos(th),y0=cy2+(Rpx-6)*Math.sin(th);
    arrow(g,x0,y0,cx2+(Rpx+22)*Math.cos(th),cy2+(Rpx+22)*Math.sin(th),C.aqua);
    arrow(g,cx2+(Rpx+44)*Math.cos(th+0.25),cy2+(Rpx+44)*Math.sin(th+0.25),cx2+(Rpx+24)*Math.cos(th+0.25),cy2+(Rpx+24)*Math.sin(th+0.25),C.orange);}
  g.fillStyle=C.ink2;g.font='14px Segoe UI';
  g.fillText('exterior time t = '+tc.toFixed(1)+' M',ox+16,H-42);
  if(mode==='u'){g.fillText('unstable mode: ω = i·'+md.gamma.toFixed(4)+'/M   growth time 1/γ = '+(1/md.gamma).toFixed(1)+' M',ox+16,H-22);}
  else{g.fillText('stable matter mode: ω ≈ '+md.wst.toFixed(4)+'/M   period '+(2*Math.PI/md.wst).toFixed(1)+' M',ox+16,H-22);}
  g.fillStyle=C.aqua;g.fillText('pressure pushes out',ox+420,70);g.fillStyle=C.orange;g.fillText('gravity pulls in',ox+420,90);
  g.fillStyle=C.muted;g.font='12px Segoe UI';g.fillText('dashed: unperturbed shell · amplitude exaggerated',ox+16,H-62);
}
function aLoop(ts){if(!A.play){A.last=null;return;}if(A.last===null)A.last=ts;const dt=(ts-A.last)/1000;A.last=ts;
  A.t+=dt*4*parseFloat(document.getElementById('aSpeed').value);aDraw();requestAnimationFrame(aLoop);}
document.getElementById('aPlay').onclick=()=>{A.play=!A.play;document.getElementById('aPlay').textContent=A.play?'❚❚ pause':'▶ play';if(A.play)requestAnimationFrame(aLoop);};
document.getElementById('aReset').onclick=()=>{A.t=0;aDraw();};
document.getElementById('aR').oninput=aInit;document.getElementById('aMode').onchange=()=>{A.t=0;aDraw();};

/* ------------------------------------------------------------------ slide 5: timeline */
function tDraw(){const cv=document.getElementById('cvT'),g=hidpi(cv),W=cv._w,H=cv._h;g.clearRect(0,0,W,H);
  const Rm=6,tr=wallTrajectory(Rm);const L=56,B=H-44,Tw=W-L-20,Th=B-24;
  let tmax=0;for(let i=0;i<tr.t.length;i++)if(isFinite(tr.t[i]))tmax=tr.t[i];const tshow=Math.min(tmax,40);
  g.strokeStyle=C.rule;g.lineWidth=1;g.beginPath();g.moveTo(L,24);g.lineTo(L,B);g.lineTo(L+Tw,B);g.stroke();
  g.fillStyle=C.ink2;g.font='13px Segoe UI';g.fillText('R/M',8,30);g.fillText('exterior time t / M',L+Tw-120,B+34);
  for(let v=0;v<=6;v+=2){const y=B-Th*v/Rm;g.fillText(v,L-18,y+4);}
  for(let v=0;v<=40;v+=10){const x=L+Tw*v/40;g.fillText(v,x-6,B+16);}
  g.setLineDash([4,4]);g.strokeStyle=C.muted;g.beginPath();g.moveTo(L,B-Th*2/Rm);g.lineTo(L+Tw,B-Th*2/Rm);g.stroke();g.setLineDash([]);
  g.fillStyle=C.muted;g.fillText('r = 2M (reached only as t → ∞)',L+Tw-205,B-Th*2/Rm-6);
  g.strokeStyle=C.blue;g.lineWidth=2.5;g.beginPath();let first=true;
  for(let i=0;i<tr.t.length;i+=5){if(!isFinite(tr.t[i])||tr.t[i]>40)break;const x=L+Tw*tr.t[i]/40,y=B-Th*tr.R[i]/Rm;first?g.moveTo(x,y):g.lineTo(x,y);first=false;}g.stroke();
  g.fillStyle=C.blue;g.font='600 13px Segoe UI';g.fillText('domain wall, R_m = 6M',L+10,40);
  // QNM periods of model A at R=6 (first-round catalogue, lowest modes): 0.18120-0.22393i, 0.66401-0.17021i
  const per=[[2*Math.PI/0.18120,'period of the lowest model-A QNM'],[2*Math.PI/0.66401,'period of the next QNM']];
  per.forEach((p,k)=>{const y=B-Th*(4.6-k*1.0)/Rm;g.strokeStyle=C.orange;g.lineWidth=5;g.beginPath();g.moveTo(L+4,y);g.lineTo(L+4+Tw*Math.min(p[0],40)/40,y);g.stroke();
    g.fillStyle=C.orange;g.font='12px Segoe UI';g.fillText(p[1]+': '+p[0].toFixed(1)+' M',L+Tw*0.30,y-7);});
  const tdyn=1/(Math.sqrt(1-2/6)*Math.sqrt((1+3*Math.sqrt(1-2/6))/(2*6)/6)),xd=L+Tw*tdyn/40;
  g.strokeStyle=C.ink2;g.lineWidth=1.5;g.beginPath();g.moveTo(xd,B);g.lineTo(xd,B-10);g.stroke();
  g.fillStyle=C.ink2;g.font='12px Segoe UI';g.fillText('1/ω_dyn = '+tdyn.toFixed(1)+' M',xd-30,B-14);
}

/* ------------------------------------------------------------------ slide 8: wave simulation */
function lambertW(z){let w=z<3?Math.log(1+z)*0.8:Math.log(z)-Math.log(Math.log(z));for(let i=0;i<60;i++){const ew=Math.exp(w),f=w*ew-z;const dw=f/(ew*(w+1)-(w+2)*f/(2*w+2));w-=dw;if(Math.abs(dw)<1e-13*(1+Math.abs(w)))break;}return w;}
const Wv={play:false};
function wInit(){
  const R=parseFloat(document.getElementById('wR').value),w0=parseFloat(document.getElementById('wW').value),l=2,lam=6;
  document.getElementById('wWv').textContent=w0.toFixed(2);
  const s=Math.sqrt(1-2/R),xR=R+2*Math.log(R/2-1),rmin=0.06*R,xL=xR-(R-rmin)/s,xMax=620,h=0.05;
  const N=Math.round((xMax-xL)/h)+1;const x=new Float64Array(N),V=new Float64Array(N),r=new Float64Array(N);
  let iR=0;for(let i=0;i<N;i++){x[i]=xL+i*h;if(x[i]<xR){r[i]=R+s*(x[i]-xR);V[i]=s*s*lam/(r[i]*r[i]);iR=i;}else{r[i]=2*(1+lambertW(Math.exp(x[i]/2-1)));const f=1-2/r[i];V[i]=f*(lam/(r[i]*r[i])-6/(r[i]**3));}}
  iR=iR+1; const Delta=s*(1-s)/R;
  const dt=0.5*h,x0=115,wid=Math.min(Math.max(1.2*2*Math.PI/w0,8),22);
  const F=xi=>Math.exp(-xi*xi/(2*wid*wid))*Math.cos(w0*xi);
  const u=new Float64Array(N),um=new Float64Array(N);for(let i=0;i<N;i++){u[i]=F(x[i]-x0);um[i]=F(x[i]-x0-dt);}
  u[0]=0;um[0]=0;
  Object.assign(Wv,{R,w0,s,xR,xL,xMax,h,N,x,V,r,iR,Delta,dt,u,um,un:null,k:0,t:0,obs:[],iObs:Math.round((75-xL)/h),delta:document.getElementById('wD').checked});
  wDraw();}
function wStep(n){const {N,h,dt,V,iR}=Wv;const c=dt*dt/(h*h),d2=dt*dt;const dlt=Wv.delta?Wv.Delta/h:0;
  if(!Wv.un||Wv.un.length!==N)Wv.un=new Float64Array(N);
  for(let k=0;k<n;k++){const u=Wv.u,um=Wv.um,un=Wv.un;
    for(let i=1;i<N-1;i++){const v=(i===iR)?V[i]+dlt:V[i];un[i]=2*u[i]-um[i]+c*(u[i+1]-2*u[i]+u[i-1])-d2*v*u[i];}
    un[0]=0;un[N-1]=u[N-1]-(dt/h)*(u[N-1]-u[N-2]);
    Wv.um=u;Wv.u=un;Wv.un=um;                     // rotate buffers (no allocation)
    Wv.t+=dt;Wv.k=(Wv.k||0)+1;if(Wv.k%4===0)Wv.obs.push([Wv.t,Wv.u[Wv.iObs]]);}}
function wDraw(){const cv=document.getElementById('cvW'),g=hidpi(cv),W=cv._w,H=cv._h;g.clearRect(0,0,W,H);
  const {x,V,u,N,xR,xL}=Wv;const X0=xL-2,X1=150,L=50,Rr=W-16,T=16,B=H-30;const X=v=>L+(Rr-L)*(v-X0)/(X1-X0);const Yu=v=>T+(B-T)*(0.5-0.42*v);
  let Vmax=0;for(let i=0;i<N;i++){if(x[i]>X1)break;if(x[i]>xR-0.2*(xR-xL)&&V[i]>Vmax)Vmax=V[i];}Vmax=Math.max(Vmax,Wv.w0*Wv.w0*1.2);
  g.fillStyle='rgba(42,120,214,0.10)';g.fillRect(X(xL),T,X(xR)-X(xL),B-T);
  g.fillStyle='rgba(138,137,132,0.22)';g.beginPath();g.moveTo(X(X0),B);for(let i=0;i<N;i++){if(x[i]>X1)break;g.lineTo(X(x[i]),B-(B-T)*0.9*Math.min(V[i]/Vmax,1.05));}g.lineTo(X(X1),B);g.closePath();g.fill();
  g.strokeStyle=C.muted;g.setLineDash([4,4]);g.beginPath();const yw=B-(B-T)*0.9*Wv.w0*Wv.w0/Vmax;g.moveTo(L,yw);g.lineTo(Rr,yw);g.stroke();g.setLineDash([]);
  g.fillStyle=C.muted;g.font='12px Segoe UI';g.fillText('ω₀² (packet energy)',Rr-120,yw-4);
  g.strokeStyle=C.ink;g.lineWidth=3;g.beginPath();g.moveTo(X(xR),T);g.lineTo(X(xR),B);g.stroke();
  g.strokeStyle=C.orange;g.lineWidth=1.6;g.beginPath();let st=false;for(let i=0;i<N;i+=2){if(x[i]>X1)break;const y=Yu(u[i]);st?g.lineTo(X(x[i]),y):g.moveTo(X(x[i]),y);st=true;}g.stroke();
  g.strokeStyle=C.rule;g.lineWidth=1;g.beginPath();g.moveTo(L,B);g.lineTo(Rr,B);g.stroke();
  g.fillStyle=C.ink2;g.font='13px Segoe UI';for(let v=-20;v<=150;v+=10){if(v<X0)continue;g.fillText(v,X(v)-8,B+16);}
  g.fillText('x / M  (tortoise coordinate)',Rr-190,B+28);
  g.fillStyle=C.blue;g.fillText('flat interior',X(xL)+4,T+16);g.fillStyle=C.ink;g.fillText('shell',X(xR)+6,T+32);
  g.fillStyle=C.ink2;g.font='600 14px Segoe UI';g.fillText('t = '+Wv.t.toFixed(0)+' M',Rr-90,T+18);g.font='13px Segoe UI';
  g.strokeStyle=C.aqua;g.lineWidth=1;g.setLineDash([2,3]);g.beginPath();g.moveTo(X(75),T);g.lineTo(X(75),B);g.stroke();g.setLineDash([]);
  g.fillStyle=C.aqua;g.fillText('detector',X(75)+4,T+48);
  // time series
  const cs=document.getElementById('cvS'),q=hidpi(cs),W2=cs._w,H2=cs._h;q.clearRect(0,0,W2,H2);
  const L2=50,R2=W2-16,T2=12,B2=H2-26,tMax=800;const Xt=v=>L2+(R2-L2)*v/tMax,Yl=v=>T2+(B2-T2)*(0-v)/8;
  q.strokeStyle=C.rule;q.beginPath();q.moveTo(L2,T2);q.lineTo(L2,B2);q.lineTo(R2,B2);q.stroke();
  q.fillStyle=C.ink2;q.font='12px Segoe UI';for(let e=0;e>=-8;e-=2)q.fillText('1e'+e,8,Yl(e)+4);for(let v=0;v<=tMax;v+=100)q.fillText(v,Xt(v)-8,B2+14);
  q.fillText('t / M at the detector',R2-130,B2+24);
  q.strokeStyle=C.aqua;q.lineWidth=1.3;q.beginPath();Wv.obs.forEach((p,k)=>{const y=Yl(Math.max(Math.log10(Math.abs(p[1])+1e-12),-8));k?q.lineTo(Xt(p[0]),y):q.moveTo(Xt(p[0]),y);});q.stroke();
}
function wLoop(){if(!Wv.play)return;const n=parseInt(document.getElementById('wSp').value)*8;wStep(n);wDraw();if(Wv.t<800)requestAnimationFrame(wLoop);else{Wv.play=false;document.getElementById('wPlay').textContent='▶ play';}}
document.getElementById('wPlay').onclick=()=>{Wv.play=!Wv.play;document.getElementById('wPlay').textContent=Wv.play?'❚❚ pause':'▶ play';if(Wv.play)requestAnimationFrame(wLoop);};
document.getElementById('wReset').onclick=()=>{Wv.play=false;document.getElementById('wPlay').textContent='▶ play';wInit();};
['wR','wW','wD'].forEach(id=>document.getElementById(id).onchange=()=>{Wv.play=false;document.getElementById('wPlay').textContent='▶ play';wInit();});
document.getElementById('wW').oninput=()=>{document.getElementById('wWv').textContent=parseFloat(document.getElementById('wW').value).toFixed(2);};

/* ------------------------------------------------------------------ init */
let inited={};
function onSlide(i){const t=slides[i].dataset.title;
  if(t==='The collapse problem'&&!inited.a){aInit();inited.a=1;}
  if(t==='Four options'&&!inited.t){tDraw();inited.t=1;}
  if(t==='Live scattering'&&!inited.w){wInit();inited.w=1;}
  if(/autoplay/.test(location.search)){                      // optional: start the simulations automatically
    if(t==='The collapse problem'&&!A.play)document.getElementById('aPlay').click();
    if(t==='Live scattering'&&!Wv.play)document.getElementById('wPlay').click();}}
const h0=parseInt((location.hash||'#1').slice(1));show(isFinite(h0)?h0-1:0);
