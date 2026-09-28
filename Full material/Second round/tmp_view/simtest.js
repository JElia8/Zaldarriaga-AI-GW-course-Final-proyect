// Node test of the presentation's wave simulation (same discretisation as wInit/wStep).
function lambertW(z){let w=z<3?Math.log(1+z)*0.8:Math.log(z)-Math.log(Math.log(z));for(let i=0;i<60;i++){const ew=Math.exp(w),f=w*ew-z;const dw=f/(ew*(w+1)-(w+2)*f/(2*w+2));w-=dw;if(Math.abs(dw)<1e-13*(1+Math.abs(w)))break;}return w;}
function run(R,w0,delta){
  const lam=6,s=Math.sqrt(1-2/R),xR=R+2*Math.log(R/2-1),rmin=0.06*R,xL=xR-(R-rmin)/s,xMax=620,h=0.05;
  const N=Math.round((xMax-xL)/h)+1;const x=new Float64Array(N),V=new Float64Array(N);let iR=0;
  for(let i=0;i<N;i++){x[i]=xL+i*h;let r;if(x[i]<xR){r=R+s*(x[i]-xR);V[i]=s*s*lam/(r*r);iR=i;}else{r=2*(1+lambertW(Math.exp(x[i]/2-1)));const f=1-2/r;V[i]=f*(lam/(r*r)-6/(r**3));}}
  iR+=1;const Delta=s*(1-s)/R,dt=0.5*h,x0=115,wid=Math.min(Math.max(1.2*2*Math.PI/w0,8),22);
  const F=xi=>Math.exp(-xi*xi/(2*wid*wid))*Math.cos(w0*xi);
  let u=new Float64Array(N),um=new Float64Array(N),un=new Float64Array(N);for(let i=0;i<N;i++){u[i]=F(x[i]-x0);um[i]=F(x[i]-x0-dt);}u[0]=0;um[0]=0;
  const c=dt*dt/(h*h),d2=dt*dt,dlt=delta?Delta/h:0,iObs=Math.round((75-xL)/h);let t=0;const rec=[];
  // check the r(x) inversion
  let maxErr=0;for(let i=iR;i<N;i+=500){const r=2*(1+lambertW(Math.exp(x[i]/2-1)));maxErr=Math.max(maxErr,Math.abs(r+2*Math.log(r/2-1)-x[i]));}
  while(t<800){for(let i=1;i<N-1;i++){const v=(i===iR)?V[i]+dlt:V[i];un[i]=2*u[i]-um[i]+c*(u[i+1]-2*u[i]+u[i-1])-d2*v*u[i];}
    un[0]=0;un[N-1]=u[N-1]-(dt/h)*(u[N-1]-u[N-2]);const tmp=um;um=u;u=un;un=tmp;t+=dt;rec.push([t,u[iObs]]);}
  // envelope in windows
  const win=50,out=[];for(let a=0;a<800;a+=win){let m=0;for(const [tt,v] of rec)if(tt>=a&&tt<a+win)m=Math.max(m,Math.abs(v));out.push(m.toExponential(1));}
  let maxAll=0;for(let i=0;i<N;i++)maxAll=Math.max(maxAll,Math.abs(u[i]));
  console.log(`R=${R} w0=${w0} delta=${delta}: tortoise inversion err ${maxErr.toExponential(1)}; |u|max at t=800: ${maxAll.toExponential(1)}`);
  console.log('   detector envelope per 50M window:',out.join(' '));
}
run(3,0.45,true);run(3,0.45,false);run(2.2,0.35,true);run(6,0.3,true);run(3,1.0,true);
