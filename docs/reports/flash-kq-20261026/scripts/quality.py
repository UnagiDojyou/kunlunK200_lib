import pathlib,array,math
def quality(ref,out):
 a=array.array('f');a.frombytes(pathlib.Path(ref).read_bytes());b=array.array('f');b.frombytes(pathlib.Path(out).read_bytes());assert len(a)==len(b) and len(a)%32000==0
 sq=en=0.;kl=[];greedy=0
 for i in range(0,len(a),32000):
  x=a[i:i+32000];y=b[i:i+32000];assert all(map(math.isfinite,y));xm=max(x);ym=max(y);sx=sum(math.exp(v-xm) for v in x);sy=sum(math.exp(v-ym) for v in y);lx=xm+math.log(sx);ly=ym+math.log(sy)
  kl.append(sum(math.exp(v-lx)*(v-lx-w+ly) for v,w in zip(x,y)));greedy+=x.index(xm)==y.index(ym)
  for v,w in zip(x,y):sq+=(v-w)**2;en+=v*v
 return {'nmse':sq/max(en,1e-30),'mean_kl':sum(kl)/len(kl),'max_kl':max(kl),'greedy_equal':greedy,'positions':len(kl)}