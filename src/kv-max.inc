// Exact maxima of tracked physical F16 rows, not logical/active KV cells.
extern "C" int k200_kv_max_diag_enabled() __attribute__((weak));
extern "C" void k200_kv_max_diag(const char *,int,uint64_t,uint64_t,uint64_t) __attribute__((weak));
namespace {
uint64_t kv_diag_clock() {
 return std::chrono::duration_cast<std::chrono::nanoseconds>(
     std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct KVDiag {
 const char *name;
 int id;
 uint64_t bytes;
 uint64_t start=k200_kv_max_diag_enabled&&k200_kv_max_diag&&k200_kv_max_diag_enabled()?kv_diag_clock():0;
 void result(const char *tag){name=tag;}
 ~KVDiag(){if(start)k200_kv_max_diag(name,id,start,kv_diag_clock(),bytes);}
};
bool kv_opt_in(const char *name){
 const char *flag=std::getenv(name);
 return !flag || std::strcmp(flag,"1") == 0;
}
}
namespace {
struct KVMaxEngine {
 int id;Kernel::Impl writer,prefill_writer,build,reduce;std::unique_ptr<Buffer> descriptor,jobs,reference;xdnn::Context *vendor=nullptr;
 int writer_config[24]={};
 bool writer_config_ready=false;
 explicit KVMaxEngine(int d):id(d){}
 ~KVMaxEngine(){std::lock_guard<std::recursive_mutex>lock(mutex);select(id);check(xpu_wait());if(vendor)xdnn::destroy_context(vendor);for(auto *f:{&writer,&prefill_writer,&build,&reduce})if(f->code)check(xpu_free(f->code));}
};
std::weak_ptr<KVMaxEngine> kv_engines[2];KVMaxStatistics kv_stats[2];uint64_t kv_generation=0;
bool kv_rows_notice_suppressed=false;
int kv_mode(){const char *p=std::getenv("KUNLUN_FLASH_KV_MAX");return !p||!std::strcmp(p,"auto")?1:!std::strcmp(p,"verify")?2:0;}
int kv_min_s(int features){const char *p=std::getenv("KUNLUN_FLASH_KV_MAX_MIN_S");if(p){int n=std::atoi(p);if(n==256||n==512||n==1024||n==2048||n==4096||n==8192||n==16384)return n;}return features>=1024?512:features>=256?8192:INT_MAX;}
}
struct KVMaxCache::Impl {
 int id,features,rows,last_s=0;void *data;size_t bytes;std::string name;uint64_t generation,revision=0;bool active=true,poison=false;
 bool ever_seeded=false;
 bool extent_invalidated=false;
 std::unique_ptr<Buffer> maxima,partials,global;int pending=-1,cached_start=0,cached_count=0;bool global_valid=false,seed_requested=false;std::shared_ptr<KVMaxEngine> engine;
 std::vector<unsigned char> zero,finite;bool all_finite=false;
 std::vector<std::pair<int,int>> dirty;
 struct Event{std::string kind;int lo,hi;uint64_t revision;};std::deque<Event> history;
 Impl(int d,void*p,size_t n,int f,int r,const char*label):id(d),features(f),rows(r),data(p),bytes(n),name(label?label:"unnamed"),generation(++kv_generation),zero(r),finite(r){dirty.push_back({0,rows});}
 void event(const char *kind,int lo,int hi){++revision;history.push_back({kind,lo,hi,revision});if(history.size()>16)history.pop_front();}
 void add_dirty(int lo,int hi){if(lo>=hi)return;extent_invalidated=extent_invalidated||global_valid;global_valid=false;all_finite=false;std::fill(zero.begin()+lo,zero.begin()+hi,0);std::fill(finite.begin()+lo,finite.begin()+hi,0);std::vector<std::pair<int,int>>out;for(auto r:dirty){if(r.second<lo)out.push_back(r);else if(hi<r.first){out.push_back({lo,hi});lo=r.first;hi=r.second;}else{lo=std::min(lo,r.first);hi=std::max(hi,r.second);}}out.push_back({lo,hi});dirty.swap(out);}
 bool rows_known(int lo,int hi) const {
  return std::none_of(dirty.begin(),dirty.end(),[&](auto range){
   return range.first<hi&&range.second>lo;
  });
 }
 void clean(int lo,int hi){std::vector<std::pair<int,int>>out;for(auto r:dirty){if(r.second<=lo||r.first>=hi)out.push_back(r);else{if(r.first<lo)out.push_back({r.first,lo});if(r.second>hi)out.push_back({hi,r.second});}}dirty.swap(out);}
 void ensure_engine(){if(!engine){engine=kv_engines[id].lock();if(!engine){engine=std::make_shared<KVMaxEngine>(id);kv_engines[id]=engine;}}}
 void ensure(){ensure_engine();if(!maxima){maxima.reset(new Buffer(id,(size_t)rows*4));global.reset(new Buffer(id,256));std::vector<unsigned int>zero(rows);select(id);check(xpu_memcpy(maxima->data(),zero.data(),rows*4,XPU_HOST_TO_DEVICE));check(xpu_wait());kv_stats[id].metadata_bytes+=rows*4+256;}}
};
namespace {
using KVKey=std::pair<int,uintptr_t>;
std::map<KVKey,KVMaxCache::Impl*> &kv_registry(){static auto *p=new std::map<KVKey,KVMaxCache::Impl*>;return *p;}
bool kv_tracking_any(){return !kv_registry().empty();}
bool kv_tracked_rows(){return kv_rows_notice_suppressed;}
KVMaxCache::Impl *kv_find(int id,const void *data,size_t span){auto &m=kv_registry();uintptr_t p=(uintptr_t)data;auto it=m.upper_bound({id,p});if(it==m.begin())return nullptr;--it;auto *r=it->second;if(r->id!=id||!r->active||p<(uintptr_t)r->data||span>r->bytes||p-(uintptr_t)r->data>r->bytes-span)return nullptr;return r;}
void kv_retire(int id,void *ptr,size_t bytes){auto &m=kv_registry();uintptr_t p=(uintptr_t)ptr;for(auto it=m.begin();it!=m.end();){auto *r=it->second;if(r->id==id&&p<(uintptr_t)r->data+r->bytes&&(uintptr_t)r->data<p+bytes){r->active=false;++kv_stats[id].retired;it=m.erase(it);}else ++it;}}
void kv_copy_notice(void *dst,size_t n){if(!n||!kv_tracking_any())return;int id=-1;check(xpu_current_device(&id));if(id>=0&&id<2)kv_max_invalidate(id,dst,n);}
}
KVMaxCache::KVMaxCache(int id,void *data,size_t bytes,int features,int rows,const char *name):impl_(nullptr){
 std::lock_guard<std::recursive_mutex>lock(mutex);require(id>=0&&id<2&&data&&features>=64&&features%64==0&&features<=16384&&rows>0&&rows<=65536&&bytes==(size_t)features*rows*2);
 impl_=new Impl(id,data,bytes,features,rows,name);auto &m=kv_registry();bool overlap=false;
 for(auto it=m.begin();it!=m.end();){auto *r=it->second;if(r->id==id&&(uintptr_t)data<(uintptr_t)r->data+r->bytes&&(uintptr_t)r->data<(uintptr_t)data+bytes){r->active=false;overlap=true;it=m.erase(it);}else ++it;}
 if(overlap)impl_->active=false;else m[{id,(uintptr_t)data}]=impl_;
}
KVMaxCache::~KVMaxCache(){std::lock_guard<std::recursive_mutex>lock(mutex);auto &m=kv_registry();auto it=m.find({impl_->id,(uintptr_t)impl_->data});if(it!=m.end()&&it->second==impl_)m.erase(it);delete impl_;}
void kv_max_invalidate(int id,void *ptr,size_t n){
 std::lock_guard<std::recursive_mutex>lock(mutex);if(!n)return;uintptr_t p=(uintptr_t)ptr;
 auto &m=kv_registry();auto it=m.lower_bound({id,p});if(it!=m.begin()){auto prev=it;--prev;if(prev->second->id==id&&(uintptr_t)prev->second->data+prev->second->bytes>p)it=prev;}
 for(;it!=m.end()&&it->first.first==id&&it->first.second<p+n;++it){auto *r=it->second;uintptr_t q=(uintptr_t)r->data;if(p>=q+r->bytes)continue;size_t lo=std::max(p,q)-q,hi=std::min(p+n,q+r->bytes)-q;int rowbytes=r->features*2;int a=lo/rowbytes,b=(hi+rowbytes-1)/rowbytes;if(r->pending>=a&&r->pending<b)r->pending=-1;r->add_dirty(a,b);r->event("write",a,b);++kv_stats[id].invalidations;}
}
void kv_max_zero(int id,void *ptr,size_t n){
 std::lock_guard<std::recursive_mutex>lock(mutex);if(!n)return;kv_max_invalidate(id,ptr,n);uintptr_t p=(uintptr_t)ptr;size_t largest=0;
 for(auto it:kv_registry()){auto *r=it.second;uintptr_t q=(uintptr_t)r->data;if(r->id==id&&p<=q&&n>=r->bytes&&q-p<=n-r->bytes){r->dirty.clear();r->poison=false;r->pending=-1;r->global_valid=false;r->seed_requested=false;r->ever_seeded=false;r->extent_invalidated=false;r->all_finite=true;std::fill(r->zero.begin(),r->zero.end(),1);std::fill(r->finite.begin(),r->finite.end(),1);r->event("zero",0,r->rows);if(r->maxima)largest=std::max(largest,(size_t)r->rows);}}
 if(largest){std::vector<unsigned int>zero(largest);select(id);for(auto it:kv_registry()){auto *r=it.second;uintptr_t q=(uintptr_t)r->data;if(r->id==id&&r->maxima&&p<=q&&n>=r->bytes&&q-p<=n-r->bytes){check(xpu_memcpy(r->maxima->data(),zero.data(),r->rows*4,XPU_HOST_TO_DEVICE));kv_stats[id].zero_bytes+=r->rows*4;}}check(xpu_wait());}
}
void kv_max_mark_rows(int id,void *data,const int64_t *rows,size_t count){
 std::lock_guard<std::recursive_mutex>lock(mutex);auto *r=kv_find(id,data,0);if(!r||r->data!=data)return;
 for(size_t j=0;j<count;++j){require(rows[j]>=0&&rows[j]<r->rows);if(r->pending==rows[j])r->pending=-1;r->add_dirty(rows[j],rows[j]+1);r->event("row-write",rows[j],rows[j]+1);}++kv_stats[id].invalidations;
}
static bool kv_max_set_rows_impl(Kernel &kernel,int id,void *dst,size_t db,const void *src,size_t sb,const void *indices,size_t ib,
                                 const SetRowsLayout &p,const int64_t *targets,size_t count,KVRowsPending *pending){
 std::lock_guard<std::recursive_mutex>lock(mutex);auto *r=kv_find(id,dst,db);
 if(!r||r->data!=dst||db!=r->bytes||p.cache_features||p.dim2!=1||p.dim3!=1||p.width!=r->features||p.destination_rows!=r->rows||p.destination_stride[0]!=r->features*2||p.source_stride[0]!=p.source_bytes||p.index_stride[0]!=p.index_bytes||p.index_extent[0]!=1||p.index_extent[1]!=1||count!=(size_t)p.rows||(p.rows>256&&(!large_pp_enabled()||p.rows>1024)))return false;
 uintptr_t da=(uintptr_t)dst,sa=(uintptr_t)src,ia=(uintptr_t)indices;
 if(da+db<da||sa+sb<sa||ia+ib<ia||(da<sa+sb&&sa<da+db)||(da<ia+ib&&ia<da+db))return false;
 require(src&&indices&&targets&&count>0&&(uintptr_t)dst%2==0&&(p.source_bytes==2||p.source_bytes==4)&&(p.index_bytes==4||p.index_bytes==8));
 require((uintptr_t)src%p.source_bytes==0&&(uintptr_t)indices%p.index_bytes==0);
 if(!pending){
  for(size_t i=0;i<count;++i)require(targets[i]>=0&&targets[i]<r->rows);
  if(!std::is_sorted(targets,targets+count)){std::vector<int64_t>sorted(targets,targets+count);std::sort(sorted.begin(),sorted.end());if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())return false;}
  else if(std::adjacent_find(targets,targets+count)!=targets+count)return false;
 }
 if(p.rows>256){
  require((size_t)(p.rows-1)*p.source_stride[1]+(size_t)p.width*p.source_bytes<=sb&&count*p.index_bytes<=ib);
  // Keep the 512-int local index array bounded, including I64 indices. The
  // outer mutex hides partial metadata updates; avoid TG semantics on a tail.
  // Targets are a set, so only the completed transaction is made visible.
  for(int first=0;first<p.rows;){
   auto part=p;part.rows=std::min(256,p.rows-first);if(p.rows-first==257)--part.rows;
   size_t so=(size_t)first*p.source_stride[1],io=(size_t)first*p.index_bytes;
   require(kv_max_set_rows(kernel,id,dst,db,(const char*)src+so,sb-so,(const char*)indices+io,ib-io,part,targets+first,part.rows));
   first+=part.rows;
  }
  return true;
 }
 const char *wm=std::getenv("KUNLUN_FLASH_KV_MAX_WRITER");const char *prefill=std::getenv("KUNLUN_FLASH_KV_MAX_PREFILL");
 const bool tp_rows=pending&&kv_mode()==1&&kv_opt_in("GGML_KUNLUN_TP_KV_MAX_ROWS")&&r->features==2048;
 bool pp_max=(prefill&&!std::strcmp(prefill,"fused"))||tp_rows;
 const bool fast_prefill=tp_rows&&p.rows==128&&p.source_bytes==4&&
     p.source_stride[1]==8192&&((da|sa)&31)==0&&
     kv_opt_in("GGML_KUNLUN_TP_KV_MAX_PP_WRITER");
 bool fused=kv_mode()==2||(kv_mode()==1&&r->rows>=kv_min_s(r->features));
 bool tg_max=p.rows==1&&(tp_rows||kv_mode()==2||(r->last_s>=kv_min_s(r->features)&&(r->all_finite||r->global_valid)));
 bool update_max=tg_max||(p.rows>1&&pp_max);
 if(wm&&!std::strcmp(wm,"lazy"))fused=false;
 if(!fused){kv_max_mark_rows(id,dst,targets,count);struct Guard{bool saved;Guard():saved(kv_rows_notice_suppressed){kv_rows_notice_suppressed=true;}~Guard(){kv_rows_notice_suppressed=saved;}}guard;require(set_rows_f16_impl(kernel,id,dst,db,src,sb,indices,ib,p,pending));return true;}
 // Same source/indices/layout contract as the original writer, with unique targets
 // validated by the backend. Maximum is computed from the stored rounded F16.
 require(p.source_bytes==2||p.source_bytes==4);require(p.index_bytes==4||p.index_bytes==8);
 require((size_t)(p.rows-1)*p.source_stride[1]+(size_t)p.width*p.source_bytes<=sb&&count*p.index_bytes<=ib);
 bool all_finite=r->all_finite;select(id);if(!update_max)r->ensure_engine();else r->ensure();if(r->pending>=0){r->add_dirty(r->pending,r->pending+1);r->pending=-1;}if(tg_max){if(!r->partials)kv_stats[id].metadata_bytes+=256;native_reserve(r->partials,id,256);}auto &e=*r->engine;
 auto &writer=fast_prefill?e.prefill_writer:e.writer;
 load_kernel(writer,fast_prefill?"kunlun-kv-max-prefill-writer":"kunlun-kv-max-writer",id);
 native_reserve(e.descriptor,id,512);
 // The reducer can lower the cached peak by rescanning row maxima if needed.
 const bool retain_overwrite=tp_rows&&tg_max&&r->global_valid&&
     kv_opt_in("GGML_KUNLUN_TP_KV_MAX_OVERWRITE")&&
     r->rows_known(r->cached_start,r->cached_start+r->cached_count)&&
     r->rows_known(targets[0],targets[0]+1);
 if(tg_max&&!r->zero[targets[0]]&&!retain_overwrite){
  r->extent_invalidated=r->extent_invalidated||r->global_valid;
  r->global_valid=false;
 }
 for(size_t i=0;i<count;++i)r->zero[targets[i]]=0;
 int cfg[128]={};std::memcpy(cfg,&p,sizeof(p));cfg[23]=p.rows==1?(tg_max?0:-1):(pp_max?1:0);
 const bool reuse=pending&&kv_mode()==1&&r->features==2048&&
     kv_opt_in("GGML_KUNLUN_TP_KV_MAX_WRITER_REUSE");
 if(!reuse||!e.writer_config_ready||std::memcmp(e.writer_config,cfg,sizeof(e.writer_config))!=0){
  KVDiag diag{"writer-config-upload",id,sizeof(cfg)};
  check(xpu_memcpy(e.descriptor->data(),cfg,sizeof(cfg),XPU_HOST_TO_DEVICE));check(xpu_wait());
  if(reuse)std::memcpy(e.writer_config,cfg,sizeof(e.writer_config));
 }else{
  KVDiag diag{"writer-config-reuse",id,0};
 }
 // Both status flags are sticky within one command. Reuse is legal only after
 // both writers completed and returned zero, with no intervening descriptor use.
 e.writer_config_ready=false;
 void *desc=e.descriptor->data(),*status=(char*)desc+256,*maximum=r->maxima?r->maxima->data():desc;
 void *partial=r->partials?r->partials->data():desc;
 auto commit=[r,targets,count,update_max,all_finite,tg_max,id,&e,reuse](bool poison){
  e.writer_config_ready=reuse&&!poison;
  if(poison)r->poison=true;
  for(size_t i=0;i<count;++i){if(update_max)r->clean(targets[i],targets[i]+1);else r->add_dirty(targets[i],targets[i]+1);r->finite[targets[i]]=1;r->event("fused-row",targets[i],targets[i]+1);}
  if(all_finite&&!poison)r->all_finite=true;
  if(tg_max)r->pending=targets[0];else{
   r->extent_invalidated=r->extent_invalidated||r->global_valid;
   r->global_valid=false;
  }
  ++kv_stats[id].writer_calls;kv_stats[id].writer_rows+=count;
 };
 const int cores=fast_prefill?16:8;
 const int launch_error=native_launch_status(writer,{native_arg(dst),native_arg(src),native_arg(indices),native_arg(desc),native_arg(status),native_arg(maximum),native_arg(partial)},cores);
 if(pending){pending->status=status;pending->status_bytes=8;pending->launch_error=launch_error;pending->commit=commit;return true;}
 check(launch_error);check(xpu_wait());
 int flags[2]={};check(xpu_memcpy(flags,status,8,XPU_DEVICE_TO_HOST));check(xpu_wait());require(flags[0]==0);
 commit(flags[1]!=0);return true;
}

bool kv_max_set_rows(Kernel &kernel,int id,void *dst,size_t db,const void *src,size_t sb,const void *indices,size_t ib,
                     const SetRowsLayout &p,const int64_t *targets,size_t count){
 return kv_max_set_rows_impl(kernel,id,dst,db,src,sb,indices,ib,p,targets,count,nullptr);
}

bool kv_set_rows_pair(const KVRowUpdate (&jobs)[2],const int64_t *targets,size_t count){
 std::lock_guard<std::recursive_mutex>lock(mutex);
 if(!targets || !count || count>128 || jobs[0].kernel==jobs[1].kernel ||
    jobs[0].layout.destination_rows!=jobs[1].layout.destination_rows ||
    !std::is_sorted(targets,targets+count) || std::adjacent_find(targets,targets+count)!=targets+count ||
    targets[0]<0 || targets[count-1]>=jobs[0].layout.destination_rows)return false;
 for(const auto & j:jobs)if(!j.kernel || !j.dst || !j.src || !j.indices ||
     j.layout.rows!=(int)count || j.layout.dim2!=1 || j.layout.dim3!=1 || j.layout.cache_features ||
     j.layout.index_extent[0]!=1 || j.layout.index_extent[1]!=1)return false;
 KVRowsPending pending[2];
 bool ok=true;
 for(int d=0;d<2;++d){
  const auto & j=jobs[d];
  if(!kv_max_set_rows_impl(*j.kernel,d,j.dst,j.dst_bytes,j.src,j.src_bytes,j.indices,j.index_bytes,j.layout,targets,count,&pending[d]))
   set_rows_f16_impl(*j.kernel,d,j.dst,j.dst_bytes,j.src,j.src_bytes,j.indices,j.index_bytes,j.layout,&pending[d]);
  if(pending[d].launch_error){ok=false;break;}
 }
 int flags[2][2]={};
 bool needs_drain[2]={};
 for(int d=0;d<2;++d)if(pending[d].status){
  select(d);
  int error=xpu_wait();
  if(!error && !pending[d].launch_error){
   const int copied=xpu_memcpy(flags[d],pending[d].status,pending[d].status_bytes,XPU_DEVICE_TO_HOST);
   error=xpu_wait();
   ok=ok && !copied && flags[d][0]==0;
  }
  needs_drain[d]=error!=0;
  ok=ok && !error;
 }
 for(int d=0;d<2;++d)if(needs_drain[d]){select(d);check(xpu_wait());}
 for(int d=0;d<2;++d){
  if(ok){if(pending[d].commit)pending[d].commit(flags[d][1]!=0);}
  else kv_max_invalidate(d,jobs[d].dst,jobs[d].dst_bytes);
 }
 return ok;
}
namespace {
bool kv_view(KVMaxCache::Impl *r,const void *data,const Tensor4Layout &p,int &start){
 if(!r||p.type!=InputType::f16||p.ne[3]!=1||p.ne[0]*p.ne[2]!=r->features||tensor4_span(p)!=tensor4_count(p)*2)return false;
 size_t off=(uintptr_t)data-(uintptr_t)r->data;size_t rowbytes=r->features*2;if(off%rowbytes)return false;start=off/rowbytes;return start+p.ne[1]<=r->rows;
}
bool kv_max_prepare_rows(int id,const void *k,const Tensor4Layout &pk,const void *v,const Tensor4Layout &pv,int queries,void *peaks,xdnn::Context *vendor){
 auto &s=kv_stats[id];++s.lookups;auto *kr=kv_find(id,k,tensor4_span(pk)),*vr=kv_find(id,v,tensor4_span(pv));int ks=0,vs=0;
 bool owned=kv_view(kr,k,pk,ks)&&kv_view(vr,v,pv,vs)&&kr!=vr;
 if(owned){kr->last_s=pk.ne[1];vr->last_s=pv.ne[1];}
 auto fallback=[&](){s.full_calls+=2;s.scan_bytes+=(tensor4_count(pk)+tensor4_count(pv))*2;return false;};
 if(!kv_mode()){++s.fallback_legacy;return fallback();}if(!owned){++s.fallback_unowned;return fallback();}
 if(kv_mode()!=2&&queries!=1){++s.fallback_pp;return fallback();}if(kv_mode()!=2&&pk.ne[1]<std::max(kv_min_s(kr->features),kv_min_s(vr->features))){++s.fallback_short;return fallback();}
 if(kr->poison||vr->poison){++s.fallback_poison;return fallback();}
 kr->ensure();vr->ensure();auto &e=*kr->engine;native_reserve(e.descriptor,id,512);
 std::vector<int>jobs;std::vector<std::pair<int,int>>clean[2];
 for(int w=0;w<2;++w){auto *r=w?vr:kr;int start=w?vs:ks;for(auto range:r->dirty){int lo=std::max(start,range.first),hi=std::min(start+pk.ne[1],range.second);if(lo>=hi)continue;clean[w].push_back({lo,hi});for(int row=lo;row<hi;++row)jobs.push_back((w<<30)|row);s.scan_bytes+=(size_t)(hi-lo)*r->features*2;}}
 if(!jobs.empty()){
  load_kernel(e.build,"kunlun-kv-max-build",id);native_reserve(e.jobs,id,jobs.size()*4);int zero[8]={};void *flag=(char*)e.descriptor->data()+256;
  e.writer_config_ready=false;
  check(xpu_memcpy(e.jobs->data(),jobs.data(),jobs.size()*4,XPU_HOST_TO_DEVICE));check(xpu_memcpy(flag,zero,sizeof(zero),XPU_HOST_TO_DEVICE));check(xpu_wait());
  const void *kd=kr->data,*vd=vr->data;void *km=kr->maxima->data(),*vm=vr->maxima->data(),*jp=e.jobs->data();int count=jobs.size();
  native_launch(e.build,{native_arg(kd),native_arg(vd),native_arg(km),native_arg(vm),native_arg(jp),native_arg(count),native_arg(kr->features),native_arg(vr->features),native_arg(flag)},8);check(xpu_wait());
  int bad=0;check(xpu_memcpy(&bad,flag,4,XPU_DEVICE_TO_HOST));check(xpu_wait());++s.rebuilds;s.rebuild_rows+=jobs.size();
  for(int w=0;w<2;++w)for(auto range:clean[w]){auto *r=w?vr:kr;r->clean(range.first,range.second);std::fill(r->finite.begin()+range.first,r->finite.begin()+range.second,1);}
  if(bad){kr->poison=vr->poison=true;++s.fallback_poison;return fallback();}
 }
 load_kernel(e.reduce,"kunlun-kv-max-reduce",id);const void *km=kr->maxima->data(),*vm=vr->maxima->data();int count=pk.ne[1];
 const void *kp=kr->partials?kr->partials->data():km,*vp=vr->partials?vr->partials->data():vm;int krow=kr->pending,vrow=vr->pending;
 int kold=kr->global_valid&&kr->cached_start==ks&&kr->cached_count<=count?kr->cached_count:-1,vold=vr->global_valid&&vr->cached_start==vs&&vr->cached_count<=count?vr->cached_count:-1;
 const char *rm=std::getenv("KUNLUN_FLASH_KV_MAX_REDUCE");if(rm&&!std::strcmp(rm,"linear"))kold=vold=-1;
 void *kg=kr->global->data(),*vg=vr->global->data();
 native_launch(e.reduce,{native_arg(peaks),native_arg(km),native_arg(vm),native_arg(ks),native_arg(vs),native_arg(count),native_arg(kp),native_arg(vp),native_arg(krow),native_arg(vrow),native_arg(kg),native_arg(vg),native_arg(kold),native_arg(vold)},8);
 kr->pending=vr->pending=-1;kr->global_valid=vr->global_valid=true;kr->cached_start=ks;vr->cached_start=vs;kr->cached_count=vr->cached_count=count;s.meta_read_bytes+=(size_t)(2*count-std::max(kold,0)-std::max(vold,0))*4+(krow>=0?132:0)+(vrow>=0?132:0)+8;++s.hits;
 if(kv_mode()==2){
  native_reserve(e.reference,id,1024);void *reference=e.reference->data(),*vref=(char*)reference+256;
  check(xdnn::findmax<float16>(vendor,(const float16*)k,(float*)reference,(int)tensor4_count(pk)));check(xdnn::findmax<float16>(vendor,(const float16*)v,(float*)vref,(int)tensor4_count(pv)));check(xpu_wait());
  float actual[256]={},expected[256]={};check(xpu_memcpy(actual,peaks,1024,XPU_DEVICE_TO_HOST));check(xpu_memcpy(expected,reference,1024,XPU_DEVICE_TO_HOST));check(xpu_wait());s.verify_calls+=2;
  for(int w=0;w<2;++w){float a=0,b=0;for(int j=0;j<4;++j){a=std::max(a,actual[w*64+j]);b=std::max(b,expected[w*64+j]);}if(std::memcmp(&a,&b,4)){
   ++s.verify_mismatches;auto *r=w?vr:kr;std::fprintf(stderr,"KV_MAX_MISMATCH pd=%d cache=%s generation=%llu revision=%llu start=%d S=%d cached=%a SDK=%a partials=%a,%a,%a,%a\n",id,r->name.c_str(),(unsigned long long)r->generation,(unsigned long long)r->revision,w?vs:ks,count,a,b,expected[w*64],expected[w*64+1],expected[w*64+2],expected[w*64+3]);for(auto ev:r->history)std::fprintf(stderr,"KV_MAX_HISTORY revision=%llu kind=%s lo=%d hi=%d\n",(unsigned long long)ev.revision,ev.kind.c_str(),ev.lo,ev.hi);std::fflush(stderr);require(false);
  }}
 }
 return true;
}
// Default sparse path: SDK seeds the exact range maximum once. Dirty metadata
// inside that immutable prefix is never scanned; only new, valid rows are read.
// A writer can preserve this prefix only when replacing a proven-zero row.
void kv_max_emit(int id,KVMaxCache::Impl *kr,KVMaxCache::Impl *vr,int ks,int vs,int count,void *peaks,int kold,int vold){
 auto &e=*kr->engine;load_kernel(e.reduce,"kunlun-kv-max-reduce",id);
 const void *km=kr->maxima->data(),*vm=vr->maxima->data(),*kp=kr->partials?kr->partials->data():km,*vp=vr->partials?vr->partials->data():vm;
 int krow=kr->pending,vrow=vr->pending;void *kg=kr->global->data(),*vg=vr->global->data();
 native_launch(e.reduce,{native_arg(peaks),native_arg(km),native_arg(vm),native_arg(ks),native_arg(vs),native_arg(count),native_arg(kp),native_arg(vp),native_arg(krow),native_arg(vrow),native_arg(kg),native_arg(vg),native_arg(kold),native_arg(vold)},8);
 kr->pending=vr->pending=-1;kr->global_valid=vr->global_valid=true;kr->seed_requested=vr->seed_requested=false;
 kr->ever_seeded=vr->ever_seeded=true;kr->extent_invalidated=vr->extent_invalidated=false;kr->cached_start=ks;vr->cached_start=vs;kr->cached_count=vr->cached_count=count;
 kv_stats[id].meta_read_bytes+=(kold==-2?32:(size_t)(2*count-std::max(kold,0)-std::max(vold,0))*4+8)+(krow>=0?132:0)+(vrow>=0?132:0);
}
bool kv_max_prepare(int id,const void *k,const Tensor4Layout &pk,const void *v,const Tensor4Layout &pv,int queries,void *peaks,xdnn::Context *vendor,bool tensor_parallel){
 KVDiag diag{"lookup",id,size_t(pk.ne[1])|(uint64_t(queries)<<32)};
 auto *kr=kv_find(id,k,tensor4_span(pk)),*vr=kv_find(id,v,tensor4_span(pv));int ks=0,vs=0;bool owned=kv_view(kr,k,pk,ks)&&kv_view(vr,v,pv,vs)&&kr!=vr;
 const int mode=kv_mode();
 const bool tp_rows=tensor_parallel&&mode==1&&kv_opt_in("GGML_KUNLUN_TP_KV_MAX_ROWS")&&
     pk.ne[0]==128&&pk.ne[2]==16&&pv.ne[0]==128&&pv.ne[2]==16&&
     pk.nb[0]==2&&pk.nb[1]==4096&&pk.nb[2]==256&&
     pv.nb[0]==2&&pv.nb[1]==4096&&pv.nb[2]==256&&
     pk.ne[1]>=256&&pk.ne[1]<=1024&&pk.ne[1]%256==0&&queries==1;
 bool eligible=owned&&mode&&(mode==2||(queries==1&&(tp_rows||pk.ne[1]>=std::max(kv_min_s(kr->features),kv_min_s(vr->features)))))&&!kr->poison&&!vr->poison;
 const char *rebuild=std::getenv("KUNLUN_FLASH_KV_MAX_REBUILD"),*reduce=std::getenv("KUNLUN_FLASH_KV_MAX_REDUCE");
 bool row_path=(rebuild&&!std::strcmp(rebuild,"rows"))||(reduce&&!std::strcmp(reduce,"linear"));
 bool unverified=false;
 if(eligible){for(int w=0;w<2;++w){auto *r=w?vr:kr;if(r->all_finite)continue;int first=w?vs:ks;for(int row=first;row<first+pk.ne[1];++row)if(!r->finite[row]){unverified=true;break;}}}
 if(unverified&&kv_mode()==2)row_path=true;
 if(row_path){diag.result("row-diagnostic");return kv_max_prepare_rows(id,k,pk,v,pv,queries,peaks,vendor);}
 auto &s=kv_stats[id];++s.lookups;auto fallback=[&](){s.full_calls+=2;s.scan_bytes+=(tensor4_count(pk)+tensor4_count(pv))*2;return false;};
 if(owned){kr->last_s=pk.ne[1];vr->last_s=pv.ne[1];kr->seed_requested=vr->seed_requested=false;}
 if(!mode){diag.result("fallback-legacy");++s.fallback_legacy;return fallback();}
 if(!owned){diag.result("fallback-unowned");++s.fallback_unowned;return fallback();}
 if(mode!=2&&queries!=1){diag.result("fallback-pp");++s.fallback_pp;return fallback();}
 if(mode!=2&&!tp_rows&&pk.ne[1]<std::max(kv_min_s(kr->features),kv_min_s(vr->features))){
  diag.result("fallback-short");++s.fallback_short;return fallback();
 }
 if(kr->poison||vr->poison){diag.result("fallback-poison");++s.fallback_poison;return fallback();}
 if(unverified){diag.result("fallback-unverified");++s.fallback_unverified;return fallback();}
 kr->ensure();vr->ensure();
 auto prefix=[&](KVMaxCache::Impl *r,int start){if(!r->global_valid||r->cached_start!=start||r->cached_count>pk.ne[1])return -1;int first=start+r->cached_count,end=start+pk.ne[1];if(first<end)for(auto d:r->dirty)if(d.first<end&&d.second>first)return -1;return r->cached_count;};
 int ko=prefix(kr,ks),vo=prefix(vr,vs);
 const bool missing_prefix=ko<0||vo<0;
 if(missing_prefix){
  if(!tp_rows||((ko<0&&!kr->rows_known(ks,ks+pk.ne[1]))||(vo<0&&!vr->rows_known(vs,vs+pk.ne[1])))){
   diag.result(kr->extent_invalidated||vr->extent_invalidated?"seed-write":
       kr->ever_seeded||vr->ever_seeded?"seed-view":"seed-cold");
   kr->seed_requested=vr->seed_requested=true;return fallback();
  }
  diag.result("hit-rows");
 }else{
  diag.result(ko<pk.ne[1]||vo<pk.ne[1]?"hit-extend":
      kr->pending>=0||vr->pending>=0?"hit-update":"hit-warm");
 }
 kv_max_emit(id,kr,vr,ks,vs,pk.ne[1],peaks,ko,vo);++s.hits;
 if(kv_mode()==2){
  auto &e=*kr->engine;native_reserve(e.reference,id,1024);void *reference=e.reference->data(),*vref=(char*)reference+256;
  check(xdnn::findmax<float16>(vendor,(const float16*)k,(float*)reference,(int)tensor4_count(pk)));check(xdnn::findmax<float16>(vendor,(const float16*)v,(float*)vref,(int)tensor4_count(pv)));check(xpu_wait());
  float a[256]={},b[256]={};check(xpu_memcpy(a,peaks,1024,XPU_DEVICE_TO_HOST));check(xpu_memcpy(b,reference,1024,XPU_DEVICE_TO_HOST));check(xpu_wait());s.verify_calls+=2;
  for(int w=0;w<2;++w){float x=0,y=0;for(int j=0;j<4;++j){x=std::max(x,a[w*64+j]);y=std::max(y,b[w*64+j]);}if(std::memcmp(&x,&y,4)){++s.verify_mismatches;auto *r=w?vr:kr;std::fprintf(stderr,"KV_MAX_MISMATCH pd=%d cache=%s generation=%llu revision=%llu start=%d S=%d cached=%a SDK=%a partials=%a,%a,%a,%a\n",id,r->name.c_str(),(unsigned long long)r->generation,(unsigned long long)r->revision,w?vs:ks,pk.ne[1],x,y,b[w*64],b[w*64+1],b[w*64+2],b[w*64+3]);for(auto ev:r->history)std::fprintf(stderr,"KV_MAX_HISTORY revision=%llu kind=%s lo=%d hi=%d\n",(unsigned long long)ev.revision,ev.kind.c_str(),ev.lo,ev.hi);require(false);}}
 }
 return true;
}
void kv_max_finish(int id,const void *k,const Tensor4Layout &pk,const void *v,const Tensor4Layout &pv,void *peaks){
 KVDiag diag{"seed-check",id,size_t(pk.ne[1])};
 auto *kr=kv_find(id,k,tensor4_span(pk)),*vr=kv_find(id,v,tensor4_span(pv));int ks=0,vs=0;
 if(!kv_view(kr,k,pk,ks)||!kv_view(vr,v,pv,vs)||kr==vr||!kr->seed_requested||!vr->seed_requested)return;
 diag.result("seed-emit");
 kv_max_emit(id,kr,vr,ks,vs,pk.ne[1],peaks,-2,-2);++kv_stats[id].seeds;
}
}
KVMaxStatistics kv_max_statistics(int id){std::lock_guard<std::recursive_mutex>lock(mutex);require(id>=0&&id<2);return kv_stats[id];}
void kv_max_log_statistics(){std::lock_guard<std::recursive_mutex>lock(mutex);for(int id=0;id<2;++id){auto &s=kv_stats[id];if(!s.lookups)continue;std::fprintf(stderr,"KV_MAX pd=%d lookups=%zu hits=%zu seeds=%zu full=%zu verify=%zu mismatch=%zu rebuilds=%zu rows=%zu writer_calls=%zu writer_rows=%zu invalidations=%zu retired=%zu metadata=%zu scan_bytes=%zu meta_read=%zu fallback_legacy=%zu pp=%zu short=%zu unowned=%zu poison=%zu unverified=%zu\n",id,s.lookups,s.hits,s.seeds,s.full_calls,s.verify_calls,s.verify_mismatches,s.rebuilds,s.rebuild_rows,s.writer_calls,s.writer_rows,s.invalidations,s.retired,s.metadata_bytes,s.scan_bytes,s.meta_read_bytes,s.fallback_legacy,s.fallback_pp,s.fallback_short,s.fallback_unowned,s.fallback_poison,s.fallback_unverified);}}
