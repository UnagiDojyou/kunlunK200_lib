from pathlib import Path
R=Path('/home/unagidojyou/kunlun-k200');D=R/'repos/llama.cpp/examples/kunlun'
L=R/'repos/KunlunK200'
for name in ['kunlun-flash-decode-online','kunlun-flash-decode-online-head','kunlun-flash-profile-decode-online']:
 p=L/'kernels'/(name+'.xpu');s=p.read_text().replace('#define ONLINE_REUSE 0','#define ONLINE_REUSE 1');p.write_text(s)
for name in ['flash.inc','flash-decode.inc']:
 p=L/'src'/name;s=p.read_text();s=s.replace('tick=now;if(profile)flash_profile_report(name,s);','tick=now;if(profile){flash_profile_report(name,s);tick=std::chrono::steady_clock::now();}').replace('tick=now;if(profile)flash_profile_report(name,s,stats);','tick=now;if(profile){flash_profile_report(name,s,stats);tick=std::chrono::steady_clock::now();}').replace(' if(profile){}\n','');p.write_text(s)
s=(D/'llm-smoke.cpp').read_text();s=s[:s.index('    int repeats=setting("K200_REPEATS"')]
s+=r'''
    // Controlled A/B benchmark only. Save the common prefill once; restoring
    // the identical KV state is outside every timed TG region. No trace callback
    // or additional wait is used inside process(). Rotate modes between trials.
    if(teacher.empty())return 2;
    const auto all_tokens=tokens;std::vector<size_t> lengths;
    if(const char *raw=std::getenv("K200_PP_LENGTHS"))while(*raw){
        if(*raw==','||*raw==' '){++raw;continue;}char *end=nullptr;long n=std::strtol(raw,&end,10);
        if(end==raw||n<1||(size_t)n>all_tokens.size())return 2;lengths.push_back(n);raw=end;
    }else lengths.push_back(all_tokens.size());
    std::vector<std::string> modes;std::string raw=std::getenv("K200_FD_MODES")?std::getenv("K200_FD_MODES"):"off,on";
    for(size_t first=0;first<raw.size();){size_t end=raw.find(',',first);if(end==std::string::npos)end=raw.size();modes.push_back(raw.substr(first,end-first));first=end+1;}
    int repeats=setting("K200_REPEATS",3),nv=llama_vocab_n_tokens(vocab);
    for(size_t length:lengths){
        tokens.assign(all_tokens.begin(),all_tokens.begin()+length);llama_memory_clear(llama_get_memory(ctx),true);setenv("KUNLUN_FLASH_DECODING","off",1);
        float *logits=nullptr;double pp_ms=0;
        for(size_t pos=0;pos<tokens.size();){int count=std::min<size_t>(cp.n_batch,tokens.size()-pos);start=Clock::now();logits=process(tokens.data()+pos,count,pos);pp_ms+=elapsed(start);if(!logits)return 1;pos+=count;}
        std::vector<float> initial(logits,logits+nv);size_t bytes=llama_state_get_size(ctx);if(!bytes||bytes>size_t(4)*1024*1024*1024)return 1;
        std::vector<uint8_t> state(bytes);size_t saved=llama_state_get_data(ctx,state.data(),state.size());if(!saved||saved>state.size())return 1;state.resize(saved);
        std::fprintf(stderr,"FDBENCH_STATE prefix=%zu bytes=%zu prefill_ms=%.6f restore_outside_timer=1\n",length,saved,pp_ms);
        std::map<std::string,unsigned long long> checksums;
        for(int trial=-1;trial<repeats;++trial)for(size_t order=0;order<modes.size();++order){
            const auto &md=modes[(order+(trial+1))%modes.size()];std::string mode=md;size_t colon=md.find(':');
            unsetenv("KUNLUN_FLASH_DECODING_CHUNKS");
            if(colon!=std::string::npos){mode=md.substr(0,colon);setenv("KUNLUN_FLASH_DECODING_CHUNKS",md.substr(colon+1).c_str(),1);}
            if(mode!="on"&&mode!="off"&&mode!="auto")return 2;setenv("KUNLUN_FLASH_DECODING",mode.c_str(),1);
            llama_memory_clear(llama_get_memory(ctx),true); // Match fresh prefill padding in every mode.
            if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())return 1;logits=initial.data();
            FILE *dump=nullptr;if(trial==0&&argc>4){std::string file=std::string(argv[4])+".p"+std::to_string(length)+"."+md;dump=std::fopen(file.c_str(),"wb");if(!dump)return 1;}
            double tg_ms=0;int steps=0;unsigned long long hash=1469598103934665603ull;
            for(int i=0;i<predict;++i){
                for(int j=0;j<nv;++j){if(!std::isfinite(logits[j]))return 1;unsigned int word;std::memcpy(&word,logits+j,4);hash^=word;hash*=1099511628211ull;}
                if(dump&&std::fwrite(logits,4,nv,dump)!=(size_t)nv)return 1;
                if(i+1==predict)break;llama_token next=teacher[i];start=Clock::now();logits=process(&next,1,tokens.size()+i);tg_ms+=elapsed(start);++steps;if(!logits)return 1;
            }
            if(dump)std::fclose(dump);
            if(!checksums.count(md))checksums[md]=hash;else if(checksums[md]!=hash){std::fprintf(stderr,"non-repeatable logits mode=%s\n",md.c_str());return 1;}
            if(trial>=0)std::fprintf(stderr,"FDBENCH trial=%d order=%zu prefix=%zu mode=%s tg_ms=%.6f steps=%d tps=%.9f logits_hash=%llu\n",trial,order,length,md.c_str(),tg_ms,steps,steps*1000./tg_ms,hash);
        }
    }
    llama_batch_ext_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
int main(int argc,char **argv){return smoke_run(argc,argv);}
'''
(D/'flash-decoding-bench.cpp').write_text(s)
p=D/'CMakeLists.txt';s=p.read_text()
if 'add_executable(kunlun-flash-decoding-bench' not in s:s+='\nadd_executable(kunlun-flash-decoding-bench flash-decoding-bench.cpp)\ntarget_link_libraries(kunlun-flash-decoding-bench PRIVATE llama ggml ggml-base ${CMAKE_DL_LIBS})\n'
p.write_text(s)
print('Saved isolated A/B benchmark; original smoke tool unchanged')
