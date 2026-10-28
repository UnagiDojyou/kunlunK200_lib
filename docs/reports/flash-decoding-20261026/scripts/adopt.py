from pathlib import Path
import os,json,subprocess,hashlib
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';L=R/'repos/KunlunK200'
for name in ['VALIDATION_DONE','AB_DONE','AUDIT_DONE','WORK_DONE','FIXED_DONE']:assert (O/name).exists()
assert not (O/'XPU_ABORT').exists()
for row in json.loads((O/'work-quality.json').read_text()):assert row['work_bitwise']
fixed=json.loads((O/'fixed-quality.json').read_text());assert all(r['bitwise'] for r in fixed if r['test'].startswith('corpus-'))
assert all(r['gain']>0.05 for r in fixed if r.get('model')=='tiny' and r.get('mode')=='fixhead')
D=R/'build/flash-decode-llama/kunlunk200/lib/kunlunk200/kernels';(O/'pre-adoption-images.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.bin')},indent=2))
p=L/'src/flash-decode.inc';s=p.read_text();old='const char *mode=std::getenv("KUNLUN_FLASH_DECODING");\n if(base.m!=1||!mode||std::strcmp(mode,"on"))return false;'
new='''if(base.m!=1)return false;
 const char *mode=std::getenv("KUNLUN_FLASH_DECODING");bool automatic=!mode||!std::strcmp(mode,"auto"),forced=mode&&!std::strcmp(mode,"on");
 if(!automatic&&!forced)return false;
 const char *kq_mode=std::getenv("KUNLUN_FLASH_KQ"),*online_mode=std::getenv("KUNLUN_FLASH_ONLINE"),*pv_mode=std::getenv("KUNLUN_FLASH_PV");
 if((kq_mode&&std::strcmp(kq_mode,"auto")&&std::strcmp(kq_mode,"reuse1024"))||(online_mode&&std::strcmp(online_mode,"auto"))||(pv_mode&&!std::strcmp(pv_mode,"narrow")))return false;
 // Only adopt the real-LLM shape/layout with measured end-to-end gains.
 // Nongrouped 7B TG already fills all four SDNNs and did not improve.
 bool measured=base.d==64&&base.h==32&&base.hk==4&&base.b==1&&base.tg_group&&base.block==1024&&base.s>=2048&&base.s<=4352&&base.has_mask&&base.mask_type&&base.max_bias==0&&base.k[1]==base.d*base.hk*2&&base.k[2]==base.d*2&&base.v[1]==base.d*base.hk*2&&base.v[2]==base.d*2;
 bool hbm_requested=std::getenv("KUNLUN_FLASH_HBM")&&!std::strcmp(std::getenv("KUNLUN_FLASH_HBM"),"1");
 if(automatic&&(!measured||hbm_requested))return false;
 bool custom=std::getenv("KUNLUN_FLASH_DECODING_CHUNK")||std::getenv("KUNLUN_FLASH_DECODING_CHUNKS")||std::getenv("KUNLUN_FLASH_DECODING_FIXED_KEYS");
 bool tuned_fixed=measured&&!custom&&base.s>4096;
'''
assert old in s;s=s.replace(old,new).replace('int fixed=0;if(const char *raw=','int fixed=tuned_fixed?1024:0;if(const char *raw=')
s=s.replace('bool affinity=head&&std::getenv("KUNLUN_FLASH_DECODING_WORK")&&!std::strcmp(std::getenv("KUNLUN_FLASH_DECODING_WORK"),"head");','const char *work=std::getenv("KUNLUN_FLASH_DECODING_WORK");bool affinity=head&&((!work||!std::strcmp(work,"auto"))?tuned_fixed:!std::strcmp(work,"head"));')
p.write_text(s)
p=L/'CMakeLists.txt';s=p.read_text().replace('VERSION 0.7.2','VERSION 0.7.3');p.write_text(s)
(L/'docs/flash-decoding.md').write_text('''# Flash-Decoding (0.7.3)

`KUNLUN_FLASH_DECODING=auto/off/on` selects the TG path; unset means auto. PP remains the 0.7.2 FlashAttention path. Auto is deliberately limited to the measured TinyLlama-like shape: M1, D64, Hq32/Hkv4, batch1, interleaved compact F16 K/V, F16 broadcast causal-style mask without ALiBi, default TG grouping/block, padded S2048..4352. S>4096 uses fixed1024 chunks and head affinity; the other adopted contexts use balanced chunks. Auto keeps nongrouped Llama2 7B on 0.7.2 because its LLM measurements did not improve. Force on only for controlled experiments; it is not faster on every shape.

The unchanged public FlashAttention API handles common cast/gather/findmax/range once, then launches KQ, chunk-local ordered Online Softmax, Wide KQV and an FP32 reduction. Multiple waves are supported. L3 Score/P is shared in place; chunk states are independent. Metadata is private192-byte aligned storage, not a public ABI change. Library and shaders must come from the same build.

Tuning: `CHUNKS`=1..64, `CHUNK`=256/512/1024 (target length rounded to a power-of-two count in balanced mode), under the `KUNLUN_FLASH_DECODING_` prefix. `FIXED_KEYS`=32/64/128/256/512/1024 instead fixes the actual chunk length and determines the count, overriding CHUNKS/CHUNK. `WORK=chunk/head/auto` selects task ordering; head affinity applies to the default head-major state layout. `LAYOUT=head/chunk` selects state layout. Explicit tuning overrides the automatic fixed1024 choice. Counts are clamped to existing32-key tiles; requests over64 chunks or16MiB fall back. Short/unsupported/unmeasured auto shapes also use0.7.2. KQ legacy/unknown, Online legacy/unknown and narrow KQV explicitly select0.7.2 rather than silently changing these controls. Existing separate Attention dispatch and GEMV CL fallback are unchanged.

`KUNLUN_FLASH_TRACE` and `KUNLUN_FLASH_DECODING_PROFILE` are diagnostics with additional waits. Unset both (including values equal to0) for normal timings. Profiling kernels use unused private metadata padding, never model/KV buffers. The benchmark target `k200-flash-decode-bench` takes PD D S M HK Hq pattern [batch]. COMPARE_LAYOUT/COMPARE_WORK/COMPARE_FIXED compare controlled variants; their old_ms then refers to that variant, not0.7.2. Without those flags it compares0.7.2 and CPU FP64. Pattern29 uses interleaved K/V and an F16 mask with255 trailing masked keys. The llama.cpp target `kunlun-flash-decoding-bench` performs alternating TG comparisons from identical cleared/restored KV states, outside the timed region; each mode must reproduce its logits checksum.

See the Japanese report and reproduction scripts under docs/reports/flash-decoding-20261026. General generation quality and context lengths beyond the measured LLM domain are not inferred from small unit errors.
''')
p=L/'docs/flash-attention.md';s=p.read_text();s+='\n## 0.7.3 Flash-Decoding\n\nSee [TG chunk execution, dispatch and benchmarks](flash-decoding.md). PP and the0.7.2 fallback remain unchanged.\n';p.write_text(s)
P=R/'repos/llama.cpp/docs/backend/kunlun-flash-decoding.md';P.write_text('''# Kunlun K200 Flash-Decoding

KunlunK2000.7.3 adds TG chunk execution inside the existing FLASH_ATTN_EXT backend operator. Graph construction and the public library API are unchanged. Enable the existing FlashAttention flags and `-fa on` (test tools: K200_FLASH=1). KUNLUN_FLASH_DECODING=auto/off/on controls the path. Auto uses only measured TinyLlama-like long-context shapes; nongrouped7B keeps0.7.2. Legacy KQ/Online and narrow KQV choices remain honored by falling back.

See vendor/KunlunK200/docs/flash-decoding.md and its Japanese evaluation report for tuning, accuracy, performance limits and reproduction. The new kunlun-flash-decoding-bench target is a controlled teacher-forced A/B tool: state clear/restore and logits checksums are outside timed llama_process calls. The original smoke tool remains unchanged.
''')
policy=dict(version='0.7.3',auto='M1 D64 H32 HK4 B1, grouped, default block1024, interleaved compact KV, F16 mask/no ALiBi, padded S2048..4352',long='S>4096 fixed1024 + head affinity; otherwise balanced',seven_b='0.7.2 fallback: no measured LLM gain',reason='both-PD long Tiny gains, bitwise worker mapping, controlled PPL and teacher-forced quality')
(O/'auto-policy.json').write_text(json.dumps(policy,indent=2));print(json.dumps(policy,indent=2))
env=os.environ.copy();env.update(XRE=str(R/'external/xre'),XDNN=str(R/'external/xdnn'),XTDK_DIR=str(R/'external/xtdk'))
for build,targets in [('flash-decode-lib',['k200-flash-decode-bench','k200-flash-contract','k200-attention-contract','k200-common-q4-layout']),('flash-decode-llama',['kunlun-flash-decoding-bench','kunlun-llm-smoke','kunlun-quality-eval','kunlun-attention-test','kunlun-swiglu-test','kunlun-backend-contract'])]:subprocess.run(['cmake','--build',str(R/'build'/build),'-j4','--target',*targets],env=env,check=True)
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.bin')};before=json.loads((O/'pre-adoption-images.json').read_text());assert after==before
print('ADOPTED; kernel images identical to measured fixed/worker builds',flush=True)
