from pathlib import Path
import json,re,statistics,math
R=Path('/home/unagidojyou/kunlun-k200');O=R/'results/flash-decoding-20261026';D=O/'delivery';D.mkdir(exist_ok=True)
def load(n):return json.loads((O/(n+'.json')).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rows)+'\n'
def fmt(v,n=3):return f'{v:.{n}f}'
def fields(s):return dict(re.findall(r'(\w+)=([^\s]+)',s))
units=load('units');ab=load('ab-performance');profiles=load('profiles-final');normal=load('performance-comparison');validation=load('validation');final=load('final-tests');memory=load('final-memory');verify=load('final-verification');selected=load('selected-quality')
out=[]
def w(s):out.append(s+'\n')
w('# Kunlun K200 Flash-Decoding 0.7.3 実装・実機評価\n')
rr=[r for r in ab if r['test'].startswith('final-auto-') and r['mode']=='auto']
w('**Flash-Decodingを実装し、実機検証を完了した。** 最終autoのTinyLlama TG@4096は、'+ '、'.join(f"PD{r['pd']}で{r['gain_pct']:.1f}%高速化" for r in rr)+'。7Bでは改善を確認できなかったため、autoは0.7.2 FlashAttentionを使う。PPは既存経路を維持した。単体APIの大きい改善率をLLM全体へ外挿しない方針で採用範囲を限定した。\n')
w('## 1. 比較基準・環境・保存\n')
w('作業前の未コミット変更を保存して0.7.2をコミットした。libraryは `d8bd87f56ae95872287c4f4a030b7ac7e68dd7b9`、llama.cppは `8d62617cb1d267d9283d5fcffe8e66c57820dfcc`。作業branchは `codex/k200-flash-decoding-20261026`。凍結した `build/flash-kq-lib` / `build/flash-kq-llama` を上書きせず、新版は `build/flash-decode-lib` / `build/flash-decode-llama` に構築した。\n')
w('実機は192.168.50.112のK200/KL1、XRE4.33、XTDK2.7.98.2、表示clock900MHz。各PDはSDNN4基、CL4 cluster×16 core、FP32演算8 lane、L3は16MiB、HBMは8064MiB。各SDNNのL1D512KiB、L1W2MiB、L1E256KiB、各L2 bank128KiBを前提に、DMA一要求128KiBの制約も維持した。\n')
w('TinyLlama1.1B Q4_0とLlama2 7B Q4_0、batch/ubatch128、CPU thread6。既存SDNN KQ/KQV、ordered Online Softmax、KV writer、Q4、RoPE、SwiGLUの旧shader52個は、両ビルドで0.7.2とSHA256一致。SwiGLU=CPU等、Attention以外の条件も固定した。通常測定ではTRACE、PROFILE、LD_PRELOAD監査、stateチェックを無効にした。\n')
w('## 2. 従来TGとFlash-Decodingの並列性の違い\n')
w('旧版はKV blockごとにKQ→Online Softmax→KQVを逐次実行する。KQは少ないKV headに対して既にkey範囲を分割し、4基のSDNNを使っていた。新方式はchunkごとの統計・出力を分離し、KQVとCLにもchunk方向のworkを加える。単にKQのkey分割数を増やす方式ではない。\n')
rows=[]
for hk,name in [(1,'profile-p0-hk1-c4'),(4,'profile-p0-hk4-c4'),(32,'profile-fixed-p0')]:
 for mode in ['legacy','decode']:
  p=next(r for r in profiles if r['test']==name and r['mode']==mode);s=p['stages'];rows.append([hk,'0.7.2' if mode=='legacy' else 'chunk',','.join(map(str,s['kq']['active'])),','.join(map(str,s['pv']['active'])),','.join(map(str,s['online']['active'])),sum(x['launches'] for x in s.values())])
w(table(['HK','方式','KQ active SDNN/起動','KQV active SDNN/起動','Online active core/起動','core起動数'],rows))
w('Hq32、D128、HK1/4はS4096・4 chunk、HK32はS4352・固定1024の5 chunk。activeはMAC命令または処理行数を記録したworker数で、PMU利用率%ではない。旧版もCLの4 clusterを使うが各cluster8 core、新版は最大16 coreを使う。HK4/32では旧版でも全SDNNが稼働し、追加のSDNN並列性があると解釈してはならない。HK1のKQVは1基→4基となる。\n')
w('## 3. アルゴリズム・カーネル・共有メモリ\n')
w('共通cast/gather、K/V findmax、Q rangeはAPIごとに1回だけ実行する。その後、全chunkのKQを1起動、全chunkのOnline Softmaxを1起動、KQVを1起動にまとめ、最後にCL reductionを起動する。1-waveでは計4起動。chunkごとに既存FA APIを呼び出す構造ではない。段階間はdefault stream上で逐次実行し、SDNN/CLの全面的な同時pipelineは実装していない。並列性は同じ起動内のhead×chunkのwork分配から得る。\n')
w('各chunkは最大値m、指数和l、未正規化accと最後のKQV partialを持つ。次waveでは前のpartialを取り込み、最大値の更新に応じてaccとlを再スケーリングする。短いchunkが後のwaveを処理しない場合、最後の状態・partialを保持する。最終reductionはFP32で次を計算する。\n\n```text\nM = max_i(m_i)\nw_i = exp(m_i - M)\nL = sum_i(w_i * l_i)\nO = sum_i(w_i * (acc_i + last_partial_i))\nY = O / L\n```\n')
w('全mask chunkを除外し、全mask行はゼロ出力。-Inf同士の差をexpへ渡さない。native expと既存ordered-sumを維持し、近似expは導入していない。試作で外れていた既存FP16 maskのzero/-Inf wordショートカットを引き継いだ版も、旧試作LLM出力と一致した。\n')
w('QのFP16丸め、SDNN DMAのINT16変換、K/V scaleの算出方式は変更していない。SDK findmaxの4部分最大値を全てreduceする。SDNNは従来のINT16 MAC＋FP32可視出力、chunk統合はCL FP32。今回MAC内部のIEEE FP32累積を新たに実証したものではない。\n')
w('Score/Pは同じL3領域を再利用する。Score/Pはchunk-major、partial/acc/statsはhead-major。reductionは最大8 chunkを連続ロードし、状態をchunkごとに細かく読む方式も比較用に残した。SDNNは2D DMAで直接head-majorへ書き込み、別transpose kernelは使わない。各SDNNのcontroller core0が自身のL1D/L1W/L2を所有し、競合する別streamは使用しない。QはL1D、KはL1Wで複数32-key tileに再利用する0.7.2の構造を保つ。\n')
w('workspaceは `B * Hq * C * (4 * block + 8 * D + 32)` byte。16MiB超過・L3確保失敗・未対応条件は旧版へ戻る。chunk数を32-key tile数以下へclampし空chunkを生成しない。metadataはprivate192-byte構造で公開API変更なし。同じビルドのlibraryとshaderを組で使用する。\n')
frames=[]
for n,v in verify['frames'].items():
 if n in ['kunlun-flash-decode-kq.device','kunlun-flash-decode-online-head.device','kunlun-flash-decode-pv-head-affinity.device','kunlun-flash-decode-reduce-head.device']:frames.append([n,','.join(v['frames']),v['bytes']])
w(table(['kernel','生成assemblyのstack frame調整byte','binary byte'],frames))
w('CL LMとstackは共用。compile-native.shのframe上限16256byte検査を全kernelで通した。L1WにはtaskのK/V範囲を置くので、chunk数分のL1容量を同時に要求しない。D128・1024 keyは256KiBで、128KiBずつ2回DMAする。\n')
w('## 4. chunk長・worker配置・性能を左右する要因\n')
w('balancedは32-key tileを均等分割し、当初はchunk数を2の冪へ切り上げた。実TGではKV長がpaddingされるため、prefix2048はS2304、prefix4096はS4352になる。S4352を8 chunkにすると544 keyずつになり、DMA/DS/MAC設定の回数が増えた。3/5 chunkも比較し、固定1024×4＋256なら旧版のMAC/DMA形状を保てることを検証した。\n')
rows=[]
for n,label in [('ab-tiny-p0','初期balanced'),('ab-7b-p0','初期balanced'),('work-ab-tiny-p0','割当比較'),('work-ab-7b-p0','割当比較'),('fixed-ab-tiny-p0','固定長比較'),('fixed-ab-7b-p0','固定長比較')]:
 for r in ab:
  if r['test']==n:rows.append([label,r['model'],r['mode'],fmt(r['median']),fmt(r['gain_pct'],2),f"{r['minimum']:.3f}..{r['maximum']:.3f}"])
w(table(['比較series','model','mode','TG token/s','旧版比%','反復min..max'],rows))
w('onはbalanced試作、on:5はbalanced5、head:5は同じ境界のhead affinity、fixed/fixheadは固定1024のchunk/head割当。この探索seriesは当時のdefaultを記録しており、最終autoとは区別する。同じ境界でworkerの割当だけを変えた64件のAPI試験と両モデル/両PDのLLM試験は出力ビット一致。配置変更の効果は量子化差と分離できた。\n')
w('実KV Cacheに近いinterleaved配置も測定した。dense tensorの単体結果だけではLLM性能を予測できない。下表は同じ論理値・同じF32 maskでKV配置だけを変えた対照試験。\n')
rows=[]
for r in units:
 if r['test'].startswith('layout-control-p0-'):rows.append([r['D'],r['pattern'],'dense' if r['pattern']==0 else 'interleaved',fmt(r['old_ms']),fmt(r['flash_ms'])])
w(table(['D','pattern','KV配置','0.7.2 API ms','固定head API ms'],rows))
w('配置によるDMA区間の違いは確認できるが、HBM bankやPMU counterから競合原因を直接証明したものではない。7Bは既に4基のSDNNを使い、KV転送量を減らさないchunk分割だけでは全体の利益が出なかった。\n')
w('## 5. API・段階別・DMAの比較\n')
rows=[]
for r in units:
 if r['test'].startswith('auto-unit-') and r['S'] in [2048,2304,4096,4352]:rows.append([r['pd'],r['S'],fmt(r['old_ms']),fmt(r['flash_ms']),fmt(r['old_ms']/r['flash_ms'],2),f"{r['cpu_nmse']:.2e}"])
w(table(['PD','S','旧API ms','最終auto API ms','倍率','CPU NMSE'],rows))
w('APIはcast/gather、findmax/range、metadata、workspace、起動と完了待ちを含む。初回cold callを除く中央値。通常timingに追加同期を混ぜていない。\n')
rows=[]
for name in ['profile-p0-hk1-c4','profile-fixed-p0']:
 for mode in ['legacy','decode']:
  p=next(r for r in profiles if r['test']==name and r['mode']==mode);s=p['stages'];rows.append([name,mode,fmt(s['kq']['critical_ms']),fmt(s['online']['critical_ms']),fmt(s['pv']['critical_ms']),fmt(s.get('reduce',s.get('final'))['critical_ms'])])
w(table(['診断case','方式','KQ ms','Online ms','KQV ms','最終CL ms'],rows))
w('get_clockの900MHz換算。SDNNは各起動の4基中最大、CLは各起動の64 worker中最大を加算した値。fence区間には命令発行・制御・待ちが含まれ、純DMA時間や利用率%ではない。diagnosticは追加waitや計測値転送を含むため、この和を通常API時間やtoken/sに置き換えない。PROFILE時のFLASH_UNITも通常性能表から除外した。詳細はprofiles-final.json/csvのphase_ms（Q DMA、Q DS、K/V DMA、DS、MAC、EW、RS、DMAOUT、other）。\n')
rows=[]
for name in ['profile-p0-hk1-c4','profile-fixed-p0']:
 for mode in ['legacy','decode']:
  p=next(r for r in profiles if r['test']==name and r['mode']==mode);k=p['stages']['kq'];v=p['stages']['pv'];rows.append([name,mode,k['t10'],k['t11'],k['t14'],v['t10'],v['t11'],v['t14']])
w(table(['case','方式','KQ Q-DMA','KQ K-DMA','KQ MAC命令','KQV P-DMA','KQV V-DMA','KQV MAC命令'],rows))
w('S4096/HK1ではQ転送が128KiB→32KiBへ減り、Kの論理転送量1MiBは同じ。S4352/D128/HK32では旧版と固定5のK/V DMAは各288回、Q-DMA160回、KQV MAC1280命令。balanced8はソース上の計数でそれぞれ512回、256回、2048命令になる（byte量は同じでも要求数が増える）。K/VのMAC読出し量は両方式で同じ（S4096は64MiB、S4352は68MiB）で、findmaxも同じKVを読む。L3 Score/Pの基本読書き量はHq*S*16byteでS4096/Hq32なら2MiB。\n')
w('## 6. PP/TGの通常推論と最終auto\n')
w('以下は0.7.2 offと初期balanced試作onの通常smoke比較。warmup1＋2反復、63 decode。FD以外の設定を固定し、診断・stateチェック・LD_PRELOADなし。PPのcode pathは共通で、差には測定変動を含む。\n')
w(table(['model','PD','prefix','旧PP t/s','試作PP t/s','旧TG t/s','試作TG t/s','TG差%'],[[r['model'],r['pd'],r['context'],fmt(r['old_pp'],2),fmt(r['new_pp'],2),fmt(r['old_tg']),fmt(r['new_tg']),fmt(r['tg_gain_pct'],2)] for r in normal]))
w('最終採用版をauto指定で再確認した結果：\n')
rows=[]
for r in rr:
 old=next(x for x in ab if x['test']==r['test'] and x['mode']=='off');rows.append([r['pd'],fmt(old['median']),fmt(r['median']),fmt(r['gain_pct'],2),f"{r['minimum']:.3f}..{r['maximum']:.3f}",r['trials']])
w(table(['PD','0.7.2 TG@4096','auto TG@4096','改善%','auto min..max','反復'],rows))
w('共通prefillの後、未使用領域もclearして同じKV stateを復元しmode順を循環させる交互比較。clear/state転送・dump・checksumはTG timer外で、測定中は通常のllama_process()だけを呼ぶ。最終再確認はwarmup1＋2反復、探索は3反復。各mode内の全反復でlogits checksum一致。最終autoの出力は採用した固定head指定の出力ともビット一致した。\n')
w('4k fixtureはprompt2048.idsを2回連結したもの。context確保は4608で、自然長文の標準PPLではない。LLMのTGはprefixから63 token進むため、単体の固定S時間とは異なる。APIはS8192/16384も検証したが、LLM評価はTG@4096まで。\n')
w('## 7. Layer/Row Split・memory・互換性\n')
sp=load('split-performance');rows=[]
for r in sp:
 if r['mode']!='auto':continue
 old=next(x for x in sp if x['model']==r['model'] and x['split']==r['split'] and x['context']==r['context'] and x['mode']=='off');rows.append([r['model'],r['split'],r['context'],fmt(old['pp_tps'],2),fmt(r['pp_tps'],2),fmt(old['tg_tps']),fmt(r['tg_tps'])])
w(table(['model','split','prefix','旧PP','auto PP','旧TG','auto TG'],rows))
w('Splitは診断なし・各1回・15 decodeの回帰確認用測定で、初回PPにはcold costを含む。採用判断には反復したSingle PDを使った。PPは変更しておらず、cold Splitの数%の時間差だけでアルゴリズムによる改善/悪化とは判断しない。\n')
rows=[]
for r in memory:
 p=r['peaks']['0'];cnt={x['key']:x for x in r['counters']};get=lambda k,f:cnt.get('tg|'+k+'|pd0',{}).get(f,0);rows.append([r['test'],p['hbm_peak'],p['l3_peak'],get('H2D','bytes'),get('D2H','bytes'),get('launch','count'),get('wait','count')])
w(table(['独立監査run','HBM peak B','L3 peak B','TG H2D B','TG D2H B','TG起動','TG wait'],rows))
w('TinyはPP4096＋TG7、7BはPP2048＋TG7のPD0監査。xpu_malloc/xpu_memcpyで観測した値で、FFN等のhost転送、初回shader uploadも含む。カーネル内部DMAとは別で、監査時のtoken/sは採用しない。旧版もScore/PをL3へ置くため、FDが新たにそのHBMアクセスをゼロにしたわけではない。\n')
w('Tinyの独立監査ではautoの追加HBMは12544byte、追加H2Dは12580byte。新shader4個のcold uploadと192対156byteのmetadata差を含む。TG起動数は1848回減少し、12起動差×22層×7stepと一致した。waitの4回増加は4個の新shaderのload直後のwaitで、通常steady処理でchunkごとに追加waitを入れたものではない。\\n')
w('TG論理workspaceはCとともに増えるが、PPが確保したNativeL3共有poolを再利用する。D128/H32のPP poolは12713984byte、D64は10616832byte。16MiB超過と、外部が14MiB確保した場合のL3不足を試し、旧版へのfallbackと出力ビット一致を両PDで確認した。\n')
w('GQA、FP16/FP32 Q、FP16 KV不変性、stride/view、batch broadcast、端数32-key tile、ALiBi、causal/all-mask/一部all-mask chunk、zero/微小/外れ値、同一pointerのKV更新を試験した。短い生成32 token、長context生成64 token、Single/Layer/Rowのstate restoreを確認。自由生成は分岐後の入力が異なるので、そのKLを固定入力の数値誤差としては使わない。Q4 layout・Attention・SwiGLU・RoPEの既存回帰と契約試験も通過した。\n')
w('## 8. 数値精度とLLM品質\n')
nu=[r for r in units if not r['test'].startswith('profile')];w(f"単体記録{len(nu)}件（同一shapeの候補・再試験を含む）。CPU FP64に対する最大NMSE {max(r['cpu_nmse'] for r in nu):.4g}、最大絶対誤差 {max(r['cpu_max'] for r in nu):.4g}。参照は実入力のFP16/F32値を読みdoubleでAttentionを計算する。誤差は入力/scale/maskにも依存し、最大値だけで通常LLMを評価しない。\n")
w('chunkの局所最大値を使うためP→INT16の量子化が旧版と変わり得る。さらに分母/出力の加算順序が変わる。Qの丸めやK/V scaleを変更した差ではない。1024周期Kを使う対照で最大値を揃え、P量子化差とFP32 reduction差を切り分けた。1 chunk、同じ境界のbalanced/fixed、worker割当だけの変更ではビット一致を要求した。\n')
rows=[]
for model in ['tiny','7b']:
 for batch in [128,1]:
  for mode in ['off','on']:
   r=next(x for x in validation if x['test']==f'corpus-{model}-b{batch}-{mode}');c=fields(next(l for l in r['lines'] if l.startswith('CORPUS ')));rows.append([model,batch,mode,c['scored'],c['perplexity']])
w(table(['model','batch','mode','scored','PPL'],rows))
w('1024入力・ctx256・1020 scored positions。batch128はPPで0.7.2とビット一致。batch1 forced FDは旧版比Tiny約-0.020%、7B約+0.014%のPPL差。固定64とbalanced4の境界が等しい実LLMコーパスは全logitsビット一致。実運用autoは短contextおよび7Bで旧版を使う。\n')
w('PPとTGは既存のQ4 GEMM/GEMV等の経路も異なる。batch1旧版をbatch128参照へ比べた差はFD新規差ではない。同じbatch/input/contextで旧版と比較して増分を評価した。\n')
rows=[]
for r in selected:rows.append([r['pd'],f"{r['nmse']:.3e}",fmt(r['max_abs'],6),f"{r['mean_KL']:.3e}",f"{r['max_KL']:.3e}",f"{r['greedy_equal']}/64",fmt(math.log(r['old_continuation_ppl']),6),fmt(math.log(r['new_continuation_ppl']),6)])
w(table(['最終auto PD','logits NMSE','max abs','平均KL','最大KL','greedy','旧fixture NLL','新fixture NLL'],rows))
w('これは固定teacher64 tokenのNLLで、標準コーパスPPLと別。teacher列の自然な正解継続としての妥当性を評価していないため、ここでは同一fixture上の差分指標として使う。採用版は64位置でgreedy一致したが、他の候補や自由生成には分岐がある。小さいfixtureによる確認で一般的な生成品質を網羅保証するものではない。NaNを任意に含む全入力組合せや、未評価のLLM長contextへ精度を外挿していない。\n')
w('## 9. 最終dispatchと未対応範囲\n')
w('`KUNLUN_FLASH_DECODING=auto/off/on`。unsetはauto。autoはM1/D64/Hq32/Hkv4/batch1、default block1024/TG grouping、interleaved compact KV、F16 mask/ALiBiなし、padded S2048..4352の測定済み形状に限定する。S>4096は固定1024＋head affinity、それ以前はbalanced。7B、短context、未評価shape、PPは0.7.2へ戻る。onは強制実験用で、全shapeの高速化を意味しない。\n')
w('`CHUNKS=1..64`、`CHUNK=256/512/1024`、`FIXED_KEYS=32/64/128/256/512/1024`、`WORK=chunk/head/auto`、`LAYOUT=head/chunk`はKUNLUN_FLASH_DECODING_ prefix。FIXED_KEYSは長さからcountを決め、前二者より優先。明示調整はautoの固定長チューニングを上書きする。legacy/unknown KQ、legacy/unknown Online、narrow KQVは旧FAへ戻して既存設定を尊重する。HBM強制時autoは旧版を使う。\n')
w('M>1のPP、D64/128、32-key整列等の既存制約を維持。非整列/非対応shapeは既存fallback。独立Attention auto/sdnn/cl、長context GQA GEMVのCL fallbackも変更なし。S8192/16384のAPI検証とLLM評価範囲は区別し、autoの上限は測定済み領域に限定した。\n')
w('## 10. 実機運用・残存課題・再現\n')
w('診断専用CLのLM2GM(meta+j)がLMアラインメント違反となり、他プロセス不在確認後、card-wide soft resetを1回実施した。aligned temporary経由へ修正し、両PDの既知正常0.7.2 KQと再試験を通した。復旧直後の一時的なxpu_smi FDでsanityを保留したが再resetはしなかった。引数不足のsanityも修正し、その記録を残した。製品経路でのdevice errorは発生していない。最終stateは両PD RUNNING/no error。incidentのソース・ログ・復旧結果を同梱した。\n')
w('残る主要項目はK/Vのstrided DMAとDS、毎tokenの全KV findmax/range、GQAなしTGの小さいMAC行数、SDNN/CL段階間の待ち。次段階ではKV更新・rollback/state restoreと整合する増分range、1-waveのzero acc転送削減、Score/P ring buffer、double bufferとevent/pipeline、未評価context/modelの拡充を優先する。pipelineにはL1/L2所有権とstream保証の最小実験が必要で、今回実現したと扱わない。\n')
w('ソース、差分、baseline snapshot、shader/image hash、ベンチマーク、入力ID、通常/診断ログ、CSV/JSON、[再現手順](REPRODUCE.ja.md)を保存。SDKとモデルweightは配布へ含めない。過去のソース書換え用scriptsは開発履歴であり、完成treeへ一括再実行しない。\n')
(D/'REPORT.ja.md').write_text(re.sub(r'\n{3,}', '\n\n', '\n'.join(out)));print('REPORT',D/'REPORT.ja.md')


