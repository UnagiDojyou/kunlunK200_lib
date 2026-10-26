# Kunlun K200 Flash-Decoding 0.7.3 実装・実機評価

**Flash-Decodingを実装し、実機検証を完了した。** 最終autoのTinyLlama TG@4096は、PD0で7.3%高速化、PD1で3.7%高速化。7Bでは改善を確認できなかったため、autoは0.7.2 FlashAttentionを使う。PPは既存経路を維持した。単体APIの大きい改善率をLLM全体へ外挿しない方針で採用範囲を限定した。

## 1. 比較基準・環境・保存

作業前の未コミット変更を保存して0.7.2をコミットした。libraryは `d8bd87f56ae95872287c4f4a030b7ac7e68dd7b9`、llama.cppは `8d62617cb1d267d9283d5fcffe8e66c57820dfcc`。作業branchは `codex/k200-flash-decoding-20261026`。凍結した `build/flash-kq-lib` / `build/flash-kq-llama` を上書きせず、新版は `build/flash-decode-lib` / `build/flash-decode-llama` に構築した。

実機は192.168.50.112のK200/KL1、XRE4.33、XTDK2.7.98.2、表示clock900MHz。各PDはSDNN4基、CL4 cluster×16 core、FP32演算8 lane、L3は16MiB、HBMは8064MiB。各SDNNのL1D512KiB、L1W2MiB、L1E256KiB、各L2 bank128KiBを前提に、DMA一要求128KiBの制約も維持した。

TinyLlama1.1B Q4_0とLlama2 7B Q4_0、batch/ubatch128、CPU thread6。既存SDNN KQ/KQV、ordered Online Softmax、KV writer、Q4、RoPE、SwiGLUの旧shader52個は、両ビルドで0.7.2とSHA256一致。SwiGLU=CPU等、Attention以外の条件も固定した。通常測定ではTRACE、PROFILE、LD_PRELOAD監査、stateチェックを無効にした。

## 2. 従来TGとFlash-Decodingの並列性の違い

旧版はKV blockごとにKQ→Online Softmax→KQVを逐次実行する。KQは少ないKV headに対して既にkey範囲を分割し、4基のSDNNを使っていた。新方式はchunkごとの統計・出力を分離し、KQVとCLにもchunk方向のworkを加える。単にKQのkey分割数を増やす方式ではない。

| HK | 方式 | KQ active SDNN/起動 | KQV active SDNN/起動 | Online active core/起動 | core起動数 |
|---|---|---|---|---|---|
| 1 | 0.7.2 | 4,4,4,4 | 1,1,1,1 | 32,32,32,32 | 13 |
| 1 | chunk | 4 | 4 | 64 | 4 |
| 4 | 0.7.2 | 4,4,4,4 | 4,4,4,4 | 32,32,32,32 | 13 |
| 4 | chunk | 4 | 4 | 64 | 4 |
| 32 | 0.7.2 | 4,4,4,4,4 | 4,4,4,4,4 | 32,32,32,32,32 | 16 |
| 32 | chunk | 4 | 4 | 64 | 4 |

Hq32、D128、HK1/4はS4096・4 chunk、HK32はS4352・固定1024の5 chunk。activeはMAC命令または処理行数を記録したworker数で、PMU利用率%ではない。旧版もCLの4 clusterを使うが各cluster8 core、新版は最大16 coreを使う。HK4/32では旧版でも全SDNNが稼働し、追加のSDNN並列性があると解釈してはならない。HK1のKQVは1基→4基となる。

## 3. アルゴリズム・カーネル・共有メモリ

共通cast/gather、K/V findmax、Q rangeはAPIごとに1回だけ実行する。その後、全chunkのKQを1起動、全chunkのOnline Softmaxを1起動、KQVを1起動にまとめ、最後にCL reductionを起動する。1-waveでは計4起動。chunkごとに既存FA APIを呼び出す構造ではない。段階間はdefault stream上で逐次実行し、SDNN/CLの全面的な同時pipelineは実装していない。並列性は同じ起動内のhead×chunkのwork分配から得る。

各chunkは最大値m、指数和l、未正規化accと最後のKQV partialを持つ。次waveでは前のpartialを取り込み、最大値の更新に応じてaccとlを再スケーリングする。短いchunkが後のwaveを処理しない場合、最後の状態・partialを保持する。最終reductionはFP32で次を計算する。

```text
M = max_i(m_i)
w_i = exp(m_i - M)
L = sum_i(w_i * l_i)
O = sum_i(w_i * (acc_i + last_partial_i))
Y = O / L
```

全mask chunkを除外し、全mask行はゼロ出力。-Inf同士の差をexpへ渡さない。native expと既存ordered-sumを維持し、近似expは導入していない。試作で外れていた既存FP16 maskのzero/-Inf wordショートカットを引き継いだ版も、旧試作LLM出力と一致した。

QのFP16丸め、SDNN DMAのINT16変換、K/V scaleの算出方式は変更していない。SDK findmaxの4部分最大値を全てreduceする。SDNNは従来のINT16 MAC＋FP32可視出力、chunk統合はCL FP32。今回MAC内部のIEEE FP32累積を新たに実証したものではない。

Score/Pは同じL3領域を再利用する。Score/Pはchunk-major、partial/acc/statsはhead-major。reductionは最大8 chunkを連続ロードし、状態をchunkごとに細かく読む方式も比較用に残した。SDNNは2D DMAで直接head-majorへ書き込み、別transpose kernelは使わない。各SDNNのcontroller core0が自身のL1D/L1W/L2を所有し、競合する別streamは使用しない。QはL1D、KはL1Wで複数32-key tileに再利用する0.7.2の構造を保つ。

workspaceは `B * Hq * C * (4 * block + 8 * D + 32)` byte。16MiB超過・L3確保失敗・未対応条件は旧版へ戻る。chunk数を32-key tile数以下へclampし空chunkを生成しない。metadataはprivate192-byte構造で公開API変更なし。同じビルドのlibraryとshaderを組で使用する。

| kernel | 生成assemblyのstack frame調整byte | binary byte |
|---|---|---|
| kunlun-flash-decode-kq.device | -512 | 3284 |
| kunlun-flash-decode-online-head.device | -9696 | 4556 |
| kunlun-flash-decode-reduce-head.device | -11008 | 1512 |
| kunlun-flash-decode-pv-head-affinity.device | -384 | 1988 |

CL LMとstackは共用。compile-native.shのframe上限16256byte検査を全kernelで通した。L1WにはtaskのK/V範囲を置くので、chunk数分のL1容量を同時に要求しない。D128・1024 keyは256KiBで、128KiBずつ2回DMAする。

## 4. chunk長・worker配置・性能を左右する要因

balancedは32-key tileを均等分割し、当初はchunk数を2の冪へ切り上げた。実TGではKV長がpaddingされるため、prefix2048はS2304、prefix4096はS4352になる。S4352を8 chunkにすると544 keyずつになり、DMA/DS/MAC設定の回数が増えた。3/5 chunkも比較し、固定1024×4＋256なら旧版のMAC/DMA形状を保てることを検証した。

| 比較series | model | mode | TG token/s | 旧版比% | 反復min..max |
|---|---|---|---|---|---|
| 初期balanced | tiny | on | 17.373 | 0.10 | 17.290..18.296 |
| 初期balanced | tiny | on:5 | 18.429 | 6.19 | 17.430..18.633 |
| 初期balanced | tiny | off | 17.355 | 0.00 | 16.496..17.577 |
| 初期balanced | 7b | on | 3.592 | -4.69 | 3.589..3.610 |
| 初期balanced | 7b | on:5 | 3.664 | -2.75 | 3.652..3.721 |
| 初期balanced | 7b | off | 3.768 | 0.00 | 3.757..3.781 |
| 割当比較 | tiny | on:5 | 18.490 | 6.43 | 18.229..18.634 |
| 割当比較 | tiny | head:5 | 18.546 | 6.75 | 17.719..18.651 |
| 割当比較 | tiny | off | 17.373 | 0.00 | 16.582..17.607 |
| 割当比較 | 7b | on:5 | 3.695 | -2.56 | 3.681..3.713 |
| 割当比較 | 7b | head:5 | 3.751 | -1.09 | 3.748..3.798 |
| 割当比較 | 7b | off | 3.792 | 0.00 | 3.751..3.793 |
| 固定長比較 | tiny | fixed | 18.015 | 4.58 | 17.377..18.543 |
| 固定長比較 | tiny | fixhead | 18.804 | 9.16 | 17.855..18.805 |
| 固定長比較 | tiny | off | 17.226 | 0.00 | 16.705..17.486 |
| 固定長比較 | 7b | fixed | 3.671 | -2.35 | 3.671..3.685 |
| 固定長比較 | 7b | fixhead | 3.743 | -0.45 | 3.727..3.757 |
| 固定長比較 | 7b | off | 3.760 | 0.00 | 3.759..3.763 |

onはbalanced試作、on:5はbalanced5、head:5は同じ境界のhead affinity、fixed/fixheadは固定1024のchunk/head割当。この探索seriesは当時のdefaultを記録しており、最終autoとは区別する。同じ境界でworkerの割当だけを変えた64件のAPI試験と両モデル/両PDのLLM試験は出力ビット一致。配置変更の効果は量子化差と分離できた。

実KV Cacheに近いinterleaved配置も測定した。dense tensorの単体結果だけではLLM性能を予測できない。下表は同じ論理値・同じF32 maskでKV配置だけを変えた対照試験。

| D | pattern | KV配置 | 0.7.2 API ms | 固定head API ms |
|---|---|---|---|---|
| 128 | 0 | dense | 2.402 | 2.304 |
| 128 | 11 | interleaved | 4.072 | 3.963 |
| 64 | 0 | dense | 0.431 | 0.345 |
| 64 | 11 | interleaved | 0.515 | 0.431 |

配置によるDMA区間の違いは確認できるが、HBM bankやPMU counterから競合原因を直接証明したものではない。7Bは既に4基のSDNNを使い、KV転送量を減らさないchunk分割だけでは全体の利益が出なかった。

## 5. API・段階別・DMAの比較

| PD | S | 旧API ms | 最終auto API ms | 倍率 | CPU NMSE |
|---|---|---|---|---|---|
| 0 | 2048 | 0.314 | 0.263 | 1.19 | 3.61e-10 |
| 0 | 2304 | 0.372 | 0.287 | 1.30 | 1.10e-09 |
| 0 | 4096 | 0.596 | 0.495 | 1.20 | 1.27e-09 |
| 0 | 4352 | 0.655 | 0.506 | 1.30 | 1.41e-09 |
| 1 | 2048 | 0.308 | 0.242 | 1.28 | 3.61e-10 |
| 1 | 2304 | 0.366 | 0.297 | 1.23 | 1.10e-09 |
| 1 | 4096 | 0.589 | 0.505 | 1.17 | 1.27e-09 |
| 1 | 4352 | 0.643 | 0.500 | 1.29 | 1.41e-09 |

APIはcast/gather、findmax/range、metadata、workspace、起動と完了待ちを含む。初回cold callを除く中央値。通常timingに追加同期を混ぜていない。

| 診断case | 方式 | KQ ms | Online ms | KQV ms | 最終CL ms |
|---|---|---|---|---|---|
| profile-p0-hk1-c4 | legacy | 0.154 | 0.301 | 0.102 | 0.004 |
| profile-p0-hk1-c4 | decode | 0.053 | 0.158 | 0.027 | 0.009 |
| profile-fixed-p0 | legacy | 1.480 | 0.333 | 1.583 | 0.004 |
| profile-fixed-p0 | decode | 1.475 | 0.187 | 1.560 | 0.010 |

get_clockの900MHz換算。SDNNは各起動の4基中最大、CLは各起動の64 worker中最大を加算した値。fence区間には命令発行・制御・待ちが含まれ、純DMA時間や利用率%ではない。diagnosticは追加waitや計測値転送を含むため、この和を通常API時間やtoken/sに置き換えない。PROFILE時のFLASH_UNITも通常性能表から除外した。詳細はprofiles-final.json/csvのphase_ms（Q DMA、Q DS、K/V DMA、DS、MAC、EW、RS、DMAOUT、other）。

| case | 方式 | KQ Q-DMA | KQ K-DMA | KQ MAC命令 | KQV P-DMA | KQV V-DMA | KQV MAC命令 |
|---|---|---|---|---|---|---|---|
| profile-p0-hk1-c4 | legacy | 512 | 16 | 256 | 4 | 8 | 32 |
| profile-p0-hk1-c4 | decode | 128 | 8 | 256 | 4 | 8 | 32 |
| profile-fixed-p0 | legacy | 160 | 288 | 8704 | 160 | 288 | 1280 |
| profile-fixed-p0 | decode | 160 | 288 | 8704 | 160 | 288 | 1280 |

S4096/HK1ではQ転送が128KiB→32KiBへ減り、Kの論理転送量1MiBは同じ。S4352/D128/HK32では旧版と固定5のK/V DMAは各288回、Q-DMA160回、KQV MAC1280命令。balanced8はソース上の計数でそれぞれ512回、256回、2048命令になる（byte量は同じでも要求数が増える）。K/VのMAC読出し量は両方式で同じ（S4096は64MiB、S4352は68MiB）で、findmaxも同じKVを読む。L3 Score/Pの基本読書き量はHq*S*16byteでS4096/Hq32なら2MiB。

## 6. PP/TGの通常推論と最終auto

以下は0.7.2 offと初期balanced試作onの通常smoke比較。warmup1＋2反復、63 decode。FD以外の設定を固定し、診断・stateチェック・LD_PRELOADなし。PPのcode pathは共通で、差には測定変動を含む。

| model | PD | prefix | 旧PP t/s | 試作PP t/s | 旧TG t/s | 試作TG t/s | TG差% |
|---|---|---|---|---|---|---|---|
| 7b | 0 | 4096 | 62.36 | 62.20 | 3.806 | 3.646 | -4.21 |
| tiny | 0 | 4096 | 219.92 | 223.72 | 17.652 | 18.377 | 4.11 |
| 7b | 0 | 128 | 71.29 | 70.55 | 7.188 | 7.225 | 0.51 |
| 7b | 0 | 512 | 70.22 | 70.76 | 6.557 | 6.439 | -1.79 |
| 7b | 0 | 1024 | 69.54 | 69.36 | 5.925 | 5.933 | 0.12 |
| 7b | 0 | 2048 | 67.16 | 67.21 | 4.992 | 4.958 | -0.69 |
| 7b | 1 | 128 | 70.86 | 71.39 | 7.129 | 7.345 | 3.03 |
| 7b | 1 | 512 | 70.48 | 71.19 | 6.550 | 6.547 | -0.05 |
| 7b | 1 | 1024 | 69.09 | 69.22 | 5.966 | 6.063 | 1.62 |
| 7b | 1 | 2048 | 66.77 | 66.74 | 5.064 | 4.940 | -2.44 |
| tiny | 0 | 128 | 284.91 | 286.63 | 22.833 | 22.809 | -0.10 |
| tiny | 0 | 512 | 280.84 | 281.13 | 22.237 | 21.994 | -1.09 |
| tiny | 0 | 1024 | 270.09 | 269.66 | 21.154 | 21.893 | 3.50 |
| tiny | 0 | 2048 | 251.41 | 250.87 | 19.968 | 20.561 | 2.97 |
| tiny | 1 | 128 | 284.99 | 285.26 | 23.071 | 23.032 | -0.17 |
| tiny | 1 | 512 | 290.71 | 286.10 | 21.555 | 21.820 | 1.23 |
| tiny | 1 | 1024 | 273.66 | 279.28 | 21.150 | 21.413 | 1.24 |
| tiny | 1 | 2048 | 258.71 | 255.41 | 18.817 | 19.766 | 5.04 |

最終採用版をauto指定で再確認した結果：

| PD | 0.7.2 TG@4096 | auto TG@4096 | 改善% | auto min..max | 反復 |
|---|---|---|---|---|---|
| 0 | 17.363 | 18.627 | 7.28 | 18.501..18.754 | 2 |
| 1 | 17.675 | 18.335 | 3.73 | 17.901..18.768 | 2 |

共通prefillの後、未使用領域もclearして同じKV stateを復元しmode順を循環させる交互比較。clear/state転送・dump・checksumはTG timer外で、測定中は通常のllama_process()だけを呼ぶ。最終再確認はwarmup1＋2反復、探索は3反復。各mode内の全反復でlogits checksum一致。最終autoの出力は採用した固定head指定の出力ともビット一致した。

4k fixtureはprompt2048.idsを2回連結したもの。context確保は4608で、自然長文の標準PPLではない。LLMのTGはprefixから63 token進むため、単体の固定S時間とは異なる。APIはS8192/16384も検証したが、LLM評価はTG@4096まで。

## 7. Layer/Row Split・memory・互換性

| model | split | prefix | 旧PP | auto PP | 旧TG | auto TG |
|---|---|---|---|---|---|---|
| 7b | layer | 128 | 70.32 | 70.30 | 7.092 | 7.080 |
| 7b | layer | 512 | 70.00 | 70.06 | 6.462 | 6.430 |
| 7b | layer | 2048 | 66.30 | 66.30 | 4.987 | 4.974 |
| 7b | row | 128 | 46.09 | 46.12 | 4.981 | 5.006 |
| 7b | row | 512 | 45.93 | 45.99 | 4.687 | 4.673 |
| 7b | row | 2048 | 44.54 | 44.41 | 3.853 | 3.824 |
| tiny | layer | 128 | 282.03 | 279.85 | 22.494 | 22.472 |
| tiny | layer | 512 | 278.87 | 279.53 | 21.536 | 21.986 |
| tiny | layer | 2048 | 252.54 | 247.57 | 19.236 | 20.335 |
| tiny | row | 128 | 178.75 | 169.66 | 12.981 | 13.197 |
| tiny | row | 512 | 170.04 | 170.55 | 12.938 | 12.820 |
| tiny | row | 2048 | 164.27 | 158.20 | 11.501 | 12.638 |

Splitは診断なし・各1回・15 decodeの回帰確認用測定で、初回PPにはcold costを含む。採用判断には反復したSingle PDを使った。PPは変更しておらず、cold Splitの数%の時間差だけでアルゴリズムによる改善/悪化とは判断しない。

| 独立監査run | HBM peak B | L3 peak B | TG H2D B | TG D2H B | TG起動 | TG wait |
|---|---|---|---|---|---|---|
| final-memory-7b-auto | 5894381568 | 12713984 | 10475784 | 20077568 | 15043 | 14222 |
| final-memory-7b-off | 5894381568 | 12713984 | 10475784 | 20077568 | 15043 | 14222 |
| final-memory-tiny-auto | 735132928 | 10616832 | 3977708 | 7115584 | 9429 | 10430 |
| final-memory-tiny-off | 735120384 | 10616832 | 3965128 | 7115584 | 11277 | 10426 |

TinyはPP4096＋TG7、7BはPP2048＋TG7のPD0監査。xpu_malloc/xpu_memcpyで観測した値で、FFN等のhost転送、初回shader uploadも含む。カーネル内部DMAとは別で、監査時のtoken/sは採用しない。旧版もScore/PをL3へ置くため、FDが新たにそのHBMアクセスをゼロにしたわけではない。

Tinyの独立監査ではautoの追加HBMは12544byte、追加H2Dは12580byte。新shader4個のcold uploadと192対156byteのmetadata差を含む。TG起動数は1848回減少し、12起動差×22層×7stepと一致した。waitの4回増加は4個の新shaderのload直後のwaitで、通常steady処理でchunkごとに追加waitを入れたものではない。\n

TG論理workspaceはCとともに増えるが、PPが確保したNativeL3共有poolを再利用する。D128/H32のPP poolは12713984byte、D64は10616832byte。16MiB超過と、外部が14MiB確保した場合のL3不足を試し、旧版へのfallbackと出力ビット一致を両PDで確認した。

GQA、FP16/FP32 Q、FP16 KV不変性、stride/view、batch broadcast、端数32-key tile、ALiBi、causal/all-mask/一部all-mask chunk、zero/微小/外れ値、同一pointerのKV更新を試験した。短い生成32 token、長context生成64 token、Single/Layer/Rowのstate restoreを確認。自由生成は分岐後の入力が異なるので、そのKLを固定入力の数値誤差としては使わない。Q4 layout・Attention・SwiGLU・RoPEの既存回帰と契約試験も通過した。

## 8. 数値精度とLLM品質

単体記録600件（同一shapeの候補・再試験を含む）。CPU FP64に対する最大NMSE 2.698e-06、最大絶対誤差 5.143e-05。参照は実入力のFP16/F32値を読みdoubleでAttentionを計算する。誤差は入力/scale/maskにも依存し、最大値だけで通常LLMを評価しない。

chunkの局所最大値を使うためP→INT16の量子化が旧版と変わり得る。さらに分母/出力の加算順序が変わる。Qの丸めやK/V scaleを変更した差ではない。1024周期Kを使う対照で最大値を揃え、P量子化差とFP32 reduction差を切り分けた。1 chunk、同じ境界のbalanced/fixed、worker割当だけの変更ではビット一致を要求した。

| model | batch | mode | scored | PPL |
|---|---|---|---|---|
| tiny | 128 | off | 1020 | 11.1246713081 |
| tiny | 128 | on | 1020 | 11.1246713081 |
| tiny | 1 | off | 1020 | 10.8131438999 |
| tiny | 1 | on | 1020 | 10.8109871464 |
| 7b | 128 | off | 1020 | 8.34594649553 |
| 7b | 128 | on | 1020 | 8.34594649553 |
| 7b | 1 | off | 1020 | 8.21526526404 |
| 7b | 1 | on | 1020 | 8.21643046992 |

1024入力・ctx256・1020 scored positions。batch128はPPで0.7.2とビット一致。batch1 forced FDは旧版比Tiny約-0.020%、7B約+0.014%のPPL差。固定64とbalanced4の境界が等しい実LLMコーパスは全logitsビット一致。実運用autoは短contextおよび7Bで旧版を使う。

PPとTGは既存のQ4 GEMM/GEMV等の経路も異なる。batch1旧版をbatch128参照へ比べた差はFD新規差ではない。同じbatch/input/contextで旧版と比較して増分を評価した。

| 最終auto PD | logits NMSE | max abs | 平均KL | 最大KL | greedy | 旧fixture NLL | 新fixture NLL |
|---|---|---|---|---|---|---|---|
| 0 | 4.785e-06 | 0.072625 | 4.018e-05 | 1.002e-04 | 64/64 | 7.327399 | 7.326317 |
| 1 | 4.785e-06 | 0.072625 | 4.018e-05 | 1.002e-04 | 64/64 | 7.327399 | 7.326317 |

これは固定teacher64 tokenのNLLで、標準コーパスPPLと別。teacher列の自然な正解継続としての妥当性を評価していないため、ここでは同一fixture上の差分指標として使う。採用版は64位置でgreedy一致したが、他の候補や自由生成には分岐がある。小さいfixtureによる確認で一般的な生成品質を網羅保証するものではない。NaNを任意に含む全入力組合せや、未評価のLLM長contextへ精度を外挿していない。

## 9. 最終dispatchと未対応範囲

`KUNLUN_FLASH_DECODING=auto/off/on`。unsetはauto。autoはM1/D64/Hq32/Hkv4/batch1、default block1024/TG grouping、interleaved compact KV、F16 mask/ALiBiなし、padded S2048..4352の測定済み形状に限定する。S>4096は固定1024＋head affinity、それ以前はbalanced。7B、短context、未評価shape、PPは0.7.2へ戻る。onは強制実験用で、全shapeの高速化を意味しない。

`CHUNKS=1..64`、`CHUNK=256/512/1024`、`FIXED_KEYS=32/64/128/256/512/1024`、`WORK=chunk/head/auto`、`LAYOUT=head/chunk`はKUNLUN_FLASH_DECODING_ prefix。FIXED_KEYSは長さからcountを決め、前二者より優先。明示調整はautoの固定長チューニングを上書きする。legacy/unknown KQ、legacy/unknown Online、narrow KQVは旧FAへ戻して既存設定を尊重する。HBM強制時autoは旧版を使う。

M>1のPP、D64/128、32-key整列等の既存制約を維持。非整列/非対応shapeは既存fallback。独立Attention auto/sdnn/cl、長context GQA GEMVのCL fallbackも変更なし。S8192/16384のAPI検証とLLM評価範囲は区別し、autoの上限は測定済み領域に限定した。

## 10. 実機運用・残存課題・再現

診断専用CLのLM2GM(meta+j)がLMアラインメント違反となり、他プロセス不在確認後、card-wide soft resetを1回実施した。aligned temporary経由へ修正し、両PDの既知正常0.7.2 KQと再試験を通した。復旧直後の一時的なxpu_smi FDでsanityを保留したが再resetはしなかった。引数不足のsanityも修正し、その記録を残した。製品経路でのdevice errorは発生していない。最終stateは両PD RUNNING/no error。incidentのソース・ログ・復旧結果を同梱した。

残る主要項目はK/Vのstrided DMAとDS、毎tokenの全KV findmax/range、GQAなしTGの小さいMAC行数、SDNN/CL段階間の待ち。次段階ではKV更新・rollback/state restoreと整合する増分range、1-waveのzero acc転送削減、Score/P ring buffer、double bufferとevent/pipeline、未評価context/modelの拡充を優先する。pipelineにはL1/L2所有権とstream保証の最小実験が必要で、今回実現したと扱わない。

ソース、差分、baseline snapshot、shader/image hash、ベンチマーク、入力ID、通常/診断ログ、CSV/JSON、[再現手順](REPRODUCE.ja.md)を保存。SDKとモデルweightは配布へ含めない。過去のソース書換え用scriptsは開発履歴であり、完成treeへ一括再実行しない。

