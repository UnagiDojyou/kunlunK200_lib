# Q6_K出力層：直接CL GEMVの試作評価

2026-10-30 JST。結論：圧縮Q6_KをK200上で直接計算する試作は精度検査に合格したが、CPUより約42〜44倍遅いため採用しない。正式library API/GGML dispatchへの統合とCPU fallback解消は未達。遅いkernelを本番へ追加せず、SwiGLU ON・Large PP OFFの既存経路を維持した。

## 原因・実装

現在のGGML MUL_MAT支持型はF32/F16/Q4_0で、library WeightTypeにもQ6_Kがない。両GGUFのoutput.weightを直接解析し、type14=Q6_K、7BはK4096/N32000、TinyはK2048/N32000を確認。詳細はmodels.json。通常の最終token logitsではM1を優先すべき形状。CPU配置のためhidden stateがCPUへ渡り、Q6_K×Q8_KのCPU dotを処理してlogitsを作る。XPU化してもsampling用の最終logits D2Hは残る。今回新しい全runtime診断は実施せず、既存の約3.6ms/tokenとgraph配置調査を利用した。

llama.cppのdequantize_row_q6_Kを基準に、ql128/qh64/scales16/d2=210byteのlayoutとsigned6（−32..31）を実装。ブロックを256byteへpaddingし、aligned DMAでCL16×4clusterへ供給。量子化値・FP16 d・signed scaleは変更せず、重み再量子化なし。F32 Activationを直接乗算、8lane FP32累積。1つの重みHBM allocation、workspace/L3なし。初回pack/uploadと定常時間を分離した。各CLのlocal配列は約2.4KBで、生成assemblyのframe上限検査を通した。SDKの通常DMAのみ使用し、experimental compiler optionやfence削除は使っていない。

## 実機結果

PD0、M1、CPU6thread。実モデルの全Q6_K出力重み、同じsin入力、warmup後各3回の中央値。upload・kernel loadを除いた完了待ち込み。CPUは現行ggml-cpuのquantize_row_q8_K＋ggml_vec_dot_q6_K_q8_K。XPUはF32入力なのでCPUのA8量子化との差も分けて評価した。

| K×N | CPU ms | XPU ms | 比較 | 対独立F64参照NMSE | 対CPU Q8_K NMSE |
| --- | --- | --- | --- | --- | --- |
| 4096×32000 | 3.567357 | 148.552644 | 41.64倍低速 | 1.72399432912e-13 | 1.00383169779e-05 |
| 2048×32000 | 1.693616 | 74.491526 | 43.98倍低速 | 8.09331663965e-14 | 8.9699397786e-06 |

XPU最小〜最大：7B148.503〜148.599ms、Tiny74.354〜74.678ms。独立参照はGGMLと同じF32復元重みに対するF64積和。最大絶対誤差7B7.15e-7、Tiny5.36e-7。参照絶対値>0.01での最大相対誤差5.65e-5/2.82e-5。全32000出力finiteかつsentinelから更新、前後guard不変、ゼロ入力で全出力0。bitwiseではない主因はFP32累積順序。CPU Q8_KとのNMSE約1e-5はActivation量子化差を含み、Q6_Kの重み再量子化ではない。

初回CPU参照はggml-cpuのFP16 lookup初期化が不足し無効だったため破棄し、CPU backend初期化・参照finite/nonzero検査を加えた最終値のみ採用。device異常なし。

## メモリと費用

| model | 元Q6_K payload | padded resident | upload＋入力転送 ms |
| --- | --- | --- | --- |
| 7B | 107,520,000byte | 131,072,000byte | 19.620 |
| Tiny | 53,760,000byte | 65,536,000byte | 9.928 |

保存効率は6.5625→8bpw。F32/F16重みの恒常展開やHBM二重常駐なし。CPU上の元重みとpackは比較用。出力128KB＋guard、入力16/8KB、コード16KB未満。7Bでは32k×16block×2の約102万DMA呼出し、論理転送量はweight131MB＋Activation524MB/呼出しとなる（実HBM帯域の測定値ではない）。bit展開・小DMA・SIMD供給を行うCL方式全体が遅い。段階別device-clock測定は追加しておらず、どれか1要因を支配的と断定しない。

## 採否・未実施

単体でCPUに勝てないため、ユーザー指定の手順に従いGGMLへの接続前に打ち切った。LLMの新経路PP512/TG128、logits/greedy、転送/graph削減、PD1/Split回帰は未実施。LLM高速化率を捏造せず未測定とする。正式API・opt-in変数・GGML対応宣言は追加していない。既存Q6_K CPU fallback、Q4 ABI/Lease/Attention/SwiGLU/Large PPを変更せず、実験sourceだけを明確に分離して残した。

次の実装候補はSDNNのsigned W8 MACを使い、Q6_Kをtile単位でL1W用W8へ復元する方式。K16ごとのd×scale補正、Activation二成分、FP32累積が必要で、既存W4カーネルへQ6をそのまま渡せない。lossless block-scale扱いとDMA/DS所有権を確認した別SD kernelが必要。load時F16/F32常駐は容量増と精度変更を伴うため今回は採用しなかった。CL版の小設定変更だけで42倍を埋めることは見込まず、深追いしていない。

## 再現・安全性

sourceはlibrary experiments/q6-output。compile-native.shの新しいq6 prefixはframe検査を追加するだけで、production CMakeにkernelを登録していない。build-run.pyのg++/SDK設定とrunner metadataを配布。今後sourceを再buildする場合はexperiments/q6-output/kunlun-q6-probe.xpuを指定する。実行は既存thermal guard/abort latch/両PD監視/利用者検査のrunnerを使う。driver/firmware/clock/OS変更・resetなし。両PD RUNNING/no error、errtaskは0.7.12品質評価終了時と同じ。SDK/models/binariesをGitへ含めない。push/PR/main/master反映なし。
