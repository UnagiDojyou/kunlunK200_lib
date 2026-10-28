# Common Q4 Activation snapshot API (0.7.7)

ホスト API の追加です。Attention は 0.7.6 の実装、共通 weight ABI は v1 のままです。量子化・MAC・join の XPU ソース／引数・数値演算は変更しません。

## API と所有権

`common_q4_prepare(workspace, pd, input, K, M, type, generation, lease)` は、連続した X[M,K] の量子化 snapshot を作ります。`common_q4_matmul_shared(workspace, pd, output, weight, lease, generation)` は、その snapshot を同じ K の複数 weight に使います。N の一致は不要です。共有する scale は Activation の scale であり、各 weight の元 Q4_0 FP16 block scale は各 MAC が従来どおり別々に参照します。共有を使わない既存 `common_q4_matmul` は従来どおりです。

- generation は入力の論理的所有者が発行する非ゼロの 64 bit 値です。入力 pointer は同一性のキーではありません。入力更新・buffer 再利用後は新しい generation が必須です。
- workspace と lease は非コピー型で、workspace は一つの PD に固定されます。新しい prepare または通常 matmul は前の snapshot を失効させます。別 workspace／PD、古い generation は起動前に拒否します。
- 一つの workspace が量子化値と scale の HBM allocation を各一つ持ちます。lease はその内部 storage の共有所有権を持ちますが、古い snapshot の複製は作りません。容量拡張は既存 native_reserve と同じ方式なので、その際だけ旧・新 allocation が一時共存し得ます。定常容量と SDK API peak を区別してください。workspace の解放時に失効し、最後の lease の破棄まで HBM の寿命を保護します。
- 呼出しは従来と同じ runtime_mutex 下で同期的に完了します。入力 owner は prepare 中の書込みを防ぎ、入力に再び変更を加えたら lease を破棄してください。raw SDK 書込みをライブラリが自動追跡する機能ではありません。
- `reset()` はその snapshot のみ失効させます。古い lease の reset が新しい snapshot を失効させることはありません。
- `chunk()` は読み取り専用の借用 view です。workspace の次の変更または破棄後に、保持していた生 pointer を使用してはいけません。`matches()` の確認と実行は API 内で mutex により一体化されます。

## 互換条件と fallback

F32/F16 の連続行のみです。K のパディングは共通 ABI から算出し、従来の IO カーネルを使います。M=1 は従来の二成分 A8、M>1 は従来の Row A8 です。M>128 は従来と同じ 128 行 chunk 列を一度準備し、同じ順序で MAC します。weight layout/version、PD、K/Kp、型、generation と storage owner を検証します。元入力と output/weight の重複、prepare 元入力と同じ workspace の quant/scale storage の重複も起動前に拒否します。

量子化値 allocation の上限は 16 MiB です。容量・型・shape・alignment が対応外なら prepare は false を返します。shared matmul が false を返した場合には出力を書き換えていないため、通常 API の型・shape・非重複条件を満たす元入力を保持した状態で通常 matmul へ戻せます。不正な別名入力をそのまま通常 API へ渡してはいけません。XPU 実行エラーは false に隠しません。既存の fail-fast と停止ラッチを使ってください。

M>128 では全 chunk の量子化値を保持するため、従来の一つの chunk 用 workspace より HBM が増えます。M=1 の確保量は 4*Kp+1024 bytes と scale 64 bytes、M>1 は ceil(M/128)*(128*Kp+1024) bytes と ceil(M/128)*8192 bytes です。上式は要求サイズで、実際の確保は既存 allocator の最小 256 bytes と SDK の alignment に従います。これは weight payload とは別の一時 workspace です。L3 や weight allocation の追加はありません。

## 診断と単体試験

`activation_statistics()` に量子化 chunk／起動回数、準備・再利用回数、拒否と HBM 確保量を追加しました。既存の CommonQ4Statistics は変更していません。`KUNLUN_Q4_ACT_PROFILE=1` のときだけ prepare／quantization のホスト経過時間を測ります。既存の同期を含む値であり、MAC 時間や LLM 時間と重複加算してはいけません。通常性能測定では unset または 0 を使用します。

`k200-q4-activation-bench PD K N M f32|f16 pattern trials consumers` で検証します。pattern 0 は通常、1 は外れ値、2 はゼロ、10 は element alignment のみを満たす入力です。consumers は 2 または 3。旧 API と chunk の定義済み quant/scale 領域、出力を bitwise 比較し、世代、PD、workspace、上書き、破棄、二スレッドでの失効を検証します。未定義の padding bytes は比較に含めません。

例：

```sh
k200-q4-activation-bench 0 4096 4096 128 f32 0 5 3
k200-q4-activation-bench 1 11008 4096 129 f16 0 5 2
```

通常利用のコードは呼出し前の入力 descriptor／generation の検証と所有期間を担当し、GGML 固有の tensor や graph をライブラリへ持ち込まないでください。
