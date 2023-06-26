# embedding_estimate

stable diffusion webui 拡張

## 概要
webuiにおける強調構文()の処理の流れは

1. ()中のトークン部分と強調部分を分け、トークン部分をCLIPモデルにおけるtransformerのencoderに入力

3. 得られたhidden_statesレイヤーからベクトルを取得して正規化

4. そこに強調構文で指定された数値をトークンごとに掛ける

となっている。

なので、embeddingで強調構文を通したベクトル値を再現するにあたり

1. 目標のベクトル出力を用意する

2. 調整するためのembeddingをencoderに入力

3. 2の出力と目標出力からloss値を算出し調整する

この方法を採用した。

## パラメータ

使う際は下記のパラメータを入力してから「estimate！」ボタンを押す。

### steps

t2i,i2iで使うstep数。普段使うstep数でいいのかもしれない(要検証)。

### text

t2i,i2iで使うプロンプト。構文はwebuiのものがそのまま使えるはず(強調構文以外の構文ではどうなるか要検証)

### layer

埋め込みに使用するトークン数。

### late

学習率。

### optimizer

学習調整手法の選択。デフォルトはAdam。

### learning step

学習ステップ数。

### name

embeddingを保存する際の名前。強制上書き注意(要セーフティロック)。





