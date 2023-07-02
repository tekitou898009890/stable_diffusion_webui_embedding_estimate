# embedding_estimate

stable diffusion webui extension

[日本語解説](#embedding_estimate_日本語)

## Overview
The process flow of the emphasis syntax () in webui is as follows: 

1. Separate the token part from the emphasis part in () and input the token part to the encoder of the transformer in the CLIP model 3.

3. get a vector from the obtained hidden_states layer and normalize it

4. multiply it by the numerical value specified in the highlighting syntax for each token

The process is as follows.

So, in order to reproduce the vector values through the highlighting syntax with embedding, we need to

1. prepare the target vector output

2. input the embedding to be adjusted to the encoder

3. calculate the loss value from the output of 2 and the target output and adjust

This method was adopted.

## Parameters

When using this method, enter the following parameters and press "estimate!

### steps

The number of steps to be used in t2i,i2i. The number of steps used in t2i,i2i may be the same as the number of steps you normally use.

### text

Prompt used by t2i,i2i. Syntax should be the same as webui's (need to check if it works with other syntax than emphasized syntax).

### layer

The number of tokens to use for embedding.

### late

Learning rate.

### optimizer

Choice of learning adjustment method. Default is Adam.

### loss

Select the loss function. Default is MSELoss.

### learning step

Number of learning steps.

### initial prompt

Additional learning prompt.

The prompt is only converted to embedding (embedding layer) before input to the encoder, so entering the emphasis syntax here will not be emphasized.

option: The number of tokens can be overridden by the number of LAYERS.

If on, the number of tokens actually created for embedding will be the number specified in layers; if off, the number of tokens will be calculated from the text of init_text and saved as the number of tokens.

If the number of tokens exceeds the number of LAYERS, it is rounded down, and if there is a shortage, the shortage is filled with a 0 vector.

The fewer the number of layers, the less learning time is required.


### name

The name under which the embedding will be saved. Can be overwritten by turning on the check button next to it.


## embedding_estimate_日本語

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

### loss

損失関数の選択。デフォルトはMSELoss。

### learning step

学習ステップ数。

### initial prompt

追加学習用のプロンプト入力欄。

プロンプトをencoderに入力する前のembedding(埋め込み層)に変換するだけで、強調構文をここで入力しても強調されない。

オプション：The number of tokens can be overridden by the number of LAYERS.

onにすると実際に作成されるembeddingのトークン数がlayerで指定した数値になる。トークン数がlayer数よりも超過した場合は切り捨て、不足した場合は不足分を0ベクトルで埋める。OFFだとinit_textのテキストから算出されたトークン数で保存される。

layer数が少ない分、学習時間は短く済む。

### name

embeddingを保存する際の名前。隣のチェックボタンをONにすることで上書き保存可能。





