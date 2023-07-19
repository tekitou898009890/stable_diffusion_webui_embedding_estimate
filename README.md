# embedding_estimate

stable diffusion webui extension

Create embdding from prompts (emphasis syntax also supported).Negative_TI creation eliminates the need to prepare images and tags.

#### The prompts for emphatic syntax can be reproduced to a certain extent.

|(bad quality:1.4)|embedding output|
|---|---|
|![00028-778087371](https://github.com/tekitou898009890/stable_diffusion_webui_embedding_estimate/assets/124915480/403d5507-5204-461c-affd-d0fcc33b1861)|![00027-778087371](https://github.com/tekitou898009890/stable_diffusion_webui_embedding_estimate/assets/124915480/fb60c633-f544-427a-aad5-8a74dd1582b5)|

<br>

#### Setting example

![Screenshot 2023-07-19 at 12-37-29 Stable Diffusion](https://github.com/tekitou898009890/stable_diffusion_webui_embedding_estimate/assets/124915480/24e61cee-ff1b-451f-a339-ddbe9dc5f433)

<br>

[日本語解説](#embedding_estimate_日本語)

## Overview
The process flow of the emphasis syntax () in webui is as follows

1. Separate the token part from the emphasis part in () and input the token part to the encoder of the transformer in the CLIP model 3.

2. get a vector from the obtained hidden_states layer and normalize it

3. multiply it by the numerical value specified in the highlighting syntax for each token

The process is as follows.

So, in order to reproduce the vector values through the highlighting syntax with embedding, we need to

1. prepare the target vector output

2. input the embedding to be adjusted to encoder or U-NET

3. calculate the loss value from the output of 2 and the target output and adjust

This method was adopted. (Incidentally, encoder could not reproduce the value of the emphasis syntax, so if you use emphasis syntax, it is recommended to use U-NET.)

## Parameters

To use this method, enter the following parameters and press the "estimate!

### prompt type

Decide whether to use Prompt or Negative Prompt.

### steps

The number of steps to use with t2i and i2i. It may be the same number of steps as you normally use.

### text

The prompt to use with t2i,i2i. Syntax should be the same as webui's (need to check if it works with other syntax than emphasized syntax).

### Training Part

Select the part to be trained.

#### transformer

Reproduction of emphatic syntax: x
Learning speed: 0

The vector of weights of 77 tokens x 768 output from the encoder part of CLIP is compared and adjusted.

#### U-NET

Learning speed: TI-like

Since the emphasis syntax of stable diffusion webui multiplies the transformer output by a specified value, it is difficult to reproduce the weight vector of the emphasis syntax by learning only with the transformer (even if 100 is multiplied, the value will be normalized and reduced to a constant value).

Therefore, we arranged the Textual Inversion method to change the input image in TI to the image generated from the text prompt, and directly learn the noise reduction of the text prompt in U-NET.

Since U-NET also has a normalization layer, we expected to be able to learn how to reduce the noise in the textual prompts by embedding the highlighted values (the generated image is likely to have many disturbance factors such as CFG Scale and sampler when directly learning how to reduce the noise).

The webui standard textual inversion used latent diffusion, but this did not give the desired results, perhaps because the noise results were too far apart, so we used the k_diffusion method for de-noising.

### token

Number of tokens used for embedding.

### late

Learning rate.

### optimizer

Choice of learning adjustment method. Default is Adam.

### loss

Select the loss function. Default is MSELoss.

### learning step

Number of learning steps.


### initial prompt

Additional learning prompt input field.

The prompt is only converted to embedding (embedding layer) before input to the encoder, and the emphasis syntax is not emphasized if you enter it here.
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

2. 調整するためのembeddingをencoderあるいはU-NETに入力

3. 2の出力と目標出力からloss値を算出し調整する

この方法を採用した。（ちなみにencoderでは強調構文の値は再現できなかったので、強調構文を使用する場合はU-NETの使用が望まれる。）

## パラメータ

使う際は下記のパラメータを入力してから「estimate！」ボタンを押す。

### prompt type

Promptで使うかNegative Promptで使うか決める。

### steps

t2i,i2iで使うstep数。普段使うstep数でいいのかもしれない(要検証)。

### text

t2i,i2iで使うプロンプト。構文はwebuiのものがそのまま使えるはず(強調構文以外の構文ではどうなるか要検証)

### Training Part

学習させる箇所を選択する。

#### transformer

強調構文の再現:×
学習速度:〇

CLIPのencoder部分に入力し出力された77トークンx768個の重みのベクトルを比較調整する。

#### U-NET

学習速度:TI並

stable diffusion webui の強調構文はtransformer出力に指定した値を掛けているので、transformerだけの学習では強調構文の重みベクトルの再現は難しい(100を掛けたとしても正規化されて値が縮小され値が一定値に収まってしまう)。

そこで、Textual Inversionの手法をアレンジし、TIにおける入力画像をtextプロンプトから生成された画像に変更し、U-NETにおけるtextプロンプトのノイズの引き方を直接学習させる。

U-NETもまた正規化層があるため強調された値も縮小されて結果が出るため、強調構文におけるノイズの引き方をembeddingで学習できると予測した(生成画像はノイズの引き方を直接学習する際にCFG Scale,samplerなど多くの外乱要素を伴うと思われる)。

webui標準のtextual inversionではlatent diffusionを使用していたが、これではノイズ結果がかけ離れていたのか、思った通りの結果にならなかったので、デノイズにはk_diffusionの手法を採用した。

### token

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

オプション：Selects the number of tokens used for embedding, either the specified number or the number set in init_text. 

埋め込みに使われるトークンの数を、指定された数か init_text で設定された数のどちらかで選択する。

トークン数がlayer数よりも超過した場合は切り捨て、不足した場合は不足分を0ベクトルで埋める。init_textだとテキストから算出されたトークン数で保存される。

※layer数が少ないとその分学習時間は短く済む。

### name

embeddingを保存する際の名前。隣のチェックボタンをONにすることで上書き保存可能。



