

    仏教的哲学的社会シミュレーション
    (Created: 2021-02-27, Time-stamp: <2022-03-05T12:56:57Z>)


** このファイルについて

「仏教的哲学的社会シミュレーション」については README.md をまずお読み
ください。

このファイルは、「仏教的哲学的社会シミュレーション」に現在含まれるファ
イルについて、簡単な説明をするものです。

なお、GitHub Gist で公開されていたもの(https://gist.github.com/ ではじ
まる URL 下にあるもの)は、少しわかりにくいですが、昔のファイルに遡るこ
とができます。往々にして、初期のファイルのほうが、バグはあるもののわか
りやすいことがあるため、初期のファイルから辿るには便利です。

また、*_2.py などの 2 の数字は 1 からではなく途中からはじまったり途中
が抜けたりする場合がありますが、それはプロジェクトの途中で作ったが、公
開されてない(その必要がないと判断した)ファイルがあるというだけのことで
す。


** このプロジェクトのドキュメント

  * README.md

最初に読んでいただきたいドキュメントで、このプロジェクトを簡単に紹介し
ています。


  * 00_README.txt

このファイルです。このプロジェクトのファイルについて簡単に説明します。


  * 01_IdeaLog.txt

ブレインストーミング的なアイデアのログです。詳しいこのプロジェクトのファ
イルの使い方は、今のところ、こちらを読んでいただくしかありません。ただ、
載っている情報は少し古いものまでです。

最新の情報が知りたければ、(主に JRF のみによってですが) ↓で議論がなさ
れています。

《グローバル共有メモ》  
http://jrockford.s1010.xrea.com/demo/shared_memo.cgi?cmd=log

また、このプロジェクトを解説する電子本が有料で発売されています。是非、
買ってお読みいただければと思います。01_IdeaLog.txt に書いたことが元に
なっていますが、仕様の解説などが書き足され、かなり読みやすくなっていま
す。

《「シミュレーション仏教」の試み - Amazon Kindle》  
https://www.amazon.co.jp/dp/B09TPTYT6Q



** このプロジェクトの小さなプログラム (公開順)

  * test_of_merchant_peasant_ratio.py

商人(merchant) と 農民(peasant) の比率と農民の農地の垂直的分布を一定の
関数に保つアルゴリズムのテスト

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/a8e857bd4377fb800952f193c87ba174


  * test_of_normal_levy_3.py

正規分布+マイナスのレヴィ分布の実験。「株式」「債券」「農地」「大バク
チ」「死蔵」それぞれの分布の形を見てみる。

初期には test_normal_levy_3.py の名前で↓で公開されてました。

https://gist.github.com/JRF-2018/50b29e2d95d067483c3e9465c9b510ad


  * generate_normal_levy_csv.py

正規分布+マイナスのレヴィ分布を使うとき必要な normal_levy_1.0.csv を生
成するプログラム。

初期には test_normal_levy_3.py と同じところで公開。


  * normal_levy_1.0.csv

generate_normal_levy_csv.pyを使って生成したファイル。


  * test_of_income_1.py

主に商業財産から決まる収入のテスト経済シミュレーション。

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/39ed54ff14cb3c79c21f42f7254fa7bc


  * test_of_matching_1.py

不倫のマッチングのシミュレーション。

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/2792da451992dae3e918c72a66ab67b0


  * test_of_reduce_2.py

不倫の終る確率のテスト。

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/dee67ffed33e367bec68cc36e1e663dd


  * test_of_matching_2.py

結婚・不倫・扶養・相続などのマッチングのシミュレーション。

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/6650bc84f238e40826251c400f45f328


  * test_of_increase_1.py

増えていく確率のテスト。

初期には↓で公開されてました。

https://gist.github.com/JRF-2018/aace48ebc44229844b6f75a39a287578


  * test_of_inheritance_1.py

相続のテスト。


  * test_of_inheritance_2.py

相続のテスト。


  * test_of_marriage_1.py

近親婚のテスト。


  * test_of_domination_2.py

支配と災害のテスト。


  * test_of_moving_1.py

転居のテスト。


  * test_of_domination_3.py

支配層の代替わりのテスト。

このプログラムは実行する前に、python test_of_matching_2.py -S -t 1200
を実行し正常終了している必要がある。


  * test_of_mean_amplifier_2.py

平均付近の増幅のテスト。

初期には↓で公開されていた。

https://gist.github.com/JRF-2018/ee1bfe8b3cc676a5389ed5f81c6f544f


  * test_of_mean_amplifier_3.py

平均付近の増幅のテスト。MeanAmplifier のテスト。


  * test_of_mean_amplifier_4.py

平均付近の増幅のテスト。BlockMeanAmplifier のテスト。


  * epub/make_fig_normal_levy.py
  * epub/make_fig_binominal.py

電子本で使うグラフの描画をするプログラム。



** 「プロトタイプ」プロジェクト

  * simbdp1.py
  * simbdp1_*.py

Simulation Buddhism Prototype No.1。「シミュレーション仏教」プロトタイ
プ 1号。test_of_matching_2.py のファイルを分割し、
test_of_merchant_peasant_ratio.py と test_of_income_1.py の成果を取り
入れたもの。


  * simbdp2.py
  * simbdp2/*.py

Simulation Buddhism Prototype No.2。「シミュレーション仏教」プロトタイ
プ 2号。simbdp1.py に支配と災害のモデルを足したもの。
test_of_domination_2.py の成果を取り入れている。


  * simbdp3.py
  * simbdp3/*.py

Simulation Buddhism Prototype No.3。「シミュレーション仏教」プロトタイ
プ 3号。simbdp2.py に僧と犯罪のモデルを足したもの。


  * simbdp3x1.py
  * simbdp3x1/*.py

Simulation Buddhism Prototype No.3 x.1。「シミュレーション仏教」プロト
タイプ 3号x.1。simbdp3.py で MeanAmplifier を使うようにしたもの。
今のところ「x.」は「extended」の略のつもり。



** 統計処理

  * stats_simbdp3/run_simbdp3.sh

複数回 simbdp3.py を実行してログを取るプログラム。

run_simbdp3.sh PREFIX NUM OPTIONS

…と指定する。NUM に回数 OPTIONS に simbdp3.py に渡すオプションを指定
する。PREFIX-01.log ... といったファイルができる。


  * stats_simbdp3/run_simbdp3x1.sh

simbdp3x1.py 用の run_simbdp3.sh。


  * stats_simbdp3/plot_logs.py

ログの結果をグラフに描くプログラム。run_simbdp3.sh の PREFIX を(複数)
引数として取る。-p AccDeath を付けると累積死亡数のグラフを表示。それ以
外のオプションについてはソースを読むなりしていただきたい。


  * stats_simbdp3/make_logs.sh
  * stats_simbdp3/make_figs.sh

ログを作り、グラフを描く際の PREFIX とパラメータの参考にするために作っ
た。著者は実際にはこれらをちょこちょこっといじったものを使って毎度ログ
を取り、グラフを描いている。これらを使って一気に make することは時間が
かかり過ぎることもあり本当のところあまり考えていない。



----
(This document is mainly written in Japanese/UTF8.)
