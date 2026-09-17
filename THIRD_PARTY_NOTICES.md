# 第三方授權與出處

本專案自己的程式碼採用 GPL-3.0-or-later，全文見 [LICENSE](LICENSE)。

這份文件列出本專案用到的別人的程式、字典資料、模型和執行檔，寫明從哪裡來、是什麼授權、本專案怎麼用，以及需要標示什麼。專案的授權只管程式碼，字典資料和模型各有自己的條件，有些禁止商用，使用前請看清楚。

授權都是 2026-09-17 照官方授權頁、模型卡、套件資料查證的。查不到或看不懂的寫「不確定」，沒有用猜的。上游隨時可能改授權，以原作者頁面為準。

「怎麼用」的意思：

- repo 內附：檔案就放在這個 repo 裡。
- 建字典時下載：在自己電腦上跑 tools/build_dict_*.py 時才從原網址下載來源檔，建好的資料庫留在自己電腦的 data/dict，repo 不放。
- 安裝時下載：裝環境時從 PyPI 或原作者下載，repo 不放。
- 模型管理下載：在設定頁的模型管理自己按下載，從原作者的 Hugging Face 或 GitHub 下載，本專案不轉存，也不預設下載非商用或授權不明的模型。
- 自己安裝：本專案不附也不自動下載，要自己裝好。
- 完整包附帶：一般的 zip 不附，只有完整包（BilingualSubtitles-<版本>-full.zip）在 offline 資料夾附上原檔，見第 6 節。

打包計畫（安裝時自動下載 Python、ffmpeg、llama.cpp）還沒做完，做完後「怎麼用」這一欄要跟著改。

## 1. repo 內附

| 名稱 | 網址 | 授權 | 本專案怎麼用 | 標示和注意 |
|---|---|---|---|---|
| Yomitan 日文去活用規則 | https://github.com/yomidevs/yomitan | GPL-3.0-or-later，Copyright (C) 2024-2026 Yomitan Authors | repo 內附，查日文單字時把活用形還原成辭書形 | 原檔保留 GPL 標頭；衍生的檔案開頭寫了出處和改了什麼，細節見 tools/yomitan/README.md |

從 Yomitan 來的檔案：

- tools/yomitan/ext/js/language/language-transforms.js：Yomitan 原檔，沒有改。
- tools/yomitan/ext/js/language/ja/japanese-transforms.js：Yomitan 原檔，沒有改。
- tools/yomitan/yomitan_ja_transforms.json：用 tools/yomitan/dump_yomitan.mjs 從上面的原檔匯出的規則表，開頭的 _about 寫著出處。
- tools/yomitan/dump_yomitan.mjs：本專案寫的匯出腳本，匯出的內容是 Yomitan 的規則。
- app/dict_ja.py 的 _rules()、deinflect()：照 Yomitan 的 ext/js/language/language-transformer.js 改寫成 Python；scan() 的查詞順序也照 Yomitan 的做法。
- web/vocab.js 的 CHAIN_ZH：鍵是 Yomitan 規則裡的活用名稱，中文是本專案自己寫的。

## 2. 字典資料

字典資料都是建字典時下載，repo 不放。自己把建好的 data/dict 資料庫分享給別人時，要照下表的授權條件。

| 名稱 | 網址 | 授權 | 本專案怎麼用 | 標示和注意 |
|---|---|---|---|---|
| JMdict（英文版 JMdict_e） | 來源檔 http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz ，專案頁 https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project | CC BY-SA 4.0，著作權屬於 James William Breen 和 Electronic Dictionary Research and Development Group（EDRDG），授權頁 https://www.edrdg.org/edrdg/licence.html | 建字典時下載，tools/build_dict_ja.py 建成 data/dict/jmdict.db，查日文單字 | 文件、網站要寫明用了 JMdict、來源是 EDRDG，並附授權頁連結（設定頁的字典區有寫）。授權要求要有定期更新資料的方法：tools/build_dict_ja.py --download 會重新下載最新版。用 Hy-MT2 把英文義項翻成中文存起來的解釋算改作，分享時一樣是 CC BY-SA 4.0。授權頁還寫到「提供查字功能的網站要在每個顯示字典內容的畫面標示」，本專案是在自己電腦上跑的程式，這條適不適用：不確定 |
| JLPT 詞表（yomitan-jlpt-vocab） | https://github.com/stephenmk/yomitan-jlpt-vocab | CC BY-SA 4.0（Stephen Kraus）；原始資料是 Jonathan Waller 的 JLPT Resources（http://www.tanos.co.uk/jlpt/ ），CC BY | 建字典時下載 original_data 的 n1 到 n5 的 csv，標日文單字的 JLPT 參考等級 | 標示：「JLPT 詞表：Stephen Kraus（yomitan-jlpt-vocab），CC BY-SA 4.0；原始資料 Jonathan Waller（tanos.co.uk），CC BY」 |
| Kanjium 重音資料 | https://github.com/mifunetoshiro/kanjium | CC BY-SA 4.0（Uros O.）；資料大部分源自 EDRDG 的 EDICT、KANJIDIC，照 EDRDG 授權 | 建字典時下載 data/source_files/raw/accents.txt，顯示日文單字重音 | 作者要求的標示文字（原文照抄）：「The pitch accent notation, verb particle data, phonetics, homonyms and other additions or modifications to EDICT, KANJIDIC or KRADFILE were provided by Uros O. through his free database.」設定頁寫的是「Kanjium 重音（Uros O.，CC BY-SA 4.0）」 |
| ECDICT | https://github.com/skywind3000/ECDICT | MIT，Copyright (c) 2025 Linwei | 建字典時下載 stardict.7z，用 OpenCC 轉成台灣繁體，建成 data/dict/en_zh.db，查英文單字 | 分享建好的資料庫時要附 MIT 授權全文。ECDICT 是從很多來源整理出來的，那些來源各自的授權：不確定 |
| CEFR-J Wordlist 1.5 | 官方 https://www.cefr-j.org/download.html ，本專案從 https://github.com/openlanguageprofiles/olp-en-cefrj 下載 | 研究和商業使用都免費，條件是正確引用；著作權屬於東京外國語大學投野由紀夫研究室 | 建字典時下載 cefrj-vocabulary-profile-1.5.csv，標英文單字 A1 到 B2 等級 | 引用句（原文照抄，設定頁有寫）：「The CEFR-J Wordlist Version 1.5. Compiled by Yukio Tono, Tokyo University of Foreign Studies. Retrieved from http://www.cefr-j.org/download.html on 1/20/2020.」官網現在是 1.6 版，本專案用的是 Open Language Profiles 轉存的 1.5 版，所以寫 1.5 |
| Octanove Vocabulary Profile C1/C2 1.0 | https://github.com/openlanguageprofiles/olp-en-cefrj | CC BY-SA 4.0（Octanove Labs，http://www.octanove.com/ ） | 建字典時下載 octanove-vocabulary-profile-c1c2-1.0.csv，標英文單字 C1、C2 等級 | 標示：「Octanove Vocabulary Profile C1/C2（Octanove Labs），CC BY-SA 4.0」 |
| wordfreq | https://github.com/rspeer/wordfreq | 程式 Apache-2.0；內附的詞頻資料 CC BY-SA 4.0 | 安裝時下載（Python 套件），英文單字的常用程度 | 標示：「wordfreq（Robyn Speer），詞頻資料 CC BY-SA 4.0」。資料裡的 SUBTLEX 詞表要標示作者（Marc Brysbaert 等），並讓人知道 SUBTLEX 是免費資料；OpenSubtitles 的資料要標示 OpenSubtitles；Google Books Ngrams 希望標示來源。學術引用：Robyn Speer. (2022). rspeer/wordfreq: v3.0 (v3.0.2). Zenodo. https://doi.org/10.5281/zenodo.7199437 |
| 教育部《重編國語辭典修訂本》（萌典 JSON 格式） | 辭典 https://dict.revised.moe.edu.tw/ ，JSON https://github.com/g0v/moedict-data ，公眾授權說明 https://language.moe.gov.tw/001/Upload/Files/site_content/M0001/respub/index.html | CC BY-ND 3.0 TW（姓名標示、禁止改作），著作權屬於中華民國教育部；萌典轉換格式的編輯著作權由 kcwu 以 CC0 釋出 | 建字典時下載 dict-revised.json.xz，建成 data/dict/zh_dict.db，單字頁的中文解釋 | 官方指定的標示寫法：「中華民國教育部（Ministry of Education, R.O.C.）。《重編國語辭典修訂本》（版本編號：臺灣學術網路第六版）網址：http://dict.revised.moe.edu.tw/」，版本編號照 moedict-data 說明寫的「中華民國110年11月臺灣學術網路第六版」。詞目、部首、筆畫、字形、音讀、釋義都不能修改，也不能轉成簡體字；不要分享用模型改寫過的內容。官方的公眾授權使用說明（PDF）要求使用者「無論再散布與否，都必須完整保留本使用說明」，本專案沒有附上這份說明，需不需要附：不確定。單字頁列表只顯示釋義的第一句（app/vocab_zh.py），這樣算不算修改：不確定 |

## 3. 模型

模型都是模型管理下載，repo 不放。設定頁每個模型列的規格後面會顯示授權，非商用和授權不明的用醒目的顏色標出來（授權文字寫在 app/config.py 的 MODEL_CATALOG）。

| 名稱 | 網址 | 授權 | 標示和注意 |
|---|---|---|---|
| Qwen3-ASR-1.7B | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | Apache-2.0 | |
| Qwen3-ForcedAligner-0.6B | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B | Apache-2.0 | |
| Whisper large-v3 | https://huggingface.co/openai/whisper-large-v3 | Apache-2.0 | |
| anime-whisper | https://huggingface.co/litagin/anime-whisper | 模型卡寫 MIT；設定頁標「非商用」 | 以 kotoba-whisper-v2.0（Apache-2.0，https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 ）微調。訓練資料 Galgame_Speech_ASR_16kHz（https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz ，GPL-3.0）轉載了原資料集 OOPPEENN/Galgame_Dataset 的規定：用這份資料訓練出來的任何模型都不得用於商業行為，而且要開源。模型卡的 MIT 和資料集的規定哪個優先：不確定，本專案保守當成非商用 |
| tsqyomi（v4） | https://huggingface.co/tsukumijima/tsqyomi-models | MIT，Copyright (c) 2026 Aivis Project | |
| Hy-MT2-7B（GGUF） | https://huggingface.co/tencent/Hy-MT2-7B-GGUF | Apache-2.0，Copyright (C) 2026 Tencent | 設定頁可以選 Q8_0、Q6_K、Q5_K_M、Q4_K_M。官方沒有 Q5_K_M，這個版本從 https://huggingface.co/unsloth/Hy-MT2-7B-GGUF 下載（unsloth 量化，檔案裡標的授權是 apache-2.0） |
| Hy-MT2-1.8B（GGUF） | https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF | Apache-2.0，Copyright (C) 2026 Tencent | |
| Sakura-14B-Qwen2.5-v1.0（GGUF） | https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF | CC BY-NC-SA 4.0，非商用 | SakuraLLM（https://github.com/SakuraLLM/SakuraLLM ）寫明所有 Sakura 模型和衍生模型禁止任何形式的商用；公開發布用它翻的譯文時，要明顯標出是機器翻譯和用的模型版本 |
| Sakura-GalTransl-7B v3.7（GGUF） | https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7 | CC BY-NC-SA 4.0，非商用 | 模型卡舉的禁止例子：提供付費翻譯介面、做要付費才能拿到的補丁、商用翻譯。發布譯文的要求同 Sakura-14B |
| CKIP bert-base-chinese-ws、bert-base-chinese-pos | https://huggingface.co/ckiplab/bert-base-chinese-ws ，https://huggingface.co/ckiplab/bert-base-chinese-pos | GPL-3.0 | 也可以用 tools/download_ckip.py 下載 |
| BS-RoFormer ep_317 sdr 12.9755（viperx） | 權重 https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models ，設定檔 https://github.com/nomadkaraoke/python-audio-separator （model-configs）或同一個 UVR 模型庫，模型清單 download_checks.json 來自 https://github.com/TRvlvr/application_data | 授權不明，設定頁標「授權不明」 | 作者 viperx 和放模型的 TRvlvr/model_repo 都沒有公開授權，沒找到允許商用的說明。Ultimate Vocal Remover（https://github.com/Anjok07/ultimatevocalremovergui ）的 README 請使用它模型的第三方程式標示 UVR 和它的開發者；這個模型算不算 UVR 自己的模型：不確定 |

## 4. 執行檔和工具

| 名稱 | 網址 | 授權 | 本專案怎麼用 | 標示和注意 |
|---|---|---|---|---|
| llama.cpp（llama-server） | https://github.com/ggml-org/llama.cpp | MIT，Copyright (c) 2023-2026 The ggml authors | 自己安裝，放在 bin/llama.cpp（作者現在用 build 10985 的 Windows CUDA 版），打包計畫改成安裝時下載；翻譯模型用它載入 | 自己轉存時要附 LICENSE。Windows 版附的 libomp.dll 是 LLVM OpenMP，Apache-2.0 WITH LLVM-exception，壓縮檔裡有 LICENSE-LLVM-OpenMP |
| NVIDIA CUDA 執行庫（cudart64_13、cublas64_13、cublasLt64_13） | https://docs.nvidia.com/cuda/eula/index.html | NVIDIA CUDA Toolkit EULA | 跟 llama.cpp 的 CUDA 版一起放在 bin/llama.cpp | EULA 附件 A 把 Windows 的 cudart、cublas、cublasLt 列為可再散布，條件包括：程式要有 SDK 以外的主要功能、散布條款要跟 EULA 一致。自己轉存前先讀完 EULA |
| FFmpeg | https://ffmpeg.org/legal.html | 看建置版本。作者現在用的 gyan.dev 7.1.1 full build 是 GPL-3.0-or-later（--enable-gpl --enable-version3）；打包計畫的候選 BtbN n8.1 gpl-shared（https://github.com/BtbN/FFmpeg-Builds ）也是 GPL | 自己安裝（預設找 C:\Program Files\FFMPEG\bin，可以用 VS_FFMPEG、VS_FFPROBE 指定），打包計畫改成安裝時下載；本專案只用命令列呼叫 | 自己鏡像或轉存 GPL 版時，要一起提供對應原始碼的取得方式 |
| 7-Zip | https://www.7-zip.org/license.txt | GNU LGPL 為主，部分程式碼是 BSD 3-clause、BSD 2-clause，另有 unRAR 授權限制 | 自己安裝，tools/build_dict_en.py 用它解開 stardict.7z；打包計畫改用 py7zr | 本專案不附 7-Zip |
| Node.js | https://nodejs.org/ | MIT | 只有開發者要更新 Yomitan 規則、跑 tools/yomitan/dump_yomitan.mjs 時才需要，一般使用不用裝 | |

打包計畫會用到、還沒做完的項目（打包定案時要再查一次）：

| 名稱 | 網址 | 授權 | 標示和注意 |
|---|---|---|---|
| uv | https://github.com/astral-sh/uv | MIT OR Apache-2.0 | |
| Python（uv 下載的 python-build-standalone） | https://github.com/astral-sh/python-build-standalone | CPython 本體是 PSF License | 裡面一起打包的其他函式庫各自的授權：不確定，打包時照它附的授權檔整理 |
| msvc-runtime（Visual C++ 執行階段 DLL） | https://pypi.org/project/msvc-runtime/ | Microsoft 專有授權（PyPI 標 Proprietary） | 可不可以跟著程式一起轉存、要什麼條件：不確定 |
| deno（yt-dlp 解 YouTube 用） | https://github.com/denoland/deno | MIT | 透過 yt-dlp[default,deno] 安裝 |
| py7zr | https://github.com/miurahr/py7zr | LGPL-2.1-or-later | 取代 7-Zip 解 stardict.7z |

## 5. Python 套件

都是安裝時下載。這裡列本專案直接用到的和要特別注意的，版本是作者環境裡的版本，授權照套件資料（PyPI 或套件內附的授權檔）。這些套件還會帶進很多間接相依，完整清單等打包計畫的 uv.lock 定案後再產生。

| 套件 | 網址 | 授權 | 本專案用來做什麼 |
|---|---|---|---|
| fastapi 0.141.1 | https://github.com/fastapi/fastapi | MIT | 後端網頁伺服器 |
| starlette 1.6.0 | https://github.com/encode/starlette | BSD-3-Clause | fastapi 的底層 |
| uvicorn 0.53.0 | https://github.com/encode/uvicorn | BSD-3-Clause | 啟動伺服器 |
| pydantic 2.13.5 | https://github.com/pydantic/pydantic | MIT | API 資料格式 |
| python-multipart 0.0.32 | https://github.com/Kludex/python-multipart | Apache-2.0 | 上傳檔案 |
| numpy 2.5.3 | https://github.com/numpy/numpy | BSD-3-Clause（含 0BSD、MIT、Zlib、CC0-1.0 的部分） | 音訊處理 |
| soundfile 0.14.0 | https://github.com/bastibe/python-soundfile | BSD-3-Clause；Windows 版內附的 libsndfile 是 LGPL-2.1 | 讀寫音訊 |
| torch 2.14.0（CUDA 13.0 版） | https://github.com/pytorch/pytorch | 套件資料寫 Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause；Windows CUDA 版內附 NVIDIA 的 cuBLAS、cuDNN 等函式庫，照 NVIDIA 的授權 | 語音辨識、對齊、人聲分離、斷詞 |
| torchaudio 2.11.0 | https://github.com/pytorch/audio | BSD（套件資料只寫 BSD License） | silero-vad 需要 |
| transformers 4.57.6 | https://github.com/huggingface/transformers | Apache-2.0 | 載入 Whisper、CKIP 模型 |
| huggingface_hub 0.36.2、hf-xet 1.6.0 | https://github.com/huggingface/huggingface_hub | Apache-2.0 | 下載模型 |
| accelerate 1.12.0、tokenizers 0.22.2、safetensors 0.8.0 | https://github.com/huggingface | Apache-2.0 | 載入模型 |
| qwen-asr 0.0.6 | https://github.com/QwenLM/Qwen3-ASR | Apache-2.0 | Qwen3-ASR 辨識和 Qwen3-ForcedAligner 對齊 |
| silero-vad 6.2.1 | https://github.com/snakers4/silero-vad | MIT（套件內附 VAD 模型） | 找出有人聲的段落 |
| onnxruntime 1.30.0 | https://github.com/microsoft/onnxruntime | MIT | 在 CPU 上跑 tsqyomi |
| jaconv 0.5.0 | https://github.com/ikegami-yukino/jaconv | MIT | 平假名、片假名轉換 |
| fugashi 1.5.2 | https://github.com/polm/fugashi | MIT AND BSD-3-Clause（內含 MeCab） | 日文斷詞 |
| unidic-lite 1.0.8 | https://github.com/polm/unidic-lite | MIT；內附的 UniDic 辭書可以在 GPL、LGPL、BSD 三種授權裡選一種 | 日文斷詞辭書 |
| pyopenjtalk-plus 0.4.1.post9 | https://github.com/tsukumijima/pyopenjtalk-plus | MIT；內附的辭書是 NAIST 和 UniDic Consortium 的 BSD 3-Clause，內附的 HTS 語音 mei 是 CC BY 3.0（名古屋工業大學） | 日文假名（本專案不用語音合成） |
| spacy 3.8.16 | https://github.com/explosion/spaCy | MIT | 英文斷句、詞性、原形 |
| en_core_web_sm 3.8.0 | https://github.com/explosion/spacy-models | MIT；套件內的 LICENSES_SOURCES 寫訓練資料用了 OntoNotes 5（Explosion 取得的商業授權）和 WordNet 3.0（WordNet 3.0 License） | 英文模型 |
| wordfreq 3.1.1 | https://github.com/rspeer/wordfreq | 程式 Apache-2.0，資料 CC BY-SA 4.0（見第 2 節） | 英文單字常用程度 |
| ckip-transformers 0.3.4 | https://github.com/ckiplab/ckip-transformers | GPL-3.0 | 中文斷詞和詞性 |
| opencc-python-reimplemented 0.1.7 | https://github.com/yichen0831/opencc-python | Apache-2.0 | 簡繁轉換 |
| yt-dlp 2026.8.19 | https://github.com/yt-dlp/yt-dlp | Unlicense | 從網址下載影片 |
| yt-dlp-ejs 0.8.0 | https://github.com/yt-dlp/ejs | Unlicense AND MIT AND ISC | yt-dlp 解 YouTube 用（打包計畫才會裝） |
| audio-separator 0.47.0 | https://github.com/nomadkaraoke/python-audio-separator | MIT | 執行 BS-RoFormer 人聲分離；它的 README 請整合 UVR 模型的專案標示 UVR 和開發者 |
| librosa 1.0.0 | https://github.com/librosa/librosa | ISC | qwen-asr、audio-separator 需要 |
| soxr 1.1.0 | https://github.com/dofuuz/python-soxr | LGPL-2.1-or-later | 音訊重新取樣（間接相依） |
| certifi | https://github.com/certifi/python-certifi | MPL-2.0 | HTTPS 憑證（間接相依） |
| tqdm | https://github.com/tqdm/tqdm | MPL-2.0 AND MIT | 進度條（間接相依） |
| psutil 7.2.2 | https://github.com/giampaolo/psutil | BSD-3-Clause | 打包計畫要用 |

要特別注意的兩個間接相依：

- soynlp 0.0.493（qwen-asr 帶進來的，https://github.com/lovit/soynlp ）：PyPI 標 GPLv3，GitHub 上的 LICENSE 是 LGPL-3.0，兩邊不一樣，實際是哪一個：不確定。兩個都跟本專案的 GPL-3.0-or-later 相容。
- diffq-fixed 0.2.4（audio-separator 在 Windows 上一定會裝，https://github.com/JackismyShephard/diffq ）：CC BY-NC 4.0，非商用。本專案用的 BS-RoFormer 不會執行到它，但它會被裝進環境。

## 6. 完整包附帶的檔案

一般的 zip 不附這些檔案，安裝時才從原網址下載。完整包（tools/build_release.py --full）把安裝執行環境時要從 GitHub 下載的檔案（很慢），和第一次打開會自動建的日文、英文字典的原始資料，原樣放在 offline 資料夾，沒有改過。安裝時核對 sha256，對了才用。

offline 資料夾裡另外有：

- files.json：每個檔案的來源網址、大小、sha256。
- README.txt：簡短的授權說明，指到這份文件。
- LICENSE-llama.cpp.txt、LICENSE-ECDICT.txt：MIT 授權全文。原本的壓縮檔裡沒有附（llama.cpp 的 zip 只有 LLVM OpenMP 的授權），MIT 要求轉散布時附上。

| 檔案 | 來源 | 授權 | 轉散布的條件和本專案的做法 |
|---|---|---|---|
| ffmpeg-n8.1.2-50-g1a748fe2cd-win64-gpl-shared-8.1.zip | https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-08-31-13-27 | GPL-3.0-or-later（BtbN 的 gpl 版本，裡面的 x264、x265 等函式庫也是 GPL） | GPL 要求轉散布執行檔時讓人拿得到對應的原始碼。FFmpeg 原始碼：https://github.com/FFmpeg/FFmpeg/commit/1a748fe2cd ；建置腳本和每個相依函式庫用的版本：https://github.com/BtbN/FFmpeg-Builds （同一個 tag）。壓縮檔裡有 LICENSE.txt，offline\README.txt 也寫了這兩個網址。GPL-3.0 第 6 條要求原始碼在散布後持續拿得到，網址放在別人的 GitHub 上，將來被刪掉時算不算做到：不確定，保險的做法是作者自己留一份原始碼 |
| llama-b10985-bin-win-cuda-13.4-x64.zip | https://github.com/ggml-org/llama.cpp/releases/tag/b10985 | MIT，Copyright (c) 2023-2026 The ggml authors；裡面的 libomp.dll 是 Apache-2.0 WITH LLVM-exception | MIT 全文放在 offline\LICENSE-llama.cpp.txt；LLVM OpenMP 的授權在壓縮檔裡（LICENSE-LLVM-OpenMP） |
| cudart-llama-bin-win-cuda-13.4-x64.zip | 同上 | NVIDIA CUDA Toolkit EULA（https://docs.nvidia.com/cuda/eula/index.html ） | 裡面是 cudart64_13、cublas64_13、cublasLt64_13 的 DLL，EULA 附件 A 列為可以隨應用程式散布的檔案。本專案是有自己主要功能的程式，這些檔案只是跟著一起放，原樣不改。EULA 的其他條件（這些檔案只能給本程式用、散布條款要跟 EULA 一致）自己轉存前先讀完 |
| en_core_web_sm-3.8.0-py3-none-any.whl | https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0 （uv.lock 指定的來源，sha256 也照 uv.lock） | MIT，Copyright 2021 ExplosionAI GmbH；訓練資料見第 5 節 | wheel 裡有 MIT 全文（LICENSE）和訓練資料的授權說明（LICENSES_SOURCES），原檔不改 |
| JMdict_e.gz | https://www.edrdg.org/pub/Nihongo/JMdict_e.gz （打包當天的版本，日期寫在 files.json） | CC BY-SA 4.0（EDRDG） | 標示見第 2 節。授權要求要有更新的方法：tools/build_dict_ja.py --download 會下載最新版 |
| jlpt_n1.csv 到 jlpt_n5.csv | https://github.com/stephenmk/yomitan-jlpt-vocab （original_data 的 n1.csv 到 n5.csv，只改了檔名） | CC BY-SA 4.0；原始資料 CC BY | 標示見第 2 節 |
| accents.txt | https://github.com/mifunetoshiro/kanjium | CC BY-SA 4.0 | 作者要求的標示見第 2 節 |
| stardict.7z | https://github.com/skywind3000/ECDICT | MIT，Copyright (c) 2025 Linwei | MIT 全文放在 offline\LICENSE-ECDICT.txt |
| cefrj-vocabulary-profile-1.5.csv | https://github.com/openlanguageprofiles/olp-en-cefrj | 研究和商業使用都免費，條件是正確引用 | 引用句寫在 offline\README.txt 和第 2 節。使用條款只寫「使用」，沒寫能不能轉散布原檔：不確定 |
| octanove-vocabulary-profile-c1c2-1.0.csv | https://github.com/openlanguageprofiles/olp-en-cefrj | CC BY-SA 4.0（Octanove Labs） | 標示見第 2 節 |

完整包不附的：

- 中文辭典的來源 dict-revised.json.xz：要在設定頁按按鈕才建；教育部的公眾授權使用說明要求再散布時完整保留那份說明，本專案沒有附。
- uv、Python、msvc-runtime、torch 和其他 Python 套件（en_core_web_sm 以外）：PyPI、PyTorch 官網、uv 的下載點很快，照常安裝時下載。
- Hugging Face 上的模型：照常下載。
- 人聲分離模型 BS-RoFormer（約 610 MB，放在 GitHub）：授權不明（見第 3 節），第一次打開也不會自動下載。要用時在設定頁自己按下載，GitHub 很慢，可能要很久。
