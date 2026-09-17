# Bilingual Subtitles 開發說明

給改程式的人看。一般使用說明在 README.md。

以前叫 video-subtitle（影片字幕）。程式內部的識別碼（例如 /api/meta 的 app）沿用舊名，不要改。

## 開發模式

專案根目錄有 `.dev` 檔案時，start.bat 不跑安裝流程，改用 conda env `subtitle`（Python 3.12），不動系統的 Python。

- 預設找 `%USERPROFILE%\.conda\envs\subtitle\python.exe`。
- env 放在別的地方時，把環境變數 `VS_PYTHON` 設成那個 python.exe 的完整路徑。
- `.dev` 被 .gitignore 擋掉，發佈的 zip 也不會有。

打包發佈用的 zip：`python -s tools/build_release.py --out <專案外的資料夾>`

完整包（附 GitHub 上很慢的 ffmpeg、llama.cpp、cudart、en_core_web_sm 和日文、英文字典的來源，放在 zip 的 `offline\`）：加 `--full`，輸出 `BilingualSubtitles-<版本>-full.zip`。檔案從 `private\bundle-cache\` 拿，沒有就下載。安裝時 sha256 對才用，不對或沒有照常下載。offline 裡的執行環境檔齊全時，precheck 不檢查 GitHub。

## 安裝流程

沒有 `.dev` 時，start.bat 會把執行環境裝在程式資料夾的 `runtime\`：

- uv 和 Python 3.12
- Python 套件（照 uv.lock，torch 約 1.9 GB）
- ffmpeg、llama.cpp、微軟 VC++ 執行階段檔案

不改系統設定、登錄檔或 PATH。安裝器在 `launcher\`。路徑有中文等非英文字元時 precheck 會擋下，因為 qwen-asr 用的 nagisa 讀不到中文路徑的檔案。

## 第一次打開的自動安裝

伺服器啟動時判斷是不是新使用者（沒有影片、任務紀錄、模型、字典），是的話照顯示卡挑模型，自動排模型下載和日文、英文字典的建立（`app/setup.py`、`app/dict_build.py`）。網頁只負責顯示進度（`web/app.js` 的「首次自動安裝」）。判斷過一次就記在資料庫，之後不會再自動裝。

- 專案根目錄有 `.dev` 時不會自動裝。要在開發資料夾測試，設環境變數 `VS_SETUP_AUTO=1`；`VS_SETUP_AUTO=0` 一律不裝。
- `VS_DICT_SOURCE_BASE=http://127.0.0.1:port`：字典的原始資料改從這個網址下載，測試時指到本機的假伺服器。
- 介面測試：`python -s tests/ui_check.py <輸出資料夾> --setup`，在頁面裡換成假的安裝狀態，桌機和手機各截一輪圖，不會改到伺服器的資料。

## 刪除與寫檔的安全

- 所有刪檔都經過 `app/safepath.py`。刪之前確認目標真的在 `data/media`、`data/work`、`data/subs`、`data/thumbs`、`data/proxy`、`data/dict`、`models` 裡面，而且不是捷徑，不符合就不刪並寫進 log。
- 設定的 cookies.txt 只會被複製來用，不會被改寫。
- 別的網站的網頁送來的改資料請求會被拒絕。
- 新的影片不能用電腦上的檔案路徑加入（區網裝置也能連進來）。以前用本機路徑加入的影片照常播放，移除時不會刪到原始檔。

## 模型與顯存

| 用途 | 模型 |
|---|---|
| 辨識：中文、英文、日文一般 | Qwen3-ASR-1.7B |
| 辨識：動畫、耳語 | anime-whisper |
| 時間軸 | Qwen3-ForcedAligner-0.6B |
| 翻譯：英文、日文一般 | Hy-MT2-7B（Q8_0、Q6_K 等版本） |
| 翻譯：動畫、耳語 | Sakura-14B Q6_K |
| 其他可下載 | Whisper large-v3、Hy-MT2-1.8B、Sakura-GalTransl-7B |

- 模型一定整個放進顯存，不會分到 CPU。
- 辨識、對齊、時間軸檢查的模型放在常駐子程序（`app/speech_worker.py`，由 `app/speech.py` 管理）。顯存放得下就同時載入，下一部影片不用重新載入；放不下就在同一個程序裡輪流載入。
- 翻譯由 llama-server 載入（`-ngl all --fit off`，放不下直接報錯），跨任務沿用。放不下時先卸載語音模型；轉字幕時翻譯模型放不下也會先關掉它。
- 佇列沒有顯卡任務滿 2 分鐘（`config.GPU_IDLE_RELEASE_S`），語音模型和翻譯模型一起釋放。
- 「釋放顯卡」按鈕：佇列沒有任務就直接釋放；還有任務時先問，確認後暫停佇列，正在跑的任務回到佇列，按「繼續佇列」從中斷的地方接著做。暫停狀態重開程式也會保留。
- 環境變數 `VS_SPEECH_WORKER=0` 改回每一步開一個子程序、跑完就結束的舊做法。

建議在 NVIDIA 控制面板 → 管理 3D 設定 → 程式設定，把 python.exe 和 llama-server.exe 的「CUDA - 系統記憶體備援原則」設成「不偏好系統記憶體備援」。

## 資料夾

- `data/library.db`：播放列表、字幕軌、佇列、設定
- `data/subs/`：字幕（JSON，含假名標註）。`*.bak.json` 是修正時間軸前留下的備份
- `data/media/`：上傳與網址下載的影片
- `data/work/`：處理中的暫存與檢查點，完成後自動刪除。`speech_worker.log` 是常駐語音程序的紀錄，重開程序時接在後面寫，超過 2 MB 換名成 `speech_worker.old.log`
- `data/cache/`：yt-dlp、Hugging Face 等套件的下載快取，可以整個刪掉
- `data/app.log`：錯誤紀錄
- `private/`：實驗腳本、研究資料、規劃文件，不上傳

## 測試

單元測試一支一支跑，例如 `python -s tests/test_cues.py`。不用先下載模型，資料庫和字幕都放系統暫存資料夾，不會動到 data：

- `test_cues.py`、`test_health.py`、`test_safety.py`（刪檔與寫檔安全）、`test_models.py`、`test_model_download.py`、`test_glosses.py`、`test_vocab_status.py`、`test_retry_guard.py`、`test_launcher.py`、`test_moved_folder.py`（程式資料夾搬家後的路徑修正）
- `test_speech.py`：常駐語音程序，用 `fake_speech_backend.py` 的假模型，不需要顯卡
- `test_portability.py`：要先裝好 ffmpeg
- `test_instance.py`：會開測試用的伺服器（預設 port 8821 到 8839）
- `test_dict_ja.py`：要先建好日文字典（`python -s tools/build_dict_ja.py`）

其他：

- `fake_jmdict.py` 不是測試，是 test_glosses 和 test_vocab_status 共用的迷你日文字典
- 介面自動測試（要開著伺服器和 Chrome）：`ui_check.py`、`pair_check.py`
- 顯卡能不能跑模型：`smoke_asr.py`、`smoke_llm.py`
- DLL 相依檢查：`dll_check.py`
