# Yomitan 日文去活用規則

這個資料夾放的是日文查字用的去活用規則（把「食べられなかった」拆回「食べる」）。規則來自 Yomitan。

- 原專案：Yomitan，https://github.com/yomidevs/yomitan
- 著作權：Copyright (C) 2024-2026 Yomitan Authors
- 授權：GPL-3.0-or-later（全文見專案根目錄的 LICENSE）

## 檔案

| 檔案 | 是什麼 | 跟原檔的差別 |
|---|---|---|
| ext/js/language/language-transforms.js | Yomitan 原檔 | 沒有改，保留原本的 GPL 標頭 |
| ext/js/language/ja/japanese-transforms.js | Yomitan 原檔 | 沒有改，保留原本的 GPL 標頭 |
| dump_yomitan.mjs | 本專案寫的匯出腳本 | 讀上面兩個原檔，匯出成 JSON |
| yomitan_ja_transforms.json | 從原檔匯出的規則表 | 正規表示式存成字串（pattern），整個詞的規則先算好還原結果（to），其他欄位照原檔；開頭多一個 _about 寫出處 |
| package.json | 讓 node 用 ES module 載入原檔 | 本專案加的 |

兩個原檔是 2026-09-17 跟 Yomitan 的 master 分支比對過，內容完全相同（當時 japanese-transforms.js 最後一次修改是 2026-08-18 的 commit 77e2004289）。

app/dict_ja.py 的去活用程式（_rules()、deinflect()）是照 Yomitan 的 ext/js/language/language-transformer.js 改寫成 Python 的，一樣是 GPL-3.0-or-later，改了哪些寫在那個檔案開頭。

## 更新規則

平常不用跑，只有想換成 Yomitan 新版規則時才需要：

1. 從 Yomitan 下載 ext/js/language/language-transforms.js 和 ext/js/language/ja/japanese-transforms.js，照原本的路徑蓋掉這個資料夾裡的兩個檔案。
2. 在專案根目錄執行 `node tools/yomitan/dump_yomitan.mjs`，會重新產生 yomitan_ja_transforms.json。
3. 重建日文字典（tools/build_dict_ja.py），新的 JSON 會一起複製到 data/dict/。
4. 跑 tests/test_dict_ja.py 確認查字結果沒有變怪。
5. 更新上面「兩個原檔是哪一天比對過」那一行。

如果 Yomitan 新版的規則出現真正的正規表示式（不只是字尾或整個詞），app/dict_ja.py 載入時會報錯，要先改 deinflect()。
