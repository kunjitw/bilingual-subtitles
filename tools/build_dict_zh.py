"""建立中文單字頁用的辭典：data/dict/zh_dict.db（教育部《重編國語辭典修訂本》，萌典 JSON）。

用法（在專案根目錄，python 用 conda env subtitle 裡的）：
  python -s tools/build_dict_zh.py

設定頁有「加入中文」按鈕（佇列的 dict 任務）；這個指令給開發者手動建，跟程式用同一份建置程式（app/dict_build.py）。
來源檔 data/dict/src/dict-revised.json.xz，沒有的話自動下載（g0v/moedict-data 固定 commit，核對 sha256）。
授權 CC BY-ND 3.0 TW：畫面要標示出處，釋義文字照原文顯示、不改寫。
約 20 秒，產生約 25 MB。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import dict_build  # noqa: E402  跟程式同一個位置（VS_DATA_DIR 改過也一樣）


def main(argv=None):
    dict_build.cli("zh", argv)


if __name__ == "__main__":
    main()
