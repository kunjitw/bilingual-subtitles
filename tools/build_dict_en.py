"""建立英文查字用的字典：data/dict/en_zh.db（ECDICT 轉台灣繁體 + CEFR 等級）。

用法（在專案根目錄，python 用 conda env subtitle 裡的）：
  python -s tools/build_dict_en.py

程式第一次打開時會自動建（佇列的 dict 任務），設定頁也有按鈕；這個指令給開發者手動建，跟程式用同一份建置程式
（app/dict_build.py）。來源檔放在 data/dict/src/，沒有的話會自動下載（可以續傳，核對 sha256）：
  stardict.7z   skywind3000/ECDICT（固定 commit，ECDICT 完整版，MIT）
                用 py7zr 解壓出 stardict.csv（約 230 MB，建完就刪）；開發環境沒有 py7zr 時改用 C:\\Program Files\\7-Zip\\7z.exe
  cefrj-vocabulary-profile-1.5.csv          CEFR-J 1.5（Tono, TUFS）
  octanove-vocabulary-profile-c1c2-1.0.csv  Octanove C1/C2（CC BY-SA 4.0）

約 3 到 5 分鐘，產生約 250 MB。先寫到 en_zh.db.tmp，完成才替換；伺服器開著時換不掉的話，下次打開程式時自動換上。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import dict_build  # noqa: E402  跟程式同一個位置（VS_DATA_DIR 改過也一樣）


def main(argv=None):
    dict_build.cli("en", argv)


if __name__ == "__main__":
    main()
