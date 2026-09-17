"""建立日文查字用的字典：data/dict/jmdict.db（JMdict 英文義項 + JLPT + 重音）。

用法（在專案根目錄，python 用 conda env subtitle 裡的）：
  python -s tools/build_dict_ja.py [--download]

程式第一次打開時會自動建（佇列的 dict 任務），設定頁也有按鈕；這個指令給開發者手動建，跟程式用同一份建置程式
（app/dict_build.py）。來源檔放在 data/dict/src/，沒有的話會自動下載（可以續傳）；加 --download 會強制重新下載 JMdict（每天更新）。
  JMdict_e.gz        https://www.edrdg.org/pub/Nihongo/JMdict_e.gz            EDRDG，CC BY-SA 4.0
  jlpt_n1~5.csv      stephenmk/yomitan-jlpt-vocab original_data（固定 commit） Stephen Kraus，CC BY-SA 4.0（原始資料 Jonathan Waller，CC BY）
  accents.txt        mifunetoshiro/kanjium data/source_files/raw（固定 commit） Kanjium，Uros O.，CC BY-SA 4.0（要標示作者）
去活用規則 tools/yomitan/yomitan_ja_transforms.json（Yomitan，GPL-3.0-or-later）會一起複製到 data/dict/。
建好的 jmdict.db 含上面三份資料，一樣是 CC BY-SA 4.0；出處和標示方式見專案根目錄的 THIRD_PARTY_NOTICES.md。

先寫到 jmdict.db.tmp，完成才替換；伺服器開著時如果換不掉，新檔會留著，下次打開程式時自動換上（或關掉伺服器再跑一次）。
約 15 秒，產生約 90 MB。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import dict_build  # noqa: E402  跟程式同一個位置（VS_DATA_DIR 改過也一樣）


def main(argv=None):
    dict_build.cli("ja", argv)


if __name__ == "__main__":
    main()
