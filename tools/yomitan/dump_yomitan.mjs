// 把 Yomitan 的日文去活用規則匯出成 JSON，給 app/dict_ja.py 用。
// 規則來源：Yomitan（https://github.com/yomidevs/yomitan）的 ext/js/language/ja/japanese-transforms.js，
// Copyright (C) 2024-2026 Yomitan Authors，授權 GPL-3.0-or-later。匯出的 JSON 也是 GPL-3.0-or-later，
// 開頭的 _about 寫著出處；來源檔和版本見 tools/yomitan/README.md。
// 匯出時改了什麼：正規表示式存成字串（pattern），整個詞的規則先算好還原結果（to），其他欄位照原檔。
// 只有要更新規則時才需要跑：
//   1. 把 yomitan 的 ext/js/language/language-transforms.js 和 ext/js/language/ja/japanese-transforms.js
//      下載到這個資料夾的 ext/js/language/ 底下（目前放的是 2026-09 的版本）
//   2. node tools/yomitan/dump_yomitan.mjs
//   3. python -s tools/build_dict_ja.py（會把新的 JSON 複製到 data/dict/）
import {japaneseTransforms} from './ext/js/language/ja/japanese-transforms.js';
import {writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = {
  _about: {
    source: 'Yomitan https://github.com/yomidevs/yomitan ext/js/language/ja/japanese-transforms.js',
    copyright: 'Copyright (C) 2024-2026 Yomitan Authors',
    license: 'GPL-3.0-or-later',
    changes: '用 tools/yomitan/dump_yomitan.mjs 匯出：正規表示式存成字串（pattern），整個詞的規則先算好還原結果（to），其他欄位照原檔。',
  },
  conditions: {},
  transforms: {},
};
for (const [k, c] of Object.entries(japaneseTransforms.conditions)) {
  out.conditions[k] = {isDictionaryForm: c.isDictionaryForm, subConditions: c.subConditions || null};
}
for (const [id, t] of Object.entries(japaneseTransforms.transforms)) {
  out.transforms[id] = {
    name: t.name,
    description: t.description || '',
    rules: t.rules.map((r) => ({
      type: r.type,
      pattern: r.isInflected.source,
      // suffix 規則可以直接取還原字串；wholeWord 用 deinflect() 算出來
      to: r.type === 'suffix' ? r.deinflected : r.deinflect(''),
      in: r.conditionsIn,
      out: r.conditionsOut,
    })),
  };
}
writeFileSync(join(here, 'yomitan_ja_transforms.json'), JSON.stringify(out));
console.log(Object.keys(out.transforms).length, 'transforms',
  Object.values(out.transforms).reduce((a, t) => a + t.rules.length, 0), 'rules');
