// 競艇日和のレースページを開いて、JavaScriptで読み込まれた後の本文テキストと、裏で呼ばれたAPIの応答を出力する。
// 使い方: node biyori.mjs <場番号> <レース番号> <YYYYMMDD> [出力先ディレクトリ]
import { createRequire } from 'module';
import { writeFileSync, mkdirSync } from 'fs';
const require = createRequire(import.meta.url);
let pw;
try { pw = require('playwright'); } catch { pw = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright'); }

const [place, race, date, outDir = '.'] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const url = `https://kyoteibiyori.com/race_shusso.php?place_no=${place}&race_no=${race}&hiduke=${date}`;
const browser = await pw.chromium.launch();
const page = await browser.newPage();
const api = [];
page.on('response', async (res) => {
  const ct = res.headers()['content-type'] || '';
  if (res.url().includes('kyoteibiyori.com') && (ct.includes('json') || res.request().resourceType() === 'xhr' || res.request().resourceType() === 'fetch')) {
    try { api.push({ url: res.url(), body: (await res.text()).slice(0, 200000) }); } catch {}
  }
});
await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
await page.waitForTimeout(3000);
writeFileSync(`${outDir}/page.txt`, await page.innerText('body'));
writeFileSync(`${outDir}/api.json`, JSON.stringify(api, null, 1));
console.log(`page.txt と api.json を ${outDir} に保存しました（API応答 ${api.length} 件）`);
await browser.close();
