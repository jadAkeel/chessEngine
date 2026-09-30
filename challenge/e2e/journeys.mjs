// End-to-end browser journeys against challenge/e2e/mock_api.py.
// Needs playwright-core and a Chromium build (not project dependencies):
//   BASE_URL=http://127.0.0.1:8021 CHROME_PATH=/path/to/chrome node journeys.mjs
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { chromium } from 'playwright-core'

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:8021'
const OUT = process.env.OUT_DIR || path.resolve('e2e-output')
const KEY_A = 'test-key-AAAA-not-real'
const KEY_B = 'test-key-BBBB-not-real'
fs.mkdirSync(OUT, { recursive: true })

const results = []
let current = null
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH, headless: true })

async function fresh(viewport) {
  const context = await browser.newContext({ viewport, acceptDownloads: true, reducedMotion: 'reduce' })
  const page = await context.newPage()
  const problems = []
  const urls = []
  // HTTP error statuses are expected in failure journeys; only JavaScript errors count.
  page.on('console', message => { if (message.type() === 'error' && !message.text().startsWith('Failed to load resource')) problems.push(message.text()) })
  page.on('pageerror', error => problems.push(String(error)))
  page.on('request', request => urls.push(request.url()))
  page.on('dialog', dialog => dialog.accept())
  current = page
  await page.goto(BASE_URL)
  return { context, page, problems, urls }
}

const card = (page, model) => page.locator(`#setup-${model}`)

async function configure(page, model, { provider, url, id, key }) {
  const root = card(page, model)
  if (provider) await root.getByLabel('Provider').selectOption(provider)
  if (url !== undefined) await root.getByLabel('API base URL').fill(url)
  await root.getByLabel('Model ID').fill(id)
  await root.getByLabel('API key', { exact: true }).fill(key)
}

async function setupPair(page, a = 'strong', b = 'weak') {
  await configure(page, 'A', { url: 'https://api.example.com/v1', id: a, key: KEY_A })
  await configure(page, 'B', { provider: 'anthropic', id: b, key: KEY_B })
}

const status = page => page.locator('.status-text')

async function journey(name, fn) {
  const started = Date.now()
  try {
    await fn()
    results.push({ name, ok: true, ms: Date.now() - started })
    console.log(`PASS ${name}`)
  } catch (error) {
    results.push({ name, ok: false, error: String(error?.message || error) })
    await current?.screenshot({ path: path.join(OUT, `failure-${results.length}.png`), fullPage: true }).catch(() => {})
    console.log(`FAIL ${name}\n  ${error?.stack || error}`)
  }
}

async function assertNoKeys(page, urls) {
  const storage = await page.evaluate(() => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage }, cookie: document.cookie }))
  assert.ok(!storage.includes('test-key'), 'key found in browser storage')
  assert.ok(!urls.some(url => url.includes('test-key')), 'key found in a URL')
  assert.ok(!page.url().includes('test-key'))
}

await journey('public onboarding: sample replay needs no key or provider calls', async () => {
  const { context, page, problems, urls } = await fresh({ width: 375, height: 812 })
  await page.getByRole('heading', { name: 'Compare AI models by how they play chess.' }).waitFor()
  const liveFen = await page.locator('.board-frame').getAttribute('aria-label')
  await page.getByRole('button', { name: 'Watch sample replay — no key needed' }).click()
  await page.getByRole('heading', { name: 'Scripted demonstration' }).waitFor()
  for (let ply = 0; ply < 10; ply++) await page.getByRole('button', { name: 'Next sample move' }).click()
  await page.getByText('End of this opening sample. No winner is determined.').waitFor()
  assert.ok(!urls.some(url => url.includes('/api/move')), 'sample made a provider request')
  assert.equal(await page.locator('.move-choice').count(), 0, 'sample entered real match history')
  assert.equal(await page.locator('.board-frame').getAttribute('aria-label'), liveFen)
  assert.equal(await page.getByRole('button', { name: 'JSON report' }).isDisabled(), true)
  assert.equal(await page.getByRole('button', { name: 'PGN', exact: true }).isDisabled(), true)
  assert.equal(await card(page, 'A').getByLabel('API key', { exact: true }).inputValue(), '')
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'sample causes mobile overflow')
  await page.screenshot({ path: path.join(OUT, 'mobile-sample.png'), fullPage: true })
  await page.getByRole('button', { name: 'Close sample replay' }).click()
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('desktop: setup validation, guidance and connection tests', async () => {
  const { context, page, problems, urls } = await fresh({ width: 1440, height: 900 })
  await page.getByRole('button', { name: 'Start match' }).click()
  await page.getByRole('alert').getByText('Finish setting up the models').waitFor()
  assert.equal(await page.locator('.field-error').count() >= 4, true)
  const a = card(page, 'A')
  await a.getByLabel('API base URL').fill('http://api.example.com/v1')
  await a.getByText('Use an https:// URL.').waitFor()
  await a.getByLabel('API base URL').fill('https://api.example.com/v1/chat/completions')
  await a.getByText(/base URL only/).waitFor()
  await setupPair(page)
  await a.getByText('https://api.example.com/v1/chat/completions').waitFor()
  await card(page, 'B').getByText('https://api.anthropic.com/v1/messages').waitFor()
  await a.getByRole('button', { name: 'Test connection' }).click()
  await a.getByText(/Connected\. Played e4/).waitFor()
  await a.getByText('Verified').waitFor()
  // Server-side SSRF guard rejects private targets even when the browser allows the URL shape.
  await a.getByLabel('API base URL').fill('https://127.0.0.1/v1')
  assert.equal(await a.getByText('Not tested').count(), 1, 'editing must clear verification')
  await a.getByRole('button', { name: 'Test connection' }).click()
  await a.getByText('Provider URL must be a public HTTPS endpoint').waitFor()
  await a.getByLabel('API base URL').fill('https://api.example.com/v1')
  await a.getByLabel('Model ID').fill('strong-badkey')
  await a.getByRole('button', { name: 'Test connection' }).click()
  await a.getByText('Provider rejected the API key or model access').waitFor()
  await a.getByText('Test failed').waitFor()
  await a.getByLabel('Model ID').fill('strong')
  const keyInput = a.getByLabel('API key', { exact: true })
  assert.equal(await keyInput.getAttribute('type'), 'password')
  await a.getByRole('button', { name: 'Show API key' }).click()
  assert.equal(await keyInput.getAttribute('type'), 'text')
  await a.getByRole('button', { name: 'Hide API key' }).click()
  await page.screenshot({ path: path.join(OUT, 'desktop-setup.png'), fullPage: true })
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('recommended settings preserve credentials, send temperature zero and lock after a move', async () => {
  const { context, page, problems, urls } = await fresh({ width: 375, height: 812 })
  await setupPair(page)
  const a = card(page, 'A')
  const b = card(page, 'B')
  for (const root of [a, b]) await root.locator('summary').click()
  await a.getByLabel('Starting preset').selectOption('economy')
  assert.equal(await a.getByLabel('Max output tokens').inputValue(), '2048')
  await a.getByLabel('Temperature (optional)').fill('0')
  await page.locator('.settings-warning').getByText(/Temperature: A 0/).waitFor()
  await page.getByLabel('Shared starting preset').selectOption('balanced')
  await page.getByRole('button', { name: 'Apply to both models' }).click()
  for (const root of [a, b]) {
    assert.equal(await root.getByLabel('Starting preset').inputValue(), 'balanced')
    assert.equal(await root.getByLabel('Max output tokens').inputValue(), '4096')
    assert.equal(await root.getByLabel('Timeout (seconds)').inputValue(), '180')
    assert.equal(await root.getByLabel('Temperature (optional)').inputValue(), '')
  }
  assert.equal(await a.getByLabel('API key', { exact: true }).inputValue(), KEY_A)
  assert.equal(await b.getByLabel('API key', { exact: true }).inputValue(), KEY_B)
  assert.equal(await a.getByLabel('Model ID').inputValue(), 'strong')
  assert.equal(await b.getByLabel('Provider').inputValue(), 'anthropic')
  assert.equal(await page.locator('.settings-warning').count(), 0)
  await b.getByLabel('Reasoning effort').selectOption('high')
  await b.getByLabel('Temperature (optional)').fill('0.5')
  await b.getByLabel('Temperature (optional)').blur()
  await b.getByText('With Anthropic thinking, leave temperature blank or set it to 1.').waitFor()
  await page.getByRole('button', { name: 'Apply to both models' }).click()
  await a.getByLabel('Temperature (optional)').fill('0')
  await b.getByLabel('Temperature (optional)').fill('0')
  await page.screenshot({ path: path.join(OUT, 'mobile-settings.png'), fullPage: true })
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'settings cause mobile overflow')
  const sent = page.waitForRequest(request => request.url().endsWith('/api/move') && request.method() === 'POST')
  await page.getByRole('button', { name: 'One move', exact: true }).click()
  const payload = (await sent).postDataJSON()
  assert.equal(payload.player.temperature, 0)
  assert.equal(payload.player.max_tokens, 4096)
  assert.deepEqual(payload.moves, [])
  await page.locator('.move-choice').first().waitFor()
  await page.locator('.rationale').getByText(/input 850 · output 40/).waitFor()
  await a.getByRole('button', { name: 'Show settings' }).click()
  assert.equal(await a.getByLabel('Starting preset').isDisabled(), true)
  assert.equal(await a.getByLabel('Temperature (optional)').isDisabled(), true)
  assert.equal(await page.getByRole('button', { name: 'Apply to both models' }).isDisabled(), true)
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'JSON report' }).click()])
  const report = JSON.parse(fs.readFileSync(await download.path(), 'utf8'))
  assert.equal(report.models.A.temperature, 0)
  assert.equal(report.models.B.temperature, 0)
  assert.equal(report.settings_match, true)
  assert.ok(!JSON.stringify(report).includes('test-key'))
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('desktop: paired match swaps colors, scores and exports without keys', async () => {
  const { context, page, problems, urls } = await fresh({ width: 1440, height: 900 })
  await setupPair(page)
  await page.getByRole('button', { name: 'Start match' }).click()
  await page.locator('.move-choice').first().waitFor()
  await page.screenshot({ path: path.join(OUT, 'desktop-live.png') })
  await page.getByText(/Game 1 finished \(1–0\)/).waitFor({ timeout: 30000 })
  await page.screenshot({ path: path.join(OUT, 'desktop-intermission.png') })
  await status(page).getByText('Match complete: Model A wins 2–0.').waitFor({ timeout: 30000 })
  await card(page, 'A').getByText('Verified').waitFor()
  await card(page, 'B').getByText('Verified').waitFor()
  const games = page.locator('.game-list li')
  assert.match(await games.nth(0).innerText(), /A White · B Black[\s\S]*1–0 · Model A wins/)
  assert.match(await games.nth(1).innerText(), /B White · A Black[\s\S]*0–1 · Model A wins/)
  assert.equal(await page.locator('.score-side').first().locator('strong').innerText(), '2')
  assert.equal(await page.locator('.score-side.right strong').innerText(), '0')
  // Game 2 bars: B (weak) is White at the bottom, A is Black at the top.
  assert.match(await page.locator('.player-bar').last().innerText(), /weak[\s\S]*White · Model B/)
  assert.match(await page.locator('.player-bar').first().innerText(), /Winner/)
  assert.match(await page.locator('.result-strip').innerText(), /0–1[\s\S]*Checkmate · Model A wins game 2/)
  await page.screenshot({ path: path.join(OUT, 'desktop-complete.png'), fullPage: true })

  // Review game 1 via the game tabs and keyboard.
  await page.getByRole('button', { name: 'Game 1', exact: true }).click()
  await page.getByText(/Reviewing game 1/).waitFor()
  await page.locator('body').click({ position: { x: 5, y: 300 } })
  await page.keyboard.press('ArrowLeft')
  await page.getByText('Reviewing game 1, after 2… g5').waitFor()
  await page.getByRole('button', { name: /^3\. Qh5#, White, Model A$/ }).click()
  await page.locator('.rationale').getByText('Delivers checkmate.').waitFor()
  await page.keyboard.press('Escape')
  await page.getByText(/Reviewing game/).waitFor({ state: 'detached' })

  const [jsonDownload] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'JSON report' }).click()])
  const report = JSON.parse(fs.readFileSync(await jsonDownload.path(), 'utf8'))
  const reportText = JSON.stringify(report)
  assert.ok(!reportText.includes('test-key'))
  assert.equal(report.complete, true)
  assert.deepEqual(report.score, { A: 2, B: 0 })
  assert.deepEqual(report.games.map(game => [game.white, game.black, game.result, game.winner]), [['A', 'B', '1-0', 'A'], ['B', 'A', '0-1', 'A']])
  assert.ok(report.games[1].moves.every(move => move.model === (move.color === 'w' ? 'B' : 'A')))
  assert.equal(report.models.B.base_url, 'https://api.anthropic.com/v1')
  const [pgnDownload] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'PGN' }).click()])
  const pgn = fs.readFileSync(await pgnDownload.path(), 'utf8')
  assert.ok(!pgn.includes('test-key'))
  assert.match(pgn, /\[Round "2"\][\s\S]*\[White "Model B: weak"\][\s\S]*\[Black "Model A: strong"\][\s\S]*\[Result "0-1"\]/)
  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2))
  fs.writeFileSync(path.join(OUT, 'games.pgn'), pgn)

  await page.getByRole('button', { name: 'New match' }).click()
  await status(page).getByText('Configure both models, then start the match.').waitFor()
  assert.equal(await page.locator('.move-list').count(), 0)
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('pause, resume and reset while a request is in flight', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await setupPair(page, 'strong-slow', 'weak')
  await page.getByRole('button', { name: 'Start match' }).click()
  await page.getByText(/Thinking \d+s/).waitFor()
  await page.screenshot({ path: path.join(OUT, 'desktop-thinking.png') })
  await page.getByRole('button', { name: 'Pause' }).click()
  await status(page).getByText('Paused before the first move.').waitFor()
  await page.waitForTimeout(4800)
  assert.equal(await page.locator('.move-choice').count(), 0, 'discarded response must not be applied')
  assert.equal(await page.locator('.bar-state.thinking').count(), 0)
  await page.getByRole('button', { name: 'Resume' }).click()
  await page.locator('.move-choice').first().waitFor({ timeout: 10000 })
  await page.getByRole('button', { name: 'Pause' }).click()
  await page.getByRole('button', { name: 'Resume' }).waitFor()
  // Settings lock after the first move; the API key stays editable while paused.
  const a = card(page, 'A')
  await a.getByRole('button', { name: 'Show settings' }).click()
  assert.equal(await a.getByLabel('Model ID').isDisabled(), true)
  assert.equal(await a.getByLabel('API key', { exact: true }).isEnabled(), true)
  await page.getByRole('button', { name: 'Resume' }).click()
  await page.getByText(/Thinking \d+s/).waitFor()
  const before = await page.locator('.move-choice').count()
  await page.getByRole('button', { name: 'Reset' }).click()
  await status(page).getByText('Configure both models, then start the match.').waitFor()
  await page.waitForTimeout(4500)
  assert.equal(await page.locator('.move-choice').count(), 0, `late move applied after reset (had ${before})`)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('provider failure pauses the match and retry continues it', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await setupPair(page, 'strong-flaky', 'weak-illegal-once')
  await page.getByRole('button', { name: 'Start match' }).click()
  const alert = page.getByRole('alert')
  await alert.getByText(/Timed out or server busy · Model [AB]/).waitFor({ timeout: 20000 })
  await alert.getByText('Provider is unavailable (HTTP 503); retry shortly').waitFor()
  await page.screenshot({ path: path.join(OUT, 'desktop-error.png') })
  await alert.getByRole('button', { name: 'Retry move' }).click()
  await page.getByRole('button', { name: 'Pause' }).waitFor()
  while (!(await status(page).innerText()).startsWith('Match complete')) {
    if (await alert.count()) await alert.getByRole('button', { name: 'Retry move' }).click()
    await page.waitForTimeout(250)
  }
  const retried = await page.locator('.move-choice small', { hasText: 'retry' }).count()
  assert.ok(retried > 0, 'illegal-once replies should be marked as retried')
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('auth failure explains the problem and keeps the key replaceable', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await setupPair(page, 'strong', 'weak-badkey')
  await page.getByRole('button', { name: 'Start match' }).click()
  const alert = page.getByRole('alert')
  await alert.getByText('API key or model access rejected · Model B').waitFor({ timeout: 10000 })
  assert.equal(await page.locator('.move-choice').count(), 1)
  await card(page, 'B').getByRole('button', { name: 'Show settings' }).click()
  assert.equal(await card(page, 'B').getByLabel('API key', { exact: true }).isEnabled(), true)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('provider errors cannot expose a submitted API key', async () => {
  const { context, page, problems, urls } = await fresh({ width: 1280, height: 800 })
  await setupPair(page)
  await page.route('**/api/move', route => route.fulfill({
    status: 401,
    contentType: 'application/json',
    body: JSON.stringify({ detail: `Provider rejected ${KEY_A}` }),
  }))
  await page.getByRole('button', { name: 'One move' }).click()
  const alert = page.getByRole('alert')
  await alert.getByText('Provider rejected [REDACTED API KEY]').waitFor()
  assert.ok(!(await alert.innerText()).includes(KEY_A))
  assert.equal(await page.locator('.move-choice').count(), 0)
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'JSON report' }).click()])
  assert.ok(!fs.readFileSync(await download.path(), 'utf8').includes(KEY_A))
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('mismatched server FEN and network failure are rejected safely', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await setupPair(page)
  await page.route('**/api/move', async route => {
    const response = await route.fetch()
    const data = await response.json()
    data.fen_after = data.fen_after.replace(' b ', ' w ')
    await route.fulfill({ response, json: data })
  })
  await page.getByRole('button', { name: 'One move' }).click()
  await page.getByRole('alert').getByText('Board mismatch · Model A').waitFor()
  assert.equal(await page.locator('.move-choice').count(), 0)
  assert.match(await page.locator('.board-frame').getAttribute('aria-label'), /start position/)
  await page.unroute('**/api/move')
  await page.route('**/api/move', route => route.abort('connectionrefused'))
  await page.getByRole('alert').getByRole('button', { name: 'Retry move' }).click()
  await page.getByRole('alert').getByText('Cannot reach the challenge server · Model A').waitFor()
  await page.unroute('**/api/move')
  await page.getByRole('alert').getByRole('button', { name: 'Retry move' }).click()
  await page.locator('.move-choice').first().waitFor()
  assert.equal(await page.getByRole('alert').count(), 0)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('truncation, uneven settings warning and failure tracking', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await setupPair(page, 'strong', 'weak-truncated')
  const b = card(page, 'B')
  await b.getByText(/^Advanced/).click()
  await b.getByLabel('Max output tokens').fill('4096')
  await page.getByRole('note').getByText('Settings differ between the models.').waitFor()
  await page.getByText('Max output tokens: A 16000 · B 4096').waitFor()
  await page.getByRole('button', { name: 'Start match' }).click()
  const alert = page.getByRole('alert')
  await alert.getByText(/ran out of output tokens/).waitFor({ timeout: 10000 })
  await alert.getByText('Request rejected · Model B').waitFor()
  await page.locator('.reliability').getByText(/Failed requests: A 0 · B 1/).waitFor()
  await page.locator('.score-card .settings-warning').waitFor()
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'JSON report' }).click()])
  const report = JSON.parse(fs.readFileSync(await download.path(), 'utf8'))
  assert.equal(report.settings_match, false)
  assert.deepEqual(report.reliability.B, { failed_requests: 1, retried_moves: 0 })
  assert.equal(report.games[0].failures[0].model, 'B')
  assert.ok(!JSON.stringify(report).includes('test-key'))
  await page.screenshot({ path: path.join(OUT, 'desktop-truncated.png'), fullPage: true })
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('mobile: single game, no horizontal overflow, readable board', async () => {
  const { context, page, problems, urls } = await fresh({ width: 375, height: 812 })
  await page.screenshot({ path: path.join(OUT, 'mobile-initial.png') })
  await page.getByLabel('Single game').check()
  await card(page, 'B').getByText('Plays Black').waitFor()
  await setupPair(page)
  await page.screenshot({ path: path.join(OUT, 'mobile-setup.png'), fullPage: true })
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  assert.ok(overflow <= 0, `horizontal overflow ${overflow}px`)
  const board = await page.locator('[data-boardid]').boundingBox()
  assert.ok(board.width >= 320 && board.x >= 0 && board.x + board.width <= 375, `board ${JSON.stringify(board)}`)
  await page.getByRole('button', { name: 'Start match' }).click()
  await status(page).getByText('Match complete: Model A wins 1–0.').waitFor({ timeout: 30000 })
  assert.equal(await page.locator('.game-list li').count(), 1)
  await page.screenshot({ path: path.join(OUT, 'mobile-complete.png'), fullPage: true })
  const overflowAfter = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  assert.ok(overflowAfter <= 0, `horizontal overflow after match ${overflowAfter}px`)
  await assertNoKeys(page, urls)
  assert.deepEqual(problems, [])
  await context.close()
})

await journey('keyboard and accessible names', async () => {
  const { context, page, problems } = await fresh({ width: 1280, height: 800 })
  await page.keyboard.press('Tab')
  assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Skip to the match')
  const unnamed = await page.evaluate(() => [...document.querySelectorAll('button, a, input, select')]
    .filter(el => {
      const label = el.getAttribute('aria-label') || el.textContent.trim() || (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)?.textContent) || el.closest('label')?.textContent
      return !label || !label.trim()
    }).map(el => el.outerHTML.slice(0, 80)))
  assert.deepEqual(unnamed, [])
  // Everything needed to start is reachable by keyboard.
  await setupPair(page)
  const start = page.getByRole('button', { name: 'Start match' })
  await start.focus()
  await page.keyboard.press('Enter')
  await page.getByRole('button', { name: 'Pause' }).waitFor()
  await page.keyboard.press('Enter')
  assert.deepEqual(problems, [])
  await context.close()
})

await browser.close()
fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 2))
const failed = results.filter(result => !result.ok)
console.log(`\n${results.length - failed.length}/${results.length} journeys passed`)
process.exit(failed.length ? 1 : 0)
