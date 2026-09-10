"""Optional real-browser check. Requires Chrome and websocket-client; no Ollama calls.

Run from the repo root: python tests/browser_smoke.py
Uses cached embeddings, ports 8017/9337, and an isolated temporary browser profile.
The backend and browser are stopped at the end; the user's server is untouched.
"""
import base64
import csv
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import websocket

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
print('Loading production backend with cached embeddings...', flush=True)
from backend import app as backend
import uvicorn

work = pathlib.Path(tempfile.mkdtemp(prefix='rag-browser-check-'))
backend.STAGE_LOG_PATH = work / 'stage-logs.jsonl'
server = uvicorn.Server(uvicorn.Config(backend.app, host='127.0.0.1', port=8017, log_level='warning'))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
chrome = ws = None
errors = []
sequence = 0


def command(method, params=None):
    global sequence
    sequence += 1
    ws.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
    while True:
        result = json.loads(ws.recv())
        if result.get('method') == 'Runtime.exceptionThrown':
            errors.append(result['params'])
        if result.get('id') == sequence:
            if 'error' in result:
                raise RuntimeError(result['error'])
            return result.get('result', {})


def evaluate(expression):
    result = command('Runtime.evaluate', {'expression': expression, 'awaitPromise': True, 'returnByValue': True})
    if 'exceptionDetails' in result:
        raise AssertionError(result['exceptionDetails'])
    return result.get('result', {}).get('value')


def wait_until(expression, timeout=150):
    end = time.time() + timeout
    while time.time() < end:
        if evaluate(expression):
            return
        time.sleep(0.4)
    raise AssertionError('Timed out: ' + str(evaluate("({progress:document.getElementById('evaluation-progress')?.textContent,error:document.getElementById('evaluation-error')?.textContent,errors:window.__errors})")))


try:
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started
    chrome_path = os.environ.get('CHROME_BINARY') or shutil.which('google-chrome') or r'C:\Program Files\Google\Chrome\Application\chrome.exe'
    chrome_log = open(work / 'chrome.log', 'w')
    chrome = subprocess.Popen([
        chrome_path, '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        '--remote-debugging-port=9337', '--window-size=1500,1050',
        '--user-data-dir=' + str(work / 'chrome-profile'), 'about:blank'
    ], stdout=chrome_log, stderr=chrome_log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    pages = []
    for _ in range(30):
        if chrome.poll() is not None:
            break
        try:
            pages = [page for page in httpx.get('http://127.0.0.1:9337/json', timeout=1).json()
                     if page.get('type') == 'page' and page.get('url') == 'about:blank']
            if pages:
                break
        except Exception:
            time.sleep(0.2)
    assert pages, 'Chrome debugging endpoint did not start: ' + str(work / 'chrome.log')
    ws = websocket.create_connection(pages[0]['webSocketDebuggerUrl'], suppress_origin=True, timeout=180)
    command('Page.enable')
    command('Runtime.enable')
    command('Page.addScriptToEvaluateOnNewDocument', {'source': '''
        const originalFetch = window.fetch.bind(window);
        window.fetch = (input, init) => originalFetch(typeof input === 'string' ? input.replaceAll('127.0.0.1:8000','127.0.0.1:8017').replaceAll('localhost:8000','127.0.0.1:8017') : input, init);
        window.__errors = [];
        window.addEventListener('error', event => window.__errors.push(event.message));
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('llm-model').value = 'synthetic-evaluator';
        }, {capture: true});
    '''})
    command('Page.navigate', {'url': 'http://127.0.0.1:8017/'})
    wait_until("document.getElementById('evaluation-progress')?.textContent.startsWith('Complete')")
    assert backend.EMBEDDING_MODELS.get('sentence-transformers/all-mpnet-base-v2') is not None, 'Real embedding model unavailable'
    assert backend.CHROMA_AVAILABLE
    assert evaluate("document.querySelectorAll('input[name=perturbation-type]').length") == 9
    assert 'Synthetic' in evaluate("document.getElementById('generation-sources').textContent")
    print('PASS: initial single-method run with real cached embeddings and Chroma', flush=True)
    evaluate('''
        document.querySelectorAll('input[name=perturbation-type]').forEach(input => input.checked = true);
        document.getElementById('perturbation-seed').value = 42;
        document.getElementById('evaluation-mode').value = 'separate';
        document.getElementById('evaluation-mode').dispatchEvent(new Event('change', {bubbles:true}));
        document.getElementById('run-pipeline-btn').click();
    ''')
    wait_until("document.getElementById('evaluation-progress').textContent === 'Complete: 9 variants.'")
    assert evaluate("document.getElementById('variant-select').options.length") == 9
    evaluate("document.getElementById('variant-select').value = 4; document.getElementById('variant-select').dispatchEvent(new Event('change'));")
    assert 'highest-ranked' in evaluate("document.getElementById('perturbation-changes').textContent")
    assert evaluate("document.getElementById('telemetry-rank-jaccard').textContent") == '1.0000'
    saved = evaluate("JSON.parse(localStorage.getItem('rag_evaluation_benchmark_history'))")
    assert len(saved) == 10 and len({row['evaluationId'] for row in saved[1:]}) == 1
    print('PASS: nine separate variants, selector, stage metrics, shared baseline in history', flush=True)
    evaluate("document.getElementById('evaluation-mode').value='combined'; document.getElementById('evaluation-mode').dispatchEvent(new Event('change',{bubbles:true})); document.getElementById('run-pipeline-btn').click();")
    wait_until("document.getElementById('evaluation-progress').textContent === 'Complete: 1 variant.'")
    saved = evaluate("JSON.parse(localStorage.getItem('rag_evaluation_benchmark_history'))")
    assert len(saved) == 11 and len(saved[-1]['result']['changes']) == 9 and saved[-1]['seed'] == 42
    screenshot = command('Page.captureScreenshot', {'format': 'png'})
    (work / 'dashboard.png').write_bytes(base64.b64decode(screenshot['data']))
    print('PASS: nine combined methods, changes, settings and seed persisted', flush=True)
    evaluate('''
        document.querySelectorAll('input[name=perturbation-type]').forEach(input => input.checked = input.value === 'emoji');
        document.getElementById('emoji-style').value = 'substitute';
        document.getElementById('emoji-count').value = 2;
        document.getElementById('custom-query-input').value = 'before 10/09/2026';
        document.getElementById('emoji-style').dispatchEvent(new Event('change', {bubbles:true}));
    ''')
    assert evaluate("document.getElementById('emoji-position').disabled")
    evaluate("document.getElementById('run-pipeline-btn').click();")
    wait_until("!document.getElementById('run-pipeline-btn').disabled && document.getElementById('evaluation-progress').textContent === 'Complete: 1 variant.'")
    saved = evaluate("JSON.parse(localStorage.getItem('rag_evaluation_benchmark_history'))")
    encoded = 'b4️⃣ 1️⃣0️⃣/0️⃣9️⃣/2️⃣0️⃣2️⃣6️⃣'
    assert evaluate("document.getElementById('perturbed-query-text').textContent") == encoded
    assert saved[-1]['settings']['emoji_style'] == 'substitute'
    assert saved[-1]['settings']['emoji_count'] == 2
    assert saved[-1]['result']['perturbed_query'] == encoded
    assert len(saved[-1]['result']['changes'][0]['details']) == 2
    print('PASS: emoji controls, word/date encoding, audit and Unicode history round-trip', flush=True)
    # Every method must work alone, without requiring a selection from the other group.
    # Leave invalid values in unrelated controls to catch hidden-field validation bugs.
    methods = evaluate("Array.from(document.querySelectorAll('input[name=perturbation-type]'), input => input.value)")
    for mode in ['combined', 'separate']:
        for method in methods:
            evaluate('''(() => {
                const method = %s;
                document.querySelectorAll('input[name=perturbation-type]').forEach(input => input.checked = input.value === method);
                document.getElementById('evaluation-mode').value = %s;
                document.getElementById('custom-query-input').value = '';
                document.querySelectorAll('[data-method-settings] input[type=number]').forEach(input => {
                    input.value = input.closest('[data-method-settings]').dataset.methodSettings === method ? input.defaultValue : '';
                });
                document.getElementById('evaluation-mode').dispatchEvent(new Event('change', {bubbles:true}));
                // Simulate a stale enabled control after browser form-state restoration.
                document.querySelectorAll('[data-method-settings] input[type=number]').forEach(input => { input.disabled = false; });
                document.getElementById('run-pipeline-btn').click();
            })()''' % (json.dumps(method), json.dumps(mode)))
            wait_until("!document.getElementById('run-pipeline-btn').disabled && document.getElementById('evaluation-progress').textContent === 'Complete: 1 variant.'")
            latest = evaluate("JSON.parse(localStorage.getItem('rag_evaluation_benchmark_history')).at(-1)")
            assert latest['methods'] == [method] and latest['selectedMethods'] == [method], latest['methods']
            assert latest['mode'] == mode
            assert evaluate("document.getElementById('evaluation-error').hidden")
        print('PASS: all nine methods run individually in ' + mode + ' mode with unrelated controls empty', flush=True)
    # An invalid setting for the selected method must explain the problem, not silently stop.
    evaluate('''
        document.querySelectorAll('input[name=perturbation-type]').forEach(input => input.checked = input.value === 'emoji');
        document.getElementById('emoji-count').value = 0;
        document.getElementById('emoji-style').dispatchEvent(new Event('change', {bubbles:true}));
        document.getElementById('run-pipeline-btn').click();
    ''')
    assert not evaluate("document.getElementById('evaluation-error').hidden")
    assert 'Maximum emoji edits' in evaluate("document.getElementById('evaluation-error').textContent")
    print('PASS: invalid selected settings show a specific validation message', flush=True)
    checks = evaluate('''(async () => {
        const storage = await import('/js/storage.js');
        const bridge = await import('/js/ragEngine.js');
        const old = [{id:1,type:'spelling',retrievalSim:1,generationSim:1,amplificationFactor:0,status:'Proportional'}];
        const saved = localStorage.getItem('rag_evaluation_benchmark_history');
        localStorage.setItem('rag_evaluation_benchmark_history', JSON.stringify(old));
        if (storage.loadHistoryFromStorage()[0].type !== 'spelling') throw Error('Legacy history failed');
        const csv = storage.historyCSV([{...old[0], result:{answer:'A "quote", and\\na newline', emoji:'b4️⃣ 2️⃣0️⃣2️⃣6️⃣'}}]);
        if (!csv.includes('"0"') || !csv.includes('"Result JSON"')) throw Error('CSV failed');
        localStorage.setItem('rag_evaluation_benchmark_history', saved);
        const events = [];
        const bytes = new TextEncoder().encode(JSON.stringify({event:'progress',message:'caf\\u00e9'})+'\\n'+JSON.stringify({event:'complete',result:{ok:true}}));
        const body = new ReadableStream({start(controller) { for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); controller.close(); }});
        const result = await bridge.readEvaluationStream(new Response(body), event => events.push(event));
        if (!result.ok || events[0].message !== 'caf\\u00e9') throw Error('Chunked UTF-8 stream failed');
        let rejected = false;
        try { await bridge.readEvaluationStream(new Response('{"event":"progress"}\\n'), () => {}); } catch { rejected = true; }
        if (!rejected) throw Error('Missing completion was not rejected');
        const fetchBefore = window.fetch;
        let payload;
        try {
            window.fetch = async (url, options) => {
                payload = JSON.parse(options.body);
                return new Response('{}', {status:200});
            };
            await bridge.executeBackendEvaluation({baselineQuery:'Question', groundTruth:'Answer',
                perturbationTypes:['spelling','shuffle'], mode:'separate', typoRate:0, seed:0});
        } finally { window.fetch = fetchBefore; }
        if (payload.typo_rate !== 0 || payload.seed !== 0 || payload.mode !== 'separate' || payload.perturbation_types.length !== 2) throw Error('Multi-method request or zero values lost');
        return {csv};
    })()''')
    rows = list(csv.reader(io.StringIO(checks['csv'])))
    assert len(rows[0]) == len(rows[1]) == 19
    assert json.loads(rows[1][-1])['answer'] == 'A "quote", and\na newline'
    assert json.loads(rows[1][-1])['emoji'] == 'b4️⃣ 2️⃣0️⃣2️⃣6️⃣'
    print('PASS: legacy history, CSV round-trip/zeroes, chunked UTF-8, request payload and incomplete stream errors', flush=True)
    evaluate("document.querySelectorAll('input[name=perturbation-type]').forEach(input => input.checked=false); document.getElementById('evaluation-mode').dispatchEvent(new Event('change',{bubbles:true}));")
    assert evaluate("document.getElementById('run-pipeline-btn').disabled")
    evaluate("document.querySelector('input[value=spelling]').checked=true; document.getElementById('evaluation-mode').dispatchEvent(new Event('change',{bubbles:true}));")
    assert not evaluate("document.getElementById('run-pipeline-btn').disabled")
    evaluate("window.fetch = async () => new Response('test failure', {status:500}); document.getElementById('run-pipeline-btn').click();")
    wait_until("!document.getElementById('evaluation-error').hidden", timeout=10)
    assert not evaluate("document.getElementById('run-pipeline-btn').disabled")
    assert not errors, errors
    assert not evaluate('window.__errors'), evaluate('window.__errors')
    print('PASS: visible API error, restored controls, no browser exceptions', flush=True)
    print('BROWSER SMOKE PASSED; artifacts: ' + str(work), flush=True)
finally:
    if ws:
        try:
            command('Browser.close')
        except Exception:
            pass
        ws.close()
    if chrome:
        try:
            chrome.wait(timeout=8)
        except subprocess.TimeoutExpired:
            chrome.terminate()
        chrome_log.close()
    server.should_exit = True
    thread.join(timeout=10)
