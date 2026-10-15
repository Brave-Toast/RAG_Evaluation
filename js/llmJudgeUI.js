/**
 * Self-contained Frontend Controller for the LLM-as-a-Judge Stage.
 * Automatically hooks into window.fetch, variant selection, and telemetry cards.
 */

let latestJudgeResultPayload = null;

function formatJudgeScore(val) {
    return typeof val === 'number' && !Number.isNaN(val) ? val.toFixed(2) : '---';
}

function updateAttributionCardFromMetrics(metrics) {
    if (!metrics || !metrics.use_llm_judge || !metrics.failure_attribution) return;
    const badge = document.getElementById('telemetry-attribution-badge');
    const desc = document.getElementById('telemetry-attribution-desc');
    if (badge) {
        badge.textContent = metrics.failure_attribution;
        const styles = {
            SUCCESS: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
            RETRIEVAL_FAILURE: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
            GENERATION_FAILURE: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
            PARAMETRIC_LEAKAGE: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
        };
        badge.className = `inline-block text-xs font-bold px-2.5 py-1 rounded-lg border ${styles[metrics.failure_attribution] || styles.SUCCESS
            }`;
    }
    if (desc && metrics.failure_root_cause) {
        desc.textContent = metrics.failure_root_cause;
        desc.title = metrics.failure_root_cause;
    }
}

export function renderJudgeTelemetry(metrics) {
    const panel = document.getElementById('judge-telemetry-panel');
    if (!panel) return;

    const isEnabled = Boolean(
        metrics &&
        metrics.use_llm_judge &&
        metrics.judge_correctness_perturbed !== null &&
        metrics.judge_correctness_perturbed !== undefined
    );

    panel.hidden = !isEnabled;
    if (!isEnabled) return;

    const corrBase = document.getElementById('telemetry-judge-corr-base');
    const corrPert = document.getElementById('telemetry-judge-corr-pert');
    const faithBase = document.getElementById('telemetry-judge-faith-base');
    const faithPert = document.getElementById('telemetry-judge-faith-pert');
    const relBase = document.getElementById('telemetry-judge-rel-base');
    const relPert = document.getElementById('telemetry-judge-rel-pert');
    const verdictEl = document.getElementById('telemetry-judge-verdict');
    const rationaleEl = document.getElementById('telemetry-judge-rationale');

    if (corrBase) corrBase.textContent = `B: ${formatJudgeScore(metrics.judge_correctness_baseline)}`;
    if (corrPert) corrPert.textContent = `P: ${formatJudgeScore(metrics.judge_correctness_perturbed)}`;
    if (faithBase) faithBase.textContent = `B: ${formatJudgeScore(metrics.judge_faithfulness_baseline)}`;
    if (faithPert) faithPert.textContent = `P: ${formatJudgeScore(metrics.judge_faithfulness_perturbed)}`;
    if (relBase) relBase.textContent = `B: ${formatJudgeScore(metrics.judge_context_relevance_baseline)}`;
    if (relPert) relPert.textContent = `P: ${formatJudgeScore(metrics.judge_context_relevance_perturbed)}`;

    if (verdictEl) {
        const verdict = metrics.judge_verdict_perturbed || '---';
        const model = metrics.judge_model || 'judge';
        verdictEl.textContent = `${verdict} (${model})`;
    }
    if (rationaleEl) {
        const rationale = metrics.judge_rationale_perturbed || '';
        rationaleEl.textContent = rationale;
        rationaleEl.title = rationale;
    }

    updateAttributionCardFromMetrics(metrics);
}

function getSelectedVariantMetrics(resultPayload) {
    if (!resultPayload) return null;
    const variantSelect = document.getElementById('variant-select');
    const idx = variantSelect ? Number.parseInt(variantSelect.value, 10) : 0;
    if (Array.isArray(resultPayload.variants) && resultPayload.variants.length > 0) {
        const chosen = resultPayload.variants[Number.isNaN(idx) ? 0 : idx] || resultPayload.variants[0];
        return chosen.metrics || resultPayload.metrics || null;
    }
    return resultPayload.metrics || null;
}

function installFetchInterceptor() {
    const previousFetch = window.fetch.bind(window);

    window.fetch = async (input, init) => {
        const url = typeof input === 'string' ? input : (input?.url || '');
        const isEvalCall =
            url.includes('/api/run-evaluation') &&
            init &&
            typeof init.body === 'string';

        let modifiedInit = init;
        if (isEvalCall) {
            try {
                const bodyObj = JSON.parse(init.body);
                const useJudgeEl = document.getElementById('use-llm-judge');
                const judgeModelEl = document.getElementById('judge-model');
                if (bodyObj.use_llm_judge === undefined && useJudgeEl) {
                    bodyObj.use_llm_judge = Boolean(useJudgeEl.checked);
                }
                if (bodyObj.judge_model === undefined && judgeModelEl) {
                    bodyObj.judge_model = judgeModelEl.value || 'ollama/qwen3.5:9b';
                }
                modifiedInit = { ...init, body: JSON.stringify(bodyObj) };
            } catch {
                modifiedInit = init;
            }
        }

        const response = await previousFetch(input, modifiedInit);

        if (!isEvalCall || !response.ok) {
            return response;
        }

        if (url.endsWith('/api/run-evaluation')) {
            response
                .clone()
                .json()
                .then((data) => {
                    latestJudgeResultPayload = data;
                    setTimeout(() => renderJudgeTelemetry(getSelectedVariantMetrics(data)), 50);
                })
                .catch(() => { });
            return response;
        }

        if (url.endsWith('/api/run-evaluation/stream') && response.body) {
            const [appStream, judgeStream] = response.body.tee();
            (async () => {
                const reader = judgeStream.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                try {
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        while (buffer.includes('\n')) {
                            const nlIdx = buffer.indexOf('\n');
                            const line = buffer.slice(0, nlIdx).trim();
                            buffer = buffer.slice(nlIdx + 1);
                            if (!line) continue;
                            try {
                                const evt = JSON.parse(line);
                                if (evt.event === 'complete' && evt.result) {
                                    latestJudgeResultPayload = evt.result;
                                    setTimeout(
                                        () => renderJudgeTelemetry(getSelectedVariantMetrics(evt.result)),
                                        60
                                    );
                                }
                            } catch { }
                        }
                    }
                } catch { }
            })();

            return new Response(appStream, {
                status: response.status,
                statusText: response.statusText,
                headers: response.headers,
            });
        }

        return response;
    };
}

async function populateJudgeModels() {
    const select = document.getElementById('judge-model');
    if (!select) return;
    try {
        const res = await fetch('/api/judges');
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data.judges) && data.judges.length > 0) {
            select.innerHTML = data.judges
                .map(
                    (j) =>
                        `<option value="${j.name}" ${j.name === data.default ? 'selected' : ''
                        }>${j.label}</option>`
                )
                .join('');
        }
    } catch { }
}

function initJudgeUI() {
    const checkbox = document.getElementById('use-llm-judge');
    const select = document.getElementById('judge-model');
    const variantSelect = document.getElementById('variant-select');

    if (checkbox && select) {
        select.disabled = !checkbox.checked;
        checkbox.addEventListener('change', () => {
            select.disabled = !checkbox.checked;
            if (!checkbox.checked) {
                const panel = document.getElementById('judge-telemetry-panel');
                if (panel) panel.hidden = true;
            }
        });
    }

    if (variantSelect) {
        variantSelect.addEventListener('change', () => {
            if (latestJudgeResultPayload) {
                setTimeout(
                    () => renderJudgeTelemetry(getSelectedVariantMetrics(latestJudgeResultPayload)),
                    20
                );
            }
        });
    }

    populateJudgeModels();
}

installFetchInterceptor();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initJudgeUI);
} else {
    initJudgeUI();
}