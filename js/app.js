/**
 * Main Application Orchestrator & Event Controller
 * Connects UI control inputs to the FastAPI Backend for true RAG evaluation and telemetry analysis.
 */

import { HOTPOTQA_DATASET, loadDatasetFromBackend, getQueryByIndex } from './dataset.js';
import { executeBackendEvaluation, checkBackendHealth } from './ragEngine.js';
import { loadHistoryFromStorage, saveHistoryToStorage, clearHistoryStorage, exportHistoryJSON, exportHistoryCSV } from './storage.js';
import { PERTURBATION_LABELS, methodLabel } from './perturbations.js';
import { renderContexts, updateMetricCards, updateHistoryUI, initChart, updateChart } from './uiManager.js';

let evaluationHistory = [];
let currentVariants = [];
let evaluationRunning = false;
let currentQueryObj = HOTPOTQA_DATASET[0];

window.addEventListener('DOMContentLoaded', async () => {
    // 1. Perform backend health check & update header status badge
    updateHeaderStatusBadge();
    updateEmbeddingProviderAvailability();
    updateLLMProviderAvailability();
    
    // 2. Fetch dynamic 50+ HotpotQA questions dataset from FastAPI backend
    await loadDatasetFromBackend();
    
    // 3. Initialize UI components
    initQuerySelect();
    initChart();
    
    // 4. Restore persisted evaluation history from local storage
    evaluationHistory = loadHistoryFromStorage();
    updateHistoryUI(evaluationHistory);
    updateChart(evaluationHistory);

    loadSelectedQuery();
    bindEventListeners();
    updatePerturbationControls();

    // Evaluation only runs when the Run Evaluation Pipeline button is clicked.
});

async function updateHeaderStatusBadge() {
    const badgeEl = document.getElementById('status-badge');
    if (!badgeEl) return;
    
    const health = await checkBackendHealth();
    if (health && health.status === "online") {
        const isChroma = health.vector_store && health.vector_store.includes("ChromaDB");
        badgeEl.innerHTML = `
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            <span class="text-slate-300 font-medium">FastAPI & ${isChroma ? 'ChromaDB' : 'Dense Retriever'} Active</span>
        `;
    } else {
        badgeEl.innerHTML = `
            <span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
            <span class="text-slate-300 font-medium">Backend Offline (Simulated Fallback)</span>
        `;
    }
}

async function updateEmbeddingProviderAvailability() {
    const health = await checkBackendHealth();
    const openaiOption = document.querySelector('#embedding-provider option[value="openai/text-embedding-3-small"]');
    if (!openaiOption) return;

    if (health && health.openai_configured) {
        openaiOption.disabled = false;
        openaiOption.textContent = "OpenAI / text-embedding-3-small";
    } else {
        openaiOption.disabled = true;
        openaiOption.textContent = "OpenAI / text-embedding-3-small (API Key Required)";
    }
}

async function updateLLMProviderAvailability() {
    const health = await checkBackendHealth();
    const geminiOption = document.querySelector('#llm-model option[value="gemini-3-flash-preview"]');
    if (!geminiOption) return;

    if (health && health.gemini_configured) {
        geminiOption.disabled = false;
        geminiOption.textContent = "Gemini 3 Flash (Live Cloud)";
    } else {
        geminiOption.disabled = true;
        geminiOption.textContent = "Gemini 3 Flash (Live Cloud) (API Key Required)";
    }
}

function initQuerySelect() {
    const select = document.getElementById('query-select');
    if (!select) return;
    select.innerHTML = '';
    
    HOTPOTQA_DATASET.forEach((item, index) => {
        const opt = document.createElement('option');
        opt.value = index;
        const shortQ = item.question.length > 50 ? item.question.substring(0, 50) + "..." : item.question;
        opt.textContent = `[Q${index + 1}] ${shortQ}`;
        select.appendChild(opt);
    });
}

function loadSelectedQuery() {
    const select = document.getElementById('query-select');
    if (!select) return;
    const selectIndex = select.value || 0;
    currentQueryObj = getQueryByIndex(selectIndex);
    
    document.getElementById('ground-truth-preview').textContent = currentQueryObj.ground_truth;
    document.getElementById('custom-query-input').value = '';
}

function bindEventListeners() {
    // Dropdown selection change
    document.getElementById('query-select')?.addEventListener('change', loadSelectedQuery);

    // Clear custom query button
    document.getElementById('clear-custom-query-btn')?.addEventListener('click', () => {
        document.getElementById('custom-query-input').value = '';
        loadSelectedQuery();
    });

    // Typo rate slider display update
    document.getElementById('typo-rate-slider')?.addEventListener('input', (e) => {
        document.getElementById('typo-rate-val').innerText = e.target.value + '%';
    });

    document.getElementById('perturbation-controls').addEventListener('change', updatePerturbationControls);

    document.getElementById('variant-select').addEventListener('change', event => {
        renderVariant(currentVariants[Number(event.target.value)]);
    });

    // Reset history button
    document.getElementById('reset-history-btn')?.addEventListener('click', resetHistory);

    // Export buttons
    document.getElementById('export-json-btn')?.addEventListener('click', () => exportHistoryJSON(evaluationHistory));
    document.getElementById('export-csv-btn')?.addEventListener('click', () => exportHistoryCSV(evaluationHistory));

    // Run evaluation pipeline button
    document.getElementById('run-pipeline-btn')?.addEventListener('click', executeEvaluation);
}

function resetHistory() {
    evaluationHistory = [];
    clearHistoryStorage();
    updateHistoryUI(evaluationHistory);
    updateChart(evaluationHistory);
}

function selectedMethods() {
    const checked = new Set(
        Array.from(
            document.querySelectorAll('input[name="perturbation-type"]:checked'),
            input => input.value
        )
    );

    return Object.keys(PERTURBATION_LABELS).filter(method => checked.has(method));
}

function updatePerturbationControls() {
    const methods = selectedMethods();
    const separate = document.getElementById('evaluation-mode').value === 'separate';

    document.querySelectorAll('[data-method-settings]').forEach(panel => {
        panel.hidden = !methods.includes(panel.dataset.methodSettings);

        panel.querySelectorAll('input, select').forEach(input => {
            input.disabled = panel.hidden || evaluationRunning;
        });
    });

    const emojiInsertion =
        methods.includes('emoji') &&
        document.getElementById('emoji-style').value !== 'substitute';

    document.getElementById('emoji-position-container').hidden = !emojiInsertion;
    document.getElementById('emoji-position').disabled = !emojiInsertion || evaluationRunning;

    document.getElementById('perturbation-order').textContent = separate
        ? 'Each selected method starts independently from the clean input.'
        : `Application order: ${methodLabel(methods).replaceAll(' + ', ' -> ') || 'Select at least one method'}`;

    const variants = separate ? methods.length : 1;

    document.getElementById('evaluation-workload').textContent = methods.length
        ? `1 shared baseline + ${variants} variant${variants === 1 ? '' : 's'} (${variants + 1} answer generations).${methods.includes('rephrase') ? ' Rephrasing may add a model call.' : ''}`
        : 'Select at least one perturbation.';

    document.getElementById('run-pipeline-btn').disabled =
        evaluationRunning || !methods.length;
}

function generationLabel(source) {
    return source === 'synthetic'
        ? 'Synthetic / scripted output'
        : source || 'Unknown';
}

function validateEvaluationInputs(methods) {
    // Only selected methods contribute required fields. Either group can run alone.
    const inputs = [
        document.getElementById('top-k'),
        document.getElementById('perturbation-seed')
    ];

    document.querySelectorAll('[data-method-settings]').forEach(panel => {
        if (methods.includes(panel.dataset.methodSettings)) {
            inputs.push(...panel.querySelectorAll('input[type="number"]'));
        }
    });

    const invalid = inputs.find(input => !input.checkValidity());

    if (!invalid) return true;

    const label =
        invalid.labels?.[0]?.textContent.trim() ||
        (invalid.id === 'top-k' ? 'Top K' : invalid.id);

    const error = document.getElementById('evaluation-error');
    error.textContent = `${label}: ${invalid.validationMessage}`;
    error.hidden = false;

    document.getElementById('evaluation-progress').textContent =
        'Check the highlighted setting.';

    invalid.reportValidity();

    return false;
}

function renderBaseline(result) {
    document.getElementById('baseline-query-text').textContent =
        result.baseline_query;

    renderContexts(
        'baseline-contexts-list',
        result.baseline_contexts || []
    );

    document.getElementById('baseline-doc-count').textContent =
        (result.baseline_contexts || []).length;

    document.getElementById('baseline-answer-text').textContent =
        result.baseline_answer;
}

function renderVariant(result) {
    if (!result) return;

    renderBaseline(result);

    document.getElementById('perturbed-query-text').textContent =
        result.perturbed_query;

    document.getElementById('perturbed-answer-text').textContent =
        result.perturbed_answer;

    const tag = document.getElementById('perturbed-tag');

    tag.textContent =
        result.methods.length > 1
            ? `${result.methods.length} methods combined`
            : methodLabel(result.methods);

    tag.title = methodLabel(result.methods);

    renderContexts(
        'perturbed-contexts-list',
        result.perturbed_contexts || []
    );

    renderContexts(
        'perturbed-retrieved-list',
        result.perturbed_retrieved_contexts || []
    );

    document.getElementById('perturbed-doc-count').textContent =
        (result.perturbed_contexts || []).length;

    document.getElementById('generation-sources').textContent =
        `Baseline: ${generationLabel(result.baseline_generation_source)}. Variant: ${generationLabel(result.perturbed_generation_source)}.`;

    const warning = document.getElementById('evaluation-warning');

    warning.textContent = result.warning || '';
    warning.hidden = !result.warning;

    const changes = document.getElementById('perturbation-changes');
    changes.replaceChildren();

    result.changes.forEach(change => {
        const item = document.createElement('details');
        item.className =
            'text-xs text-slate-300 bg-slate-950 p-2 rounded';

        const summary = document.createElement('summary');
        summary.textContent = `${PERTURBATION_LABELS[change.method] || change.method}: ${change.status}: ${change.summary}`;
        const detail = document.createElement('pre');

        detail.className =
            'whitespace-pre-wrap break-words text-[11px] mt-2';

        detail.textContent =
            JSON.stringify(change.details, null, 2);

        item.append(summary, detail);
        changes.appendChild(item);
    });

    // Update UI Telemetry Metric Cards
    const metrics = result.metrics;

    let badgeClass =
        "bg-amber-500/20 text-amber-300 border-amber-500/30";

    if (metrics.status_category === "Absorbed") {
        badgeClass =
            "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    }

    if (metrics.status_category === "Amplified") {
        badgeClass =
            "bg-rose-500/20 text-rose-300 border-rose-500/30";
    }

    updateMetricCards({
        retrievalSim: metrics.retrieval_sim,
        generationSim: metrics.generation_sim,
        amplificationFactor: metrics.amplification_factor,
        statusText: metrics.status_text,
        statusBadgeClass: badgeClass,
        statusDesc: metrics.status_desc,
        query_embedding_shift: metrics.query_embedding_shift,
        retrieval_rank_shift_jaccard: metrics.retrieval_rank_shift_jaccard,
        context_construction_loss: metrics.context_construction_loss,
        generation_vs_gt_baseline: metrics.generation_vs_gt_baseline,
        generation_vs_gt_perturbed: metrics.generation_vs_gt_perturbed
    });
}

export async function executeEvaluation() {
    const runBtn =
        document.getElementById('run-pipeline-btn');

    if (!runBtn || evaluationRunning) return;

    // Resynchronise restored/toggled controls before validating the current selection.
    updatePerturbationControls();

    const methods = selectedMethods();

    if (!methods.length) return;

    if (!validateEvaluationInputs(methods)) return;

    evaluationRunning = true;

    document
        .querySelectorAll('aside input, aside select, aside button')
        .forEach(input => {
            input.disabled = true;
        });

    runBtn.innerHTML =
        '<i class="fa-solid fa-spinner fa-spin"></i> <span>Running Evaluation...</span>';

    document.getElementById('evaluation-progress').textContent =
        'Starting evaluation...';

    const error =
        document.getElementById('evaluation-error');

    error.hidden = true;

    document.getElementById('evaluation-warning').hidden = true;

    document.getElementById('experiment-summary').textContent = '';
    document.getElementById('generation-sources').textContent = '';
    document.getElementById('perturbation-changes').replaceChildren();

    const selector =
        document.getElementById('variant-select');

    selector.replaceChildren();
    selector.disabled = true;

    currentVariants = [];

    try {
        const customQuery =
            document.getElementById('custom-query-input').value.trim();

        const seedText =
            document.getElementById('perturbation-seed').value;

        const result =
            await executeBackendEvaluation({
                questionId: currentQueryObj.id,
                customQuery: customQuery || null,
                baselineQuery: customQuery || currentQueryObj.question,
                groundTruth: currentQueryObj.ground_truth,
                contexts: currentQueryObj.contexts,
                supportingTitles:
                    customQuery
                        ? null
                        : currentQueryObj.supporting_titles,

                perturbationTypes: methods,

                mode:
                    document.getElementById('evaluation-mode').value,

                settings: {
                    ...Object.fromEntries(
                        [
                            ['remove_evidence', 'remove_count', 'remove-count'],
                            ['irrelevant', 'irrelevant_count', 'irrelevant-count'],
                            ['alter_facts', 'fact_count', 'fact-count'],
                            ['truncate', 'retain_percent', 'retain-percent']
                        ]
                            .filter(([method]) =>
                                methods.includes(method)
                            )
                            .map(([, key, id]) => [
                                key,
                                Number(
                                    document.getElementById(id).value
                                )
                            ])
                    ),

                    ...(methods.includes('emoji')
                        ? {
                            emoji_style:
                                document.getElementById('emoji-style').value,

                            emoji_count:
                                Number(
                                    document.getElementById('emoji-count').value
                                ),

                            emoji_position:
                                document.getElementById('emoji-position').value
                        }
                        : {}),

                    ...(methods.includes('metadata_injection')
                        ? {
                            metadata_term_count:
                                Number(
                                    document.getElementById('metadata-term-count').value
                                )
                        }
                        : {})
                },

                seed:
                    seedText === ''
                        ? null
                        : Number(seedText),

                typoRate:
                    Number(
                        document.getElementById('typo-rate-slider').value
                    ) / 100,

                embeddingProvider:
                    document.getElementById('embedding-provider').value,

                searchType:
                    document.getElementById('search-type').value,

                topK:
                    Number(
                        document.getElementById('top-k').value
                    ),

                llmModel:
                    document.getElementById('llm-model').value,

                onEvent: event => {
                    if (event.event === 'progress') {
                        document.getElementById(
                            'evaluation-progress'
                        ).textContent = event.message;
                    }

                    if (event.event === 'baseline') {
                        renderBaseline(event);
                    }

                    if (event.event === 'variant') {
                        currentVariants.push(event.variant);

                        const option =
                            document.createElement('option');

                        option.value =
                            currentVariants.length - 1;

                        option.textContent =
                            methodLabel(event.variant.methods);

                        selector.appendChild(option);

                        selector.disabled = false;

                        if (currentVariants.length === 1) {
                            renderVariant(event.variant);
                        }
                    }
                }
            });

        document.getElementById('experiment-summary').textContent =
            `${result.mode === 'separate' ? 'Separate comparisons' : 'Combined'} | Seed ${result.seed} | Shared baseline ${result.evaluation_id.slice(0, 8)}`;

        document.getElementById('evaluation-progress').textContent =
            `Complete: ${result.variants.length} variant${result.variants.length === 1 ? '' : 's'}.`;

        for (const variant of result.variants) {
            evaluationHistory.push({
                id: evaluationHistory.length + 1,
                type: methodLabel(variant.methods),
                retrievalSim:
                    variant.metrics.retrieval_sim,
                generationSim:
                    variant.metrics.generation_sim,
                amplificationFactor:
                    variant.metrics.amplification_factor,
                statusCategory:
                    variant.metrics.status_category,
                status:
                    variant.metrics.status_text,
                evaluationId:
                    result.evaluation_id,
                variantId:
                    variant.variant_id,
                mode:
                    result.mode,
                selectedMethods:
                    result.selected_methods,
                methods:
                    variant.methods,
                settings:
                    result.settings,
                seed:
                    result.seed,
                questionId:
                    currentQueryObj.id,
                embeddingProvider:
                    document.getElementById('embedding-provider').value,
                llmModel:
                    document.getElementById('llm-model').value,
                searchType:
                    document.getElementById('search-type').value,
                topK:
                    Number(
                        document.getElementById('top-k').value
                    ),
                result:
                    variant
            });
        }

        saveHistoryToStorage(evaluationHistory);

        updateHistoryUI(evaluationHistory);
        updateChart(evaluationHistory);

    } catch (err) {
        error.textContent = err.message;
        error.hidden = false;

        document.getElementById('evaluation-progress').textContent =
            'Evaluation stopped before completion.';

        console.error(
            'Evaluation pipeline execution error:',
            err
        );

    } finally {
        evaluationRunning = false;

        document
            .querySelectorAll('aside input, aside select, aside button')
            .forEach(input => {
                input.disabled = false;
            });

        runBtn.innerHTML =
            '<i class="fa-solid fa-play"></i> <span>Run Evaluation Pipeline</span>';

        updatePerturbationControls();
    }
}