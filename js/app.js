/**
 * Main Application Orchestrator & Event Controller
 * Connects UI control inputs to the FastAPI Backend for true RAG evaluation and telemetry analysis.
 */

import { HOTPOTQA_DATASET, loadDatasetFromBackend, getQueryByIndex } from './dataset.js';
import { executeBackendEvaluation, checkBackendHealth, fetchRepairOptions, fetchRerankerOptions } from './ragEngine.js';
import { loadHistoryFromStorage, saveHistoryToStorage, clearHistoryStorage, exportHistoryJSON, exportHistoryCSV } from './storage.js';
import { PERTURBATION_LABELS, methodLabel } from './perturbations.js';
import { renderContexts, updateMetricCards, updateHistoryUI, initChart, updateChart, renderChips, openPicker, closePicker } from './uiManager.js';

//true when the llm judge is on n says the perturbed anwser is correct, used so the card n history agree
const judgeRight=m=>Boolean(m&&m.use_llm_judge&&typeof m.judge_correctness_perturbed==='number'&&m.judge_correctness_perturbed>=0.5);

let evaluationHistory = [];
let currentVariants = [];
let evaluationRunning = false;
let currentQueryObj = HOTPOTQA_DATASET[0];

// ---- Selection state (the sidebar chips are drawn from this, not from checkboxes) ----
const QUESTION_GROUP = 'Question changes (before retrieval)';
const DOCUMENT_GROUP = 'Document changes';

// What each perturbation does, shown in its popup. Settings fields live in index.html (#detail-dialog).
const PERTURBATION_INFO = {
    spelling: { name: 'Spelling errors', short: 'Spelling', group: QUESTION_GROUP,
        description: 'Adds random typos to the question: swapped letters, dropped letters and substitutions. Typo severity sets how heavy the typos are.' },
    rephrase: { name: 'Query rephrasing', short: 'Rephrase', group: QUESTION_GROUP,
        description: 'Rewords the question with the selected LLM while keeping its meaning. May add one extra model call.' },
    emoji: { name: 'Emoji perturbation', short: 'Emoji', group: QUESTION_GROUP,
        description: 'Inserts emoji into the question, or encodes words and dates as emoji. The edit limit caps how many changes are made.' },
    negation_flip: { name: 'Negation flip', short: 'Negation', group: QUESTION_GROUP,
        description: 'Inserts or removes a "not" so the polarity of the question is flipped.' },
    entity_substitution: { name: 'Entity substitution', short: 'Entity swap', group: QUESTION_GROUP,
        description: 'Swaps a known entity in the question for an unrelated one.' },
    conflict: { name: 'Conflicting context', short: 'Conflict', group: DOCUMENT_GROUP,
        description: 'Adds a contradictory candidate document before retrieval, so it competes with the real evidence.' },
    metadata_injection: { name: 'Metadata injection', short: 'Metadata', group: DOCUMENT_GROUP,
        description: 'Before retrieval, tags an off-topic passage with keywords taken from the question so that it may be retrieved.' },
    remove_evidence: { name: 'Remove supporting evidence', short: 'Remove evidence', group: DOCUMENT_GROUP,
        description: 'After retrieval, removes passages from the context. Falls back to the highest-ranked passage.' },
    irrelevant: { name: 'Add irrelevant passages', short: 'Irrelevant', group: DOCUMENT_GROUP,
        description: 'After retrieval, adds passages sampled from other benchmark questions.' },
    alter_facts: { name: 'Alter facts', short: 'Alter facts', group: DOCUMENT_GROUP,
        description: 'After retrieval, replaces numbers and date tokens inside the passages.' },
    shuffle: { name: 'Shuffle passage order', short: 'Shuffle', group: DOCUMENT_GROUP,
        description: 'After retrieval, changes the order of the passages. The passage contents are preserved.' },
    truncate: { name: 'Truncate passages', short: 'Truncate', group: DOCUMENT_GROUP,
        description: 'After retrieval, keeps only a percentage of the text of the passages.' }
};

const selectedPerturbations = new Set(['spelling']); // spelling was ticked by default before the redesign
const selectedRepairNames = new Set();
let repairCatalog = [];            // [{name, description}] from the backend
let settingValues = {};            // last confirmed value of every settings field, keyed by element id
let detailTarget = null;           // {kind: 'perturbation' | 'repair', id} while the detail popup is open

window.addEventListener('DOMContentLoaded', async () => {
    captureSettingDefaults();

    // 1. Perform backend health check & update header status badge
    updateHeaderStatusBadge();
    updateEmbeddingProviderAvailability();
    updateLLMProviderAvailability();
    updateOllamaAvailability();
    
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
    await loadRepairOptions();
    await loadRerankerOptions();
    updatePerturbationControls();
    
    /***
     5. Run initial baseline evaluation
     
    executeEvaluation();
    ***/
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

/**
 * Disables the local Ollama embedding option when the Ollama service is not reachable,
 * so a "nomic" run cannot be started against a service that is down.
 */
async function updateOllamaAvailability() {
    const health = await checkBackendHealth();
    const offline = !(health && health.ollama_support);
    document.querySelectorAll('#embedding-provider option[value^="ollama/"]').forEach(opt => {
        const base = opt.textContent.replace(/ \(Ollama not running\)$/, '');
        opt.disabled = offline;
        opt.textContent = offline ? `${base} (Ollama not running)` : base;
    });
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

    document.getElementById('perturbation-controls').addEventListener('change', updatePerturbationControls);
    bindSelectionPopups();
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
    return Object.keys(PERTURBATION_LABELS).filter(method => selectedPerturbations.has(method));
}

function repairLabel(name) {
    const repair = repairCatalog.find(item => item.name === name);
    return repair?.label || name;
}

function selectedRepairs() {
    return repairCatalog.map(repair => repair.name).filter(name => selectedRepairNames.has(name));
}

// Repairs come from backend/repairs.py, so adding one there needs no HTML change.
async function loadRepairOptions() {
    repairCatalog = await fetchRepairOptions();
    const known = new Set(repairCatalog.map(repair => repair.name));
    [...selectedRepairNames].forEach(name => { if (!known.has(name)) selectedRepairNames.delete(name); });
    updatePerturbationControls();
}

//fills the reranker dropdown from /api/rerankers, same idea as loadRepairOptions
async function loadRerankerOptions() {
    const select = document.getElementById('reranker-model');
    if (!select) return;
    const data = await fetchRerankerOptions();
    select.replaceChildren(...data.rerankers.map(r => {
        const opt = document.createElement('option');
        opt.value = r.name;
        opt.textContent = r.label || r.name;
        return opt;
    }));
    if (data.default) select.value = data.default;  //start on the recomended one (bge base)
    document.getElementById('use-reranker')?.addEventListener('change', syncRerankerControls);
    syncRerankerControls();
}

//dropdown only usable wen the checkbox is ticked, n never while a run is going
function syncRerankerControls() {
    const box = document.getElementById('use-reranker');
    const select = document.getElementById('reranker-model');
    if (!box || !select) return;
    box.disabled = evaluationRunning || !select.options.length;  //no models loaded = backend offline, so cant turn it on
    select.disabled = evaluationRunning || !box.checked;
}

/* ---------- popups: catalog -> detail (description + settings) ---------- */

function settingInputs() {
    return Array.from(document.querySelectorAll('[data-method-settings] input, [data-method-settings] select'));
}

// The default values written in index.html become the starting settings.
function captureSettingDefaults() {
    settingValues = Object.fromEntries(settingInputs().map(input => [input.id, input.value]));
}

function activePanel() {
    return Array.from(document.querySelectorAll('[data-method-settings]')).find(panel => !panel.hidden) || null;
}

function updateEmojiPositionVisibility() {
    const container = document.getElementById('emoji-position-container');
    if (container) container.hidden = document.getElementById('emoji-style').value === 'substitute';
}

function chipDetail(method) {
    const value = id => settingValues[id];
    switch (method) {
        case 'spelling': return `${value('typo-rate-slider')}%`;
        case 'emoji': return `${value('emoji-style')}, ${value('emoji-count')}`;
        case 'metadata_injection': return `${value('metadata-term-count')} keywords`;
        case 'remove_evidence': return `-${value('remove-count')}`;
        case 'irrelevant': return `+${value('irrelevant-count')}`;
        case 'alter_facts': return `${value('fact-count')}`;
        case 'truncate': return `${value('retain-percent')}%`;
        default: return '';
    }
}

function openPerturbationPicker() {
    const groups = [QUESTION_GROUP, DOCUMENT_GROUP].map(label => ({
        label,
        items: Object.keys(PERTURBATION_LABELS)
            .filter(method => PERTURBATION_INFO[method]?.group === label)
            .map(method => ({
                id: method,
                title: PERTURBATION_INFO[method].name,
                description: PERTURBATION_INFO[method].description,
                selected: selectedPerturbations.has(method)
            }))
    }));
    openPicker({ title: 'Add a perturbation', groups, onPick: id => openDetail('perturbation', id) });
}

function openRepairPicker() {
    openPicker({
        title: 'Add a repair',
        groups: [{
            label: 'Runs after retrieval, before the answer is generated',
            items: repairCatalog.map(repair => ({
                id: repair.name,
                title: repair.label || repair.name,
                description: repair.description,
                selected: selectedRepairNames.has(repair.name)
            }))
        }],
        onPick: id => openDetail('repair', id)
    });
}

function openDetail(kind, id) {
    const isPerturbation = kind === 'perturbation';
    const info = isPerturbation ? PERTURBATION_INFO[id] : repairCatalog.find(repair => repair.name === id);
    if (!info) return;
    detailTarget = { kind, id };
    const selected = isPerturbation ? selectedPerturbations.has(id) : selectedRepairNames.has(id);

    document.getElementById('detail-kind').textContent = isPerturbation ? 'Perturbation' : 'Repair';
    document.getElementById('detail-title').textContent = isPerturbation ? info.name : (info.label || info.name);
    document.getElementById('detail-description').textContent = info.description || 'No description provided.';

    // Show only this item's settings, filled with its last confirmed values.
    let hasSettings = false;
    document.querySelectorAll('[data-method-settings]').forEach(panel => {
        const show = isPerturbation && panel.dataset.methodSettings === id;
        panel.hidden = !show;
        if (!show) return;
        hasSettings = true;
        panel.querySelectorAll('input, select').forEach(input => { input.value = settingValues[input.id]; });
    });
    document.getElementById('typo-rate-val').textContent = document.getElementById('typo-rate-slider').value + '%';
    updateEmojiPositionVisibility();
    document.getElementById('detail-no-settings').hidden = hasSettings;
    document.getElementById('detail-error').hidden = true;
    document.getElementById('detail-remove').hidden = !selected;
    document.getElementById('detail-confirm').textContent = selected ? 'Save changes' : 'Add to selection';

    closePicker();
    const dialog = document.getElementById('detail-dialog');
    if (!dialog.open) dialog.showModal();
}

function closeDetail() {
    detailTarget = null;
    const dialog = document.getElementById('detail-dialog');
    if (dialog.open) dialog.close();
}

function confirmDetail(event) {
    event.preventDefault();
    if (!detailTarget) return;
    const { kind, id } = detailTarget;
    if (kind === 'perturbation') {
        const panel = activePanel();
        if (panel) {
            const invalid = Array.from(panel.querySelectorAll('input')).find(input => !input.checkValidity());
            if (invalid) {
                const label = invalid.labels?.[0]?.textContent.trim() || invalid.id;
                const error = document.getElementById('detail-error');
                error.textContent = `${label}: ${invalid.validationMessage}`;
                error.hidden = false;
                invalid.reportValidity();
                return;
            }
            panel.querySelectorAll('input, select').forEach(input => { settingValues[input.id] = input.value; });
        }
        selectedPerturbations.add(id);
    } else {
        selectedRepairNames.add(id);
    }
    closeDetail();
    updatePerturbationControls();
}

function removeSelection(kind, id) {
    (kind === 'perturbation' ? selectedPerturbations : selectedRepairNames).delete(id);
    updatePerturbationControls();
}

function bindSelectionPopups() {
    document.getElementById('add-perturbation-btn')?.addEventListener('click', openPerturbationPicker);
    document.getElementById('add-repair-btn')?.addEventListener('click', openRepairPicker);
    document.getElementById('picker-close')?.addEventListener('click', closePicker);
    document.getElementById('detail-back')?.addEventListener('click', closeDetail);
    document.getElementById('detail-form')?.addEventListener('submit', confirmDetail);
    document.getElementById('detail-remove')?.addEventListener('click', () => {
        if (!detailTarget) return;
        const { kind, id } = detailTarget;
        closeDetail();
        removeSelection(kind, id);
    });
    document.getElementById('typo-rate-slider')?.addEventListener('input', event => {
        document.getElementById('typo-rate-val').textContent = event.target.value + '%';
    });
    document.getElementById('emoji-style')?.addEventListener('change', updateEmojiPositionVisibility);
    // Clicking the dark backdrop closes a popup (the card fills the dialog, so only the backdrop hits the dialog itself).
    ['picker-dialog', 'detail-dialog'].forEach(id => {
        const dialog = document.getElementById(id);
        dialog?.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    });
    document.getElementById('detail-dialog')?.addEventListener('close', () => { detailTarget = null; });
}

function updatePerturbationControls() {
    const methods = selectedMethods();
    const repairs = selectedRepairs();
    const separate = document.getElementById('evaluation-mode').value === 'separate';

    renderChips('perturbation-chips', 'perturbation-empty',
        methods.map(method => {
            const detail = chipDetail(method);
            return { id: method, label: detail ? `${PERTURBATION_INFO[method].short} (${detail})` : PERTURBATION_INFO[method].short,
                title: `Edit ${PERTURBATION_INFO[method].name}` };
        }),
        { disabled: evaluationRunning, onEdit: id => openDetail('perturbation', id), onRemove: id => removeSelection('perturbation', id) });
    renderChips('repair-chips', 'repair-empty',
        repairs.map(name => { const label = repairLabel(name); return { id: name, label, title: `Edit ${label}` }; }),
        { disabled: evaluationRunning, onEdit: id => openDetail('repair', id), onRemove: id => removeSelection('repair', id) });
    // Nothing to add when the backend registers no repairs.
    const addRepair = document.getElementById('add-repair-btn');
    addRepair.disabled = evaluationRunning || !repairCatalog.length;
    addRepair.title = repairCatalog.length ? '' : 'No repairs registered yet. Add one in backend/repairs.py and it appears here.';
    document.getElementById('add-perturbation-btn').disabled = evaluationRunning;

    document.getElementById('perturbation-order').textContent = separate
        ? 'Each selected method starts independently from the clean input.'
        : `Application order: ${methodLabel(methods).replaceAll(' + ', ' -> ') || 'Select at least one method'}`;
    const variants = separate ? methods.length : 1;
    document.getElementById('evaluation-workload').textContent = methods.length
        ? `1 shared baseline + ${variants} variant${variants === 1 ? '' : 's'} (${variants + 1} answer generations).${methods.includes('rephrase') ? ' Rephrasing may add a model call.' : ''}`
        : 'Select at least one perturbation.';
    document.getElementById('run-pipeline-btn').disabled = evaluationRunning || !methods.length;
}

// Settings sent to the backend: same keys as before the redesign, only for the selected methods.
function buildMethodSettings(methods) {
    const number = id => Number(settingValues[id]);
    return {
        ...Object.fromEntries([
            ['remove_evidence', 'remove_count', 'remove-count'],
            ['irrelevant', 'irrelevant_count', 'irrelevant-count'],
            ['alter_facts', 'fact_count', 'fact-count'],
            ['truncate', 'retain_percent', 'retain-percent']
        ].filter(([method]) => methods.includes(method)).map(([, key, id]) => [key, number(id)])),
        ...(methods.includes('emoji') ? {
            emoji_style: settingValues['emoji-style'],
            emoji_count: number('emoji-count'),
            emoji_position: settingValues['emoji-position']
        } : {}),
        ...(methods.includes('metadata_injection') ? {
            metadata_term_count: number('metadata-term-count')
        } : {})
    };
}

function generationLabel(source) {
    return source === 'synthetic' ? 'Synthetic / scripted output' : source || 'Unknown';
}

function validateEvaluationInputs() {
    // Method settings were validated when they were confirmed in their popup; only the shared fields remain.
    const inputs = [document.getElementById('top-k'), document.getElementById('perturbation-seed')];
    const invalid = inputs.find(input => !input.checkValidity());
    if (!invalid) return true;
    const label = invalid.labels?.[0]?.textContent.trim() || (invalid.id === 'top-k' ? 'Top K' : invalid.id);
    const error = document.getElementById('evaluation-error');
    error.textContent = `${label}: ${invalid.validationMessage}`;
    error.hidden = false;
    document.getElementById('evaluation-progress').textContent = 'Check the highlighted setting.';
    invalid.reportValidity();
    return false;
}

function renderBaseline(result) {
    document.getElementById('baseline-query-text').textContent = result.baseline_query;
    renderContexts('baseline-contexts-list', result.baseline_contexts || []);
    document.getElementById('baseline-doc-count').textContent = (result.baseline_contexts || []).length;
    document.getElementById('baseline-answer-text').textContent = result.baseline_answer;
}

function renderVariant(result) {
    if (!result) return;
    renderBaseline(result);
    document.getElementById('perturbed-query-text').textContent = result.perturbed_query;
    document.getElementById('perturbed-answer-text').textContent = result.perturbed_answer;
    const tag = document.getElementById('perturbed-tag');
    tag.textContent = result.methods.length > 1 ? `${result.methods.length} methods combined` : methodLabel(result.methods);
    tag.title = methodLabel(result.methods);
    renderContexts('perturbed-contexts-list', result.perturbed_contexts || []);
    renderContexts('perturbed-retrieved-list', result.perturbed_retrieved_contexts || []);
    document.getElementById('perturbed-doc-count').textContent = (result.perturbed_contexts || []).length;
    document.getElementById('generation-sources').textContent = `Baseline: ${generationLabel(result.baseline_generation_source)}. Variant: ${generationLabel(result.perturbed_generation_source)}.`;
    const warning = document.getElementById('evaluation-warning');
    warning.textContent = result.warning || '';
    warning.hidden = !result.warning;
    const changes = document.getElementById('perturbation-changes');
    changes.replaceChildren();
    result.changes.forEach(change => {
        const item = document.createElement('details');
        item.className = 'text-xs text-slate-300 bg-slate-950 p-2 rounded';
        const summary = document.createElement('summary');
        summary.textContent = `${PERTURBATION_LABELS[change.method] || change.method}: ${change.status}: ${change.summary}`;
        const detail = document.createElement('pre');
        detail.className = 'whitespace-pre-wrap break-words text-[11px] mt-2';
        detail.textContent = JSON.stringify(change.details, null, 2);
        item.append(summary, detail);
        changes.appendChild(item);
    });
    // 4. Update UI Telemetry Metric Cards
    const metrics = result.metrics;
    let badgeClass = "bg-amber-500/20 text-amber-300 border-amber-500/30";
    if (metrics.status_category === "Absorbed") badgeClass = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    if (metrics.status_category === "Amplified") badgeClass = "bg-rose-500/20 text-rose-300 border-rose-500/30";

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
        generation_vs_gt_perturbed: metrics.generation_vs_gt_perturbed,
        context_recall_perturbed: metrics.context_recall_perturbed,
        answer_f1_perturbed: metrics.answer_f1_perturbed,
        failure_attribution: metrics.failure_attribution,
        failure_root_cause: metrics.failure_root_cause,
        failure_localisation: result.failure_localisation,
        use_llm_judge: metrics.use_llm_judge,
        judge_correctness_perturbed: metrics.judge_correctness_perturbed
    });

}

export async function executeEvaluation() {
    const runBtn = document.getElementById('run-pipeline-btn');
    if (!runBtn || evaluationRunning) return;
    // Resynchronise restored/toggled controls before validating the current selection.
    updatePerturbationControls();
    const methods = selectedMethods();
    if (!methods.length) return;
    if (!validateEvaluationInputs()) return;

    evaluationRunning = true;
    document.querySelectorAll('aside input, aside select, aside button').forEach(input => { input.disabled = true; });
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Running Evaluation...</span>';
    document.getElementById('evaluation-progress').textContent = 'Starting evaluation...';
    const error = document.getElementById('evaluation-error');
    error.hidden = true;
    document.getElementById('evaluation-warning').hidden = true;
    document.getElementById('experiment-summary').textContent = '';
    document.getElementById('generation-sources').textContent = '';
    document.getElementById('perturbation-changes').replaceChildren();
    const selector = document.getElementById('variant-select');
    selector.replaceChildren();
    selector.disabled = true;
    currentVariants = [];
    try {
        const customQuery = document.getElementById('custom-query-input').value.trim();
        const seedText = document.getElementById('perturbation-seed').value;
        const result = await executeBackendEvaluation({
            questionId: currentQueryObj.id,
            customQuery: customQuery || null,
            baselineQuery: customQuery || currentQueryObj.question,
            groundTruth: currentQueryObj.ground_truth,
            contexts: currentQueryObj.contexts,
            goldTitles: currentQueryObj.gold_titles || [],
            supportingTitles: customQuery ? null : currentQueryObj.supporting_titles,
            perturbationTypes: methods,
            repairMethods: selectedRepairs(),
            mode: document.getElementById('evaluation-mode').value,
            settings: buildMethodSettings(methods),
            seed: seedText === '' ? null : Number(seedText),
            typoRate: Number(settingValues['typo-rate-slider']) / 100,
            embeddingProvider: document.getElementById('embedding-provider').value,
            searchType: document.getElementById('search-type').value,
            topK: Number(document.getElementById('top-k').value),
            useReranker: document.getElementById('use-reranker').checked,
            rerankerModel: document.getElementById('reranker-model').value,
            llmModel: document.getElementById('llm-model').value,
            onEvent: event => {
                if (event.event === 'progress') document.getElementById('evaluation-progress').textContent = event.message;
                if (event.event === 'baseline') renderBaseline(event);
                if (event.event === 'variant') {
                    currentVariants.push(event.variant);
                    const option = document.createElement('option');
                    option.value = currentVariants.length - 1;
                    option.textContent = methodLabel(event.variant.methods);
                    selector.appendChild(option);
                    selector.disabled = false;
                    if (currentVariants.length === 1) renderVariant(event.variant);
                }
            }
        });
        document.getElementById('experiment-summary').textContent = `${result.mode === 'separate' ? 'Separate comparisons' : 'Combined'} | Seed ${result.seed} | Shared baseline ${result.evaluation_id.slice(0, 8)}`;
        document.getElementById('evaluation-progress').textContent = `Complete: ${result.variants.length} variant${result.variants.length === 1 ? '' : 's'}.`;
        for (const variant of result.variants) {
            evaluationHistory.push({
                id: evaluationHistory.length + 1,
                type: methodLabel(variant.methods),
                retrievalSim: variant.metrics.retrieval_sim,
                generationSim: variant.metrics.generation_sim,
                amplificationFactor: variant.metrics.amplification_factor,
                statusCategory: variant.metrics.status_category,
                status: variant.metrics.status_text,
                //same rule as the root attribution card so they always match
                failureAttribution: judgeRight(variant.metrics)?"NONE":(variant.failure_localisation?.failed_at||variant.metrics.failure_attribution),
                failureRootCause: judgeRight(variant.metrics)?"LLM judge marked the perturbed answer correct, so no stage failed.":(variant.failure_localisation?.reason||variant.metrics.failure_root_cause),
                contextRecall: variant.metrics.context_recall_perturbed,
                answerF1: variant.metrics.answer_f1_perturbed,
                evaluationId: result.evaluation_id,
                variantId: variant.variant_id,
                mode: result.mode,
                selectedMethods: result.selected_methods,
                methods: variant.methods,
                settings: result.settings,
                seed: result.seed,
                questionId: currentQueryObj.id,
                embeddingProvider: document.getElementById('embedding-provider').value,
                llmModel: document.getElementById('llm-model').value,
                searchType: document.getElementById('search-type').value,
                topK: Number(document.getElementById('top-k').value),
                useReranker: document.getElementById('use-reranker').checked,
                rerankerModel: document.getElementById('use-reranker').checked ? document.getElementById('reranker-model').value : null,
                result: variant
            });
        }
        saveHistoryToStorage(evaluationHistory);
        updateHistoryUI(evaluationHistory);
        updateChart(evaluationHistory);
    } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        document.getElementById('evaluation-progress').textContent = 'Evaluation stopped before completion.';
        console.error('Evaluation pipeline execution error:', err);
    } finally {
        evaluationRunning = false;
        document.querySelectorAll('aside input, aside select, aside button').forEach(input => { input.disabled = false; });
        runBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Run Evaluation Pipeline</span>';
        updatePerturbationControls();
        syncRerankerControls();  //finally re-enables evrything, so grey the dropdown out again if reranker is unticked
    }
}