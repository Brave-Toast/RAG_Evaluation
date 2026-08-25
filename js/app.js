/**
 * Main Application Orchestrator & Event Controller
 * Connects UI control inputs to the FastAPI Backend for true RAG evaluation and telemetry analysis.
 */

import { HOTPOTQA_DATASET, loadDatasetFromBackend, getQueryByIndex } from './dataset.js';
import { executeBackendEvaluation, checkBackendHealth } from './ragEngine.js';
import { loadHistoryFromStorage, saveHistoryToStorage, clearHistoryStorage, exportHistoryJSON, exportHistoryCSV } from './storage.js';
import { renderContexts, updateMetricCards, updateHistoryUI, initChart, updateChart } from './uiManager.js';

let evaluationHistory = [];
let currentQueryObj = HOTPOTQA_DATASET[0];

window.addEventListener('DOMContentLoaded', async () => {
    // 1. Perform backend health check & update header status badge
    updateHeaderStatusBadge();
    
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
    
    // 5. Run initial baseline evaluation
    executeEvaluation();
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

export async function executeEvaluation() {
    const runBtn = document.getElementById('run-pipeline-btn');
    if (!runBtn) return;
    
    runBtn.disabled = true;
    runBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Running Evaluation...</span>`;

    try {
        // Collect UI parameters
        const customQuery = document.getElementById('custom-query-input').value.trim();
        const activeQueryText = customQuery || currentQueryObj.question;
        const groundTruth = currentQueryObj.ground_truth;
        
        const perturbationType = document.querySelector('input[name="perturbation-type"]:checked')?.value || 'spelling';
        const typoRate = parseFloat(document.getElementById('typo-rate-slider').value) / 100;
        const llmModelChoice = document.getElementById('llm-model').value;
        const embeddingProvider = document.getElementById('embedding-provider')?.value || 'sentence-transformers/all-mpnet-base-v2';
        const searchType = document.getElementById('search-type').value;
        const topK = parseInt(document.getElementById('top-k').value, 10) || 2;

        // Update perturbation visual tag
        const perturbedTagEl = document.getElementById('perturbed-tag');
        if (perturbedTagEl) {
            if (perturbationType === 'spelling') {
                perturbedTagEl.textContent = `Spelling Typo (${Math.round(typoRate * 100)}%)`;
            } else if (perturbationType === 'rephrase') {
                perturbedTagEl.textContent = `LLM Query Rephrase`;
            } else if (perturbationType === 'conflict') {
                perturbedTagEl.textContent = `Conflicting Context Injected`;
            }
        }

        // Call FastAPI Backend Evaluation Endpoint (/api/run-evaluation)
        const result = await executeBackendEvaluation({
            questionId: currentQueryObj.id,
            customQuery: customQuery || null,
            baselineQuery: activeQueryText,
            groundTruth: groundTruth,
            contexts: currentQueryObj.contexts,
            perturbationType: perturbationType,
            typoRate: typoRate,
            embeddingProvider: embeddingProvider,
            searchType: searchType,
            topK: topK,
            llmModel: llmModelChoice
        });

        // 1. Render Baseline vs Perturbed Queries
        document.getElementById('baseline-query-text').textContent = result.baseline_query;
        document.getElementById('perturbed-query-text').textContent = result.perturbed_query;

        // 2. Render Retrieved Document Lists
        renderContexts('baseline-contexts-list', result.baseline_contexts || []);
        renderContexts('perturbed-contexts-list', result.perturbed_contexts || []);
        document.getElementById('baseline-doc-count').textContent = (result.baseline_contexts || []).length;
        document.getElementById('perturbed-doc-count').textContent = (result.perturbed_contexts || []).length;

        // 3. Render Baseline vs Perturbed Generated Answers
        document.getElementById('baseline-answer-text').textContent = result.baseline_answer;
        document.getElementById('perturbed-answer-text').textContent = result.perturbed_answer;

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
            generation_vs_gt_perturbed: metrics.generation_vs_gt_perturbed
        });

        // Log run record to evaluation history & update Chart
        const runRecord = {
            id: evaluationHistory.length + 1,
            type: perturbationType,
            retrievalSim: metrics.retrieval_sim,
            generationSim: metrics.generation_sim,
            amplificationFactor: metrics.amplification_factor,
            statusCategory: metrics.status_category,
            status: metrics.status_text
        };

        evaluationHistory.push(runRecord);
        saveHistoryToStorage(evaluationHistory);
        updateHistoryUI(evaluationHistory);
        updateChart(evaluationHistory);

        if (result.warning) {
            console.warn("Backend evaluation warning:", result.warning);
        }

    } catch (err) {
        console.error("Evaluation pipeline execution error:", err);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `<i class="fa-solid fa-play"></i> <span>Run Evaluation Pipeline</span>`;
    }
}
