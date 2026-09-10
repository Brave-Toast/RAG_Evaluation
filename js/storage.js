/**
 * Benchmark History LocalStorage & Export Manager
 */

const STORAGE_KEY = 'rag_evaluation_benchmark_history';

/**
 * Loads saved evaluation history from localStorage.
 * @returns {Array}
 */
export function loadHistoryFromStorage() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        const history = stored ? JSON.parse(stored) : [];
        return Array.isArray(history) ? history : [];
    } catch (e) {
        console.warn("Failed to load history from localStorage:", e);
        return [];
    }
}

/**
 * Saves evaluation history to localStorage.
 * @param {Array} history 
 */
export function saveHistoryToStorage(history) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        console.warn("Failed to save history to localStorage:", e);
    }
}

/**
 * Clears saved evaluation history from localStorage.
 */
export function clearHistoryStorage() {
    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
        console.warn("Failed to clear history in localStorage:", e);
    }
}

/**
 * Exports history array as downloadable JSON file.
 * @param {Array} history 
 */
export function exportHistoryJSON(history) {
    if (!history || history.length === 0) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `rag_evaluation_history_${new Date().toISOString().slice(0,10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

/**
 * Exports history array as downloadable CSV file.
 * @param {Array} history 
 */
export function exportHistoryCSV(history) {
    if (!history || history.length === 0) return;
    const link = document.createElement("a");
    link.setAttribute("href", 'data:text/csv;charset=utf-8,' + encodeURIComponent(historyCSV(history)));
    link.setAttribute("download", `rag_evaluation_history_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
}

export function historyCSV(history) {
    const headers = ['Run ID', 'Perturbation Type', 'Final Context Cosine Sim', 'Generation Cosine Sim',
        'Amplification Factor', 'Status', 'Evaluation ID', 'Variant ID', 'Mode', 'Selected Methods',
        'Variant Methods', 'Settings', 'Seed', 'Question ID', 'Embedding Provider', 'LLM Model',
        'Search Type', 'Top K', 'Result JSON'];
    const rows = history.map(h => [h.id, h.type, h.retrievalSim, h.generationSim,
        h.amplificationFactor ?? 'N/A', h.status, h.evaluationId, h.variantId, h.mode,
        h.selectedMethods, h.methods, h.settings, h.seed, h.questionId, h.embeddingProvider,
        h.llmModel, h.searchType, h.topK, h.result]);
    const cell = value => '"' + (typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '')).replaceAll('"', '""') + '"';
    return [headers, ...rows].map(row => row.map(cell).join(',')).join('\r\n');
}
