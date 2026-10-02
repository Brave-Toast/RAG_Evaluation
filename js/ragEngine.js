/**
 * RAG Evaluation Framework Client & Backend API Bridge
 * Connects frontend dashboard directly to Python FastAPI backend (/api/run-evaluation).
 */

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

/**
 * Checks backend health status.
 * @param {string} backendUrl 
 * @returns {Promise<Object>}
 */
export async function checkBackendHealth(backendUrl = DEFAULT_BACKEND_URL) {
    try {
        const response = await fetch(`${backendUrl}/api/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
        if (response.ok) {
            return await response.json();
        }
    } catch (err) {
        console.warn("Backend health check warning:", err);
    }
    return { status: "offline", embeddings_active: false, vector_store: "disconnected" };
}

/**
 * Executes a full baseline vs perturbed RAG pipeline evaluation on the FastAPI backend.
 * Uses true ChromaDB vector retrieval (Similarity / MMR), sentence-transformers,
 * and local Ollama or cloud LLMs.
 * 
 * @param {Object} params 
 * @returns {Promise<Object>} evaluation response containing baseline/perturbed contexts, answers, and telemetry metrics
 */
export async function executeBackendEvaluation(params) {
    const backendUrl = params.backendUrl || DEFAULT_BACKEND_URL;
    const endpoint = `${backendUrl}/api/run-evaluation${params.onEvent ? '/stream' : ''}`;

    const payload = {
        question_id: params.questionId || null,
        custom_query: params.customQuery || null,
        baseline_query: params.baselineQuery,
        ground_truth: params.groundTruth,
        contexts: params.contexts || [],
        gold_titles: params.goldTitles || params.gold_titles || [],
        perturbation_type: params.perturbationType || 'spelling',
        perturbation_types: params.perturbationTypes,
        mode: params.mode || 'combined',
        settings: params.settings || {},
        seed: params.seed ?? null,
        supporting_titles: params.supportingTitles ?? null,
        repair_methods: params.repairMethods || null,
        typo_rate: params.typoRate ?? 0.25,
        embedding_provider: params.embeddingProvider || 'sentence-transformers/all-mpnet-base-v2',
        search_type: params.searchType || 'similarity',
        top_k: parseInt(params.topK, 10) || 2,
        llm_model: params.llmModel || 'ollama/qwen3.5:9b',
        gemini_api_key: params.geminiApiKey || window.GEMINI_API_KEY || null
    };

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        if (!params.onEvent) return await response.json();
        return await readEvaluationStream(response, params.onEvent);
    } catch (err) {
        console.error("Error executing backend evaluation:", err);
        throw err;
    }
}

export async function readEvaluationStream(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result;
    const consume = line => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.event === 'error') throw new Error(event.message);
        if (event.event === 'complete') result = event.result;
        onEvent(event);
    };
    try {
        while (true) {
            const {value, done} = await reader.read();
            buffer += decoder.decode(value, {stream: !done});
            const lines = buffer.split('\n');
            buffer = lines.pop();
            lines.forEach(consume);
            if (done) break;
        }
        consume(buffer);
        if (!result) throw new Error('Evaluation ended before completion. Check the backend terminal.');
        return result;
    } finally {
        await reader.cancel();
        reader.releaseLock();
    }
}
