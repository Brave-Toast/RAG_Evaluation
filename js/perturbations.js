/**
 * Perturbation Injection Module
 */

// Display order matches the backend's fixed composition order.
export const PERTURBATION_LABELS = {
    rephrase: 'Query rephrasing',
    spelling: 'Spelling errors',
    emoji: 'Emoji perturbation',
    conflict: 'Conflicting context',
    remove_evidence: 'Remove supporting evidence',
    alter_facts: 'Alter facts',
    truncate: 'Truncate passages',
    irrelevant: 'Add irrelevant passages',
    shuffle: 'Shuffle passage order'
};

export function methodLabel(methods) {
    return methods.map(method => PERTURBATION_LABELS[method] || method).join(' + ');
}

/**
 * Injects synthetic spelling errors (swaps, drops, character substitutions) into text.
 * @param {string} text 
 * @param {number} errorRate (0.0 to 1.0)
 * @returns {string}
 */
export function injectSpellingErrors(text, errorRate = 0.25) {
    if (!text) return "";
    const words = text.split(' ');
    const perturbed = words.map(word => {
        if (word.length > 3 && Math.random() < errorRate) {
            const ops = ['swap', 'drop', 'sub'];
            const op = ops[Math.floor(Math.random() * ops.length)];
            const idx = Math.floor(Math.random() * (word.length - 2)) + 1;
            
            if (op === 'swap') {
                const arr = word.split('');
                const tmp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = tmp;
                return arr.join('');
            } else if (op === 'drop') {
                return word.slice(0, idx) + word.slice(idx + 1);
            } else { // sub
                const chars = 'abcdefghijklmnopqrstuvwxyz';
                const randomChar = chars[Math.floor(Math.random() * chars.length)];
                return word.slice(0, idx) + randomChar + word.slice(idx + 1);
            }
        }
        return word;
    });
    return perturbed.join(' ');
}

/**
 * Injects a synthetic counter-factual document into retrieved contexts.
 * @param {Array} originalContexts 
 * @param {string} query 
 * @param {string} groundTruth 
 * @returns {Array}
 */
export function injectConflictingContext(originalContexts, query, groundTruth) {
    const conflictDoc = {
        title: "Synthetic_Conflicting_Record",
        text: `NOTICE: Regarding the query '${query}', conflicting modern sources refute that '${groundTruth}' is accurate. Historical logs indicate the complete opposite is true.`
    };
    return [conflictDoc, ...originalContexts];
}

/**
 * Generates an LLM rephrased query.
 * @param {string} activeQueryText 
 * @param {string} llmModelChoice 
 * @param {Function} fetchGeminiAnswerFn 
 * @returns {Promise<string>}
 */
export async function generateRephrase(activeQueryText, llmModelChoice, fetchGeminiAnswerFn) {
    if (llmModelChoice === 'gemini-3-flash-preview' && typeof fetchGeminiAnswerFn === 'function') {
        const apiRephrase = await fetchGeminiAnswerFn(
            `Rephrase this question concisely while maintaining exact meaning: "${activeQueryText}"`, 
            "Return ONLY the rephrased query text."
        );
        if (apiRephrase) return apiRephrase;
    }
    return `Could you inform me regarding: ${activeQueryText.toLowerCase()}?`;
}
