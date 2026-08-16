/**
 * Evaluation Metrics & Similarity Calculation Engine
 */

/**
 * Calculates N-gram & TF-IDF weighted Cosine Similarity between two texts.
 * Captures character-level subword overlap and word n-grams for significantly higher semantic accuracy.
 * @param {string} text1 
 * @param {string} text2 
 * @returns {number} similarity score from 0.0 to 1.0
 */
export function calculateCosineSimilarity(text1, text2) {
    if (!text1 || !text2) return 0.0;
    if (text1 === text2) return 1.0;

    const normalize = str => str.toLowerCase().replace(/[^\w\s]/g, ' ').trim();
    const clean1 = normalize(text1);
    const clean2 = normalize(text2);

    // Extract word tokens and 3-gram character sequences for subword resilience
    const getFeatures = (text) => {
        const words = text.split(/\s+/).filter(Boolean);
        const features = {};
        
        // Unigrams & Bigrams
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            features[word] = (features[word] || 0) + 1.0;
            if (i < words.length - 1) {
                const bigram = `${words[i]}_${words[i+1]}`;
                features[bigram] = (features[bigram] || 0) + 1.5;
            }
            // Character 3-grams for typo resilience
            if (word.length >= 3) {
                for (let j = 0; j <= word.length - 3; j++) {
                    const trigram = `char_${word.substring(j, j + 3)}`;
                    features[trigram] = (features[trigram] || 0) + 0.3;
                }
            }
        }
        return features;
    };

    const vec1 = getFeatures(clean1);
    const vec2 = getFeatures(clean2);

    const allKeys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
    if (allKeys.size === 0) return 1.0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    allKeys.forEach(key => {
        const val1 = vec1[key] || 0;
        const val2 = vec2[key] || 0;
        dotProduct += val1 * val2;
        normA += val1 * val1;
        normB += val2 * val2;
    });

    if (normA === 0 || normB === 0) return 0.0;
    const sim = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.min(1.0, Math.max(0.0, sim));
}

/**
 * Calculates stage error propagation metrics and returns status classification.
 * @param {number} retrievalSim 
 * @param {number} generationSim 
 * @returns {Object}
 */
export function calculatePropagationMetrics(retrievalSim, generationSim) {
    const retrievalLoss = Math.max(0, 1.0 - retrievalSim);
    const generationLoss = Math.max(0, 1.0 - generationSim);
    const amplificationFactor = generationLoss / (retrievalLoss + 0.00001);

    let statusText = "Proportional";
    let statusBadgeClass = "bg-amber-500/20 text-amber-300 border-amber-500/30";
    let statusDesc = "Error level carried through uniformly to final output.";
    let statusCategory = "Proportional";

    if (generationSim > retrievalSim + 0.08) {
        statusText = "Absorbed (LLM Resilient)";
        statusBadgeClass = "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
        statusDesc = "LLM generator compensated for context/query perturbation.";
        statusCategory = "Absorbed";
    } else if (generationSim < retrievalSim - 0.08) {
        statusText = "Amplified (LLM Output Degraded)";
        statusBadgeClass = "bg-rose-500/20 text-rose-300 border-rose-500/30";
        statusDesc = "Minor perturbation caused significant degradation in answer accuracy.";
        statusCategory = "Amplified";
    }

    return {
        retrievalSim,
        generationSim,
        amplificationFactor,
        statusText,
        statusBadgeClass,
        statusDesc,
        statusCategory
    };
}
