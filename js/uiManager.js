/**
 * UI Manager & Chart.js Lifecycle Controller
 */

let stageChart = null;

/**
 * Renders context cards into a target container.
 * @param {string} containerId 
 * @param {Array} contexts 
 */
export function renderContexts(containerId, contexts) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    
    contexts.forEach((c, i) => {
        const card = document.createElement('div');
        const isConflictDoc = c.title && c.title.includes("Synthetic");
        const hasScore = c.score !== undefined && c.score !== null;
        const scoreVal = hasScore ? Number(c.score).toFixed(4) : null;
        
        card.className = `p-2.5 rounded-lg border text-xs transition fade-in ${
            isConflictDoc 
                ? 'bg-rose-950/40 border-rose-500/50 text-rose-200' 
                : 'bg-slate-950 border-slate-800 text-slate-300'
        }`;

        card.innerHTML = `
            <div class="flex items-center justify-between font-bold mb-1.5 text-[11px]">
                <span class="${isConflictDoc ? 'text-rose-400' : 'text-indigo-400'} truncate mr-2">[Doc ${i+1}] ${c.title}</span>
                <div class="flex items-center gap-1.5 shrink-0">
                    ${hasScore ? `<span class="text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30" title="ChromaDB Relevance Score"><i class="fa-solid fa-bullseye text-[8px] mr-1"></i>Score: ${scoreVal}</span>` : ''}
                    ${isConflictDoc ? '<span class="text-[9px] bg-rose-500/20 px-1.5 py-0.5 rounded text-rose-300">Conflicting Injected</span>' : ''}
                </div>
            </div>
            <p class="leading-relaxed font-sans text-[11px]">${c.text}</p>
        `;
        container.appendChild(card);
    });
}

/**
 * Updates top metric cards display and granular telemetry.
 * @param {Object} metrics 
 */
export function updateMetricCards(metrics) {
    const { 
        retrievalSim, 
        generationSim, 
        amplificationFactor, 
        statusText, 
        statusBadgeClass, 
        statusDesc,
        query_embedding_shift,
        retrieval_rank_shift_jaccard,
        context_construction_loss,
        generation_vs_gt_baseline,
        generation_vs_gt_perturbed
    } = metrics;
    
    document.getElementById('metric-retrieval-sim').textContent = (retrievalSim !== undefined ? retrievalSim : 1.0).toFixed(3);
    document.getElementById('metric-generation-sim').textContent = (generationSim !== undefined ? generationSim : 1.0).toFixed(3);
    document.getElementById('metric-amplification').textContent = (amplificationFactor !== undefined ? amplificationFactor : 0.0).toFixed(2) + 'x';

    const statusBadgeEl = document.getElementById('metric-status-badge');
    if (statusBadgeEl) {
        statusBadgeEl.textContent = statusText || "Absorbed";
        statusBadgeEl.className = `inline-block text-sm font-bold px-3 py-1 rounded-lg border ${statusBadgeClass || 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'}`;
    }
    document.getElementById('metric-status-desc').textContent = statusDesc || "";

    // Granular telemetry fields
    if (document.getElementById('telemetry-query-shift')) {
        document.getElementById('telemetry-query-shift').textContent = (query_embedding_shift ?? 1.0).toFixed(4);
    }
    if (document.getElementById('telemetry-rank-jaccard')) {
        document.getElementById('telemetry-rank-jaccard').textContent = (retrieval_rank_shift_jaccard ?? 1.0).toFixed(4);
    }
    if (document.getElementById('telemetry-context-loss')) {
        document.getElementById('telemetry-context-loss').textContent = (context_construction_loss ?? 1.0).toFixed(4);
    }
    if (document.getElementById('telemetry-gen-gt-base')) {
        document.getElementById('telemetry-gen-gt-base').textContent = `B: ${(generation_vs_gt_baseline ?? 1.0).toFixed(2)}`;
    }
    if (document.getElementById('telemetry-gen-gt-pert')) {
        document.getElementById('telemetry-gen-gt-pert').textContent = `P: ${(generation_vs_gt_perturbed ?? 1.0).toFixed(2)}`;
    }
}

/**
 * Updates History Log list UI and resilience rate badge.
 * @param {Array} history 
 */
export function updateHistoryUI(history) {
    const list = document.getElementById('history-list');
    if (!list) return;
    list.innerHTML = '';
    
    history.slice().reverse().forEach(run => {
        const item = document.createElement('div');
        let badgeClass = "bg-slate-800 text-slate-300";
        if (run.statusCategory === "Absorbed" || run.status === "Absorbed") badgeClass = "bg-emerald-500/20 text-emerald-300";
        if (run.statusCategory === "Amplified" || run.status === "Amplified") badgeClass = "bg-rose-500/20 text-rose-300";
        if (run.statusCategory === "Proportional" || run.status === "Proportional") badgeClass = "bg-amber-500/20 text-amber-300";

        item.className = "flex items-center justify-between p-2 rounded-lg bg-slate-950 border border-slate-800 text-xs fade-in";
        item.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="font-mono text-[10px] text-slate-500">#${run.id}</span>
                <span class="font-semibold text-slate-200 capitalize">${run.type}</span>
            </div>
            <div class="flex items-center space-x-2">
                <span class="text-[10px] text-slate-400">R: ${run.retrievalSim} | G: ${run.generationSim}</span>
                <span class="text-[10px] px-2 py-0.5 rounded ${badgeClass}">${run.statusCategory || run.status}</span>
            </div>
        `;
        list.appendChild(item);
    });

    document.getElementById('total-runs-count').textContent = history.length;
    const absorbedCount = history.filter(r => (r.statusCategory === "Absorbed" || r.status === "Absorbed")).length;
    const rate = history.length > 0 ? Math.round((absorbedCount / history.length) * 100) : 100;
    document.getElementById('resilience-rate').textContent = `Resilience Rate: ${rate}%`;
}

/**
 * Initializes Chart.js bar chart for stage degradation tracking.
 */
export function initChart() {
    const canvas = document.getElementById('stageComparisonChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    stageChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Initial Baseline'],
            datasets: [
                {
                    label: 'Retrieval Cosine Sim',
                    data: [1.0],
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: 'Generation Cosine Sim',
                    data: [1.0],
                    backgroundColor: 'rgba(167, 139, 250, 0.7)',
                    borderColor: 'rgba(167, 139, 250, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1', font: { size: 11 } }
                }
            }
        }
    });
}

/**
 * Updates Chart.js dataset with history metrics.
 * FIXED: Corrected genData.data > 0 bug to genData.length > 0
 * @param {Array} history 
 */
export function updateChart(history = []) {
    if (!stageChart) return;
    
    const labels = history.map(h => `Run #${h.id} (${h.type})`);
    const retData = history.map(h => h.retrievalSim);
    const genData = history.map(h => h.generationSim);

    stageChart.data.labels = labels.length > 0 ? labels : ['Baseline'];
    stageChart.data.datasets[0].data = retData.length > 0 ? retData : [1.0];
    stageChart.data.datasets[1].data = genData.length > 0 ? genData : [1.0]; // FIXED
    stageChart.update();
}
