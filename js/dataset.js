/**
 * Dynamic HotpotQA Evaluation Dataset Manager
 * Loads 50+ benchmark questions from FastAPI backend (/api/dataset) with local fallback.
 */

// Fallback dataset in case backend is offline on page load
export let HOTPOTQA_DATASET = [
    {
        id: "1",
        question: "Which magazine was started first, Arthur's Magazine or First for Women?",
        ground_truth: "Arthur's Magazine",
        contexts: [
            { title: "Arthur's Magazine", text: "Arthur's Magazine (1844–1846) was an American literary magazine published in Philadelphia in the 19th century." },
            { title: "First for Women", text: "First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989." },
            { title: "Publishing History Quarterly", text: "19th century literary magazines laid the groundwork for specialized modern fashion and lifestyle publications." }
        ]
    },
    {
        id: "2",
        question: "Were Scott Derrickson and Rupert Wyatt both born in the same decade?",
        ground_truth: "No, Scott Derrickson was born in 1966 and Rupert Wyatt in 1972.",
        contexts: [
            { title: "Scott Derrickson", text: "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He directed Doctor Strange." },
            { title: "Rupert Wyatt", text: "Rupert Wyatt (born 26 October 1972) is an English film director, screenwriter, and producer who directed Rise of the Planet of the Apes." },
            { title: "British & American Directors Registry", text: "Modern cinema directors emerging in the late 20th century frequently crossed over between indie projects and blockbuster franchises." }
        ]
    },
    {
        id: "3",
        question: "What clean energy source produces power at the Hoover Dam?",
        ground_truth: "Hydroelectric power",
        contexts: [
            { title: "Hoover Dam", text: "Hoover Dam is a concrete black-canyon arch-gravity dam on the Colorado River, constructing Lake Mead. Its plant generates massive hydroelectric power." },
            { title: "Clean Energy Overview", text: "Hydroelectric energy utilizes flowing water to spin turbines, producing renewable zero-emission electricity." },
            { title: "Renewable Power Infrastructure", text: "Large scale dam reservoirs store kinetic energy in liquid mass to provide grid stabilization during peak energy consumption." }
        ]
    }
];

export async function loadDatasetFromBackend(backendUrl = 'http://localhost:8000') {
    try {
        const response = await fetch(`${backendUrl}/api/dataset`);
        if (response.ok) {
            const data = await response.json();
            if (data && Array.isArray(data.dataset) && data.dataset.length > 0) {
                HOTPOTQA_DATASET = data.dataset;
                console.log(`Loaded ${HOTPOTQA_DATASET.length} HotpotQA benchmark questions from backend.`);
                return HOTPOTQA_DATASET;
            }
        }
    } catch (err) {
        console.warn("Could not fetch dataset from FastAPI backend, using local fallback dataset:", err);
    }
    return HOTPOTQA_DATASET;
}

export function getQueryByIndex(index) {
    const idx = parseInt(index, 10);
    return HOTPOTQA_DATASET[idx] || HOTPOTQA_DATASET[0];
}
