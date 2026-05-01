// ------------------------------
// 1. Load PDF
// ------------------------------
async function loadPDFText(pdfUrl) {
    const pdf = await pdfjsLib.getDocument(pdfUrl).promise;
    let fullText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const strings = content.items.map(item => item.str).join(" ");
        fullText += strings + "\n";
    }

    return fullText;
}

// ------------------------------
// 2. Chunk text
// ------------------------------
function chunkText(text, chunkSize = 800) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

// ------------------------------
// 3. Phi‑1.5 unified model
// ------------------------------
let phiModel;

async function initModel() {
    phiModel = await window.transformers.pipeline(
        "text-generation",
        "Xenova/phi-1_5"
    );
    console.log("Phi‑1.5 loaded");
}

initModel();

// ------------------------------
// 4. Embedding using Phi‑1.5
// ------------------------------
async function embed(text) {
    const output = await phiModel(text, {
        max_new_tokens: 1,
        return_full_text: false
    });

    // Convert text to simple numeric embedding
    const encoder = new TextEncoder();
    const data = encoder.encode(output[0].generated_text);
    return Array.from(data).slice(0, 256);
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ------------------------------
// 5. Build knowledge base
// ------------------------------
let knowledgeChunks = [];
let knowledgeEmbeddings = [];

async function initKnowledgeBase() {
    const pdfText = await loadPDFText("e_budget_speech-2026-27.pdf");
    knowledgeChunks = chunkText(pdfText);

    for (const chunk of knowledgeChunks) {
        knowledgeEmbeddings.push(await embed(chunk));
    }

    console.log("Knowledge base ready:", knowledgeChunks.length, "chunks");
}

initKnowledgeBase();

// ------------------------------
// 6. Retrieve relevant chunks
// ------------------------------
async function retrieveRelevantChunks(query) {
    const queryEmbedding = await embed(query);

    const scored = knowledgeEmbeddings.map((emb, i) => ({
        chunk: knowledgeChunks[i],
        score: cosineSimilarity(queryEmbedding, emb)
    }));

    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, 3).map(s => s.chunk);
}

// ------------------------------
// 7. Generate answer using Phi‑1.5
// ------------------------------
async function runLLM(prompt) {
    if (!phiModel) return "Model loading… please wait.";

    const output = await phiModel(prompt, {
        max_new_tokens: 180,
        temperature: 0.7,
        top_p: 0.9
    });

    return output[0].generated_text;
}

// ------------------------------
// 8. Chat interaction
// ------------------------------
async function answerUser(query) {
    const context = await retrieveRelevantChunks(query);

    const prompt = `
Use the following knowledge to answer the user's question.
If the answer is not in the knowledge, say you are not sure.

Knowledge:
${context.join("\n\n")}

User question: ${query}

Answer:
`;

    return runLLM(prompt);
}

async function sendMessage() {
    const input = document.getElementById("userInput");
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";

    const reply = await answerUser(text);
    addMessage(reply, "bot");
}

function addMessage(msg, sender) {
    const box = document.getElementById("chatbox");
    const div = document.createElement("div");
    div.className = sender;
    div.textContent = msg;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}
