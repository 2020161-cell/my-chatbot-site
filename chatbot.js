// ------------------------------
// 1. Load PDF from GitHub Pages
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
// 2. Chunk text for retrieval
// ------------------------------
function chunkText(text, chunkSize = 800) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

// ------------------------------
// 3. Embedding using Transformers.js
// ------------------------------
let embedder;

async function embed(text) {
    if (!embedder) {
        embedder = await window.transformers.pipeline(
            "feature-extraction",
            "Xenova/distilbert-base-uncased"
        );
    }

    const output = await embedder(text, { pooling: "mean", normalize: true });
    return output.data;
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ------------------------------
// 4. Build knowledge base
// ------------------------------
let knowledgeChunks = [];
let knowledgeEmbeddings = [];

async function initKnowledgeBase() {
    const pdfFile = "e_budget_speech-2026-27.pdf";
    const pdfText = await loadPDFText(pdfFile);
    knowledgeChunks = chunkText(pdfText);

    for (const chunk of knowledgeChunks) {
        knowledgeEmbeddings.push(await embed(chunk));
    }

    console.log("Knowledge base loaded:", knowledgeChunks.length, "chunks");
}

initKnowledgeBase();

// ------------------------------
// 5. Retrieve relevant chunks
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
// 6. LLM using Transformers.js
// ------------------------------
let generator;

async function initLLM() {
    generator = await window.transformers.pipeline(
        "text-generation",
        "Xenova/gpt2"
    );
}

initLLM();

async function runLocalLLM(prompt) {
    if (!generator) return "Model loading… please wait.";

    const output = await generator(prompt, {
        max_new_tokens: 150,
        temperature: 0.7,
        top_p: 0.9
    });

    return output[0].generated_text;
}

// ------------------------------
// 7. Chat interaction
// ------------------------------
async function answerUser(query) {
    const context = await retrieveRelevantChunks(query);

    const prompt = `
Use the following knowledge to answer the user's question.
If the answer is not clearly supported by the knowledge, say you are not sure.

Knowledge:
${context.join("\n\n")}

User question: ${query}

Answer:
`;

    return runLocalLLM(prompt);
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
