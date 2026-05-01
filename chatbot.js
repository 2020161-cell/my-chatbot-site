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
// 3. Simple embedding (browser-only)
// ------------------------------
async function embed(text) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    return data.slice(0, 256); // simple local embedding
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
// 6. WebLLM integration
// ------------------------------
let webllmEngine = null;
let webllmReady = false;
let webllmLoadingError = null;

async function initLLM() {
    try {
        // WebLLM is exposed as global "webllm" when you include its script in index.html
        // Example model: Llama-3.2-1B-Instruct-q4f32_1-MLC
        webllmEngine = await webllm.CreateMLCEngine(
            "Llama-3.2-1B-Instruct-q4f32_1-MLC",
            {
                // optional config
                temperature: 0.2,
                top_p: 0.9
            }
        );
        webllmReady = true;
        console.log("WebLLM model loaded.");
    } catch (err) {
        console.error("Error loading WebLLM:", err);
        webllmLoadingError = err;
    }
}

// kick off model loading
initLLM();

async function runLocalLLM(prompt) {
    if (webllmLoadingError) {
        return "LLM failed to load. Please refresh the page or try again later.";
    }
    if (!webllmReady || !webllmEngine) {
        return "Model is still loading… please wait a few seconds and ask again.";
    }

    const result = await webllmEngine.chat.completions.create({
        messages: [
            {
                role: "system",
                content: "You are a helpful assistant answering questions about the Hong Kong Budget 2026–27. Use only the provided context; if the answer is not in the context, say you are not sure."
            },
            {
                role: "user",
                content: prompt
            }
        ],
        max_tokens: 256
    });

    const choice = result.choices && result.choices[0];
    if (!choice || !choice.message || !choice.message.content) {
        return "I could not generate a response. Please try again.";
    }

    return choice.message.content;
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
