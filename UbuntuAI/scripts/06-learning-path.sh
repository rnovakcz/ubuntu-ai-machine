#!/bin/bash
#===============================================================================
# 06-learning-path.sh - AI Developer Learning Path
# Vše lokálně spustitelné bez placených účtů
#===============================================================================

set -e

BLUE='\033[0;34m'; GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${BLUE}[INFO]${NC} $1"; }
ok() { echo -e "${GREEN}[OK]${NC} $1"; }

[[ $EUID -ne 0 ]] && { echo "Spusťte jako root (sudo)"; exit 1; }
USER_REAL=${SUDO_USER:-$USER}
HOME_REAL=$(getent passwd "$USER_REAL" | cut -d: -f6)
LEARN="$HOME_REAL/AI-Learning"

log "========== AI DEVELOPER LEARNING PATH =========="

sudo -u "$USER_REAL" mkdir -p "$LEARN"

#===============================================================================
# 01 - PYTHON PRO AI
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/01-python-ai"
cat > "$LEARN/01-python-ai/README.md" << 'EOF'
# 01 - Python pro AI

## Co se naučíš
- NumPy pro numerické výpočty
- Pandas pro práci s daty
- Matplotlib/Seaborn pro vizualizace

## Soubory
- `numpy_basics.py` - Základy NumPy
- `pandas_basics.py` - Základy Pandas
- `visualization.py` - Vizualizace dat
EOF

cat > "$LEARN/01-python-ai/numpy_basics.py" << 'EOF'
#!/usr/bin/env python3
"""NumPy základy pro AI."""
import numpy as np

print("=== NumPy Základy ===\n")

# Vytvoření arrays
arr = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {arr}")

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D Matrix:\n{matrix}")

# Speciální arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
random = np.random.randn(3, 3)  # Normální distribuce
print(f"\nRandom matrix:\n{random}")

# Operace
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\nSčítání: {a + b}")
print(f"Násobení: {a * b}")
print(f"Dot product: {np.dot(a, b)}")

# Reshaping - důležité pro neuronové sítě
data = np.arange(12)
reshaped = data.reshape(3, 4)
print(f"\nReshaped (3x4):\n{reshaped}")

# Indexing a slicing
print(f"\nPrvní řádek: {reshaped[0]}")
print(f"Sloupec 1: {reshaped[:, 1]}")

# Broadcasting - automatické rozšíření dimenzí
matrix = np.ones((3, 3))
vector = np.array([1, 2, 3])
result = matrix + vector
print(f"\nBroadcasting:\n{result}")

# Statistika
data = np.random.randn(1000)
print(f"\nStatistika:")
print(f"Mean: {np.mean(data):.4f}")
print(f"Std: {np.std(data):.4f}")
print(f"Min: {np.min(data):.4f}")
print(f"Max: {np.max(data):.4f}")
EOF

cat > "$LEARN/01-python-ai/pandas_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Pandas základy pro AI."""
import pandas as pd
import numpy as np

print("=== Pandas Základy ===\n")

# DataFrame vytvoření
df = pd.DataFrame({
    'jmeno': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'vek': [25, 30, 35, 28],
    'plat': [50000, 60000, 75000, 55000],
    'oddeleni': ['IT', 'HR', 'IT', 'Marketing']
})
print("DataFrame:")
print(df)
print()

# Základní info
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nInfo:")
print(df.info())
print(f"\nStatistika:")
print(df.describe())

# Selekce
print(f"\nSloupec 'jmeno':\n{df['jmeno']}")
print(f"\nŘádky kde vek > 28:\n{df[df['vek'] > 28]}")

# Groupby - velmi užitečné pro analýzu
print(f"\nPrůměrný plat podle oddělení:")
print(df.groupby('oddeleni')['plat'].mean())

# Práce s chybějícími hodnotami
df_missing = df.copy()
df_missing.loc[1, 'plat'] = np.nan
print(f"\nChybějící hodnoty:\n{df_missing.isnull().sum()}")
df_filled = df_missing.fillna(df_missing['plat'].mean())
print(f"Po vyplnění:\n{df_filled}")

# Ukládání a načítání
df.to_csv('/tmp/test_data.csv', index=False)
loaded = pd.read_csv('/tmp/test_data.csv')
print(f"\nNačteno z CSV:\n{loaded}")
EOF

cat > "$LEARN/01-python-ai/visualization.py" << 'EOF'
#!/usr/bin/env python3
"""Vizualizace dat pro AI."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== Vizualizace ===")
print("Grafy se uloží do /tmp/\n")

# Data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue')
plt.plot(x, y2, label='cos(x)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sinusovky')
plt.legend()
plt.grid(True)
plt.savefig('/tmp/line_plot.png', dpi=150)
plt.close()
print("Uloženo: /tmp/line_plot.png")

# Histogram
data = np.random.randn(1000)
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Hodnota')
plt.ylabel('Četnost')
plt.title('Histogram normálního rozdělení')
plt.savefig('/tmp/histogram.png', dpi=150)
plt.close()
print("Uloženo: /tmp/histogram.png")

# Seaborn heatmap (confusion matrix style)
matrix = np.random.rand(5, 5)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues')
plt.title('Heatmap')
plt.savefig('/tmp/heatmap.png', dpi=150)
plt.close()
print("Uloženo: /tmp/heatmap.png")

# Scatter plot
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.savefig('/tmp/scatter.png', dpi=150)
plt.close()
print("Uloženo: /tmp/scatter.png")

print("\nHotovo! Otevři obrázky v /tmp/")
EOF

#===============================================================================
# 02 - PYTORCH ZÁKLADY
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/02-pytorch-basics"
cat > "$LEARN/02-pytorch-basics/README.md" << 'EOF'
# 02 - PyTorch Základy

## Co se naučíš
- Tensory a GPU akcelerace
- Autograd (automatická derivace)
- Základní neuronová síť

## Soubory
- `tensors.py` - Práce s tensory
- `autograd.py` - Automatická derivace
- `simple_nn.py` - První neuronová síť
EOF

cat > "$LEARN/02-pytorch-basics/tensors.py" << 'EOF'
#!/usr/bin/env python3
"""PyTorch tensory."""
import torch

print("=== PyTorch Tensory ===\n")
print(f"PyTorch verze: {torch.__version__}")
print(f"CUDA dostupná: {torch.cuda.is_available()}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Používám: {device}\n")

# Vytvoření tensorů
t1 = torch.tensor([1, 2, 3, 4])
print(f"Tensor: {t1}")

t2 = torch.zeros(3, 3)
print(f"Zeros:\n{t2}")

t3 = torch.randn(3, 3)  # Normální rozdělení
print(f"Random:\n{t3}")

# GPU tensor
if torch.cuda.is_available():
    gpu_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"\nGPU tensor shape: {gpu_tensor.shape}")
    print(f"GPU tensor device: {gpu_tensor.device}")

# Operace
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"\nSčítání: {a + b}")
print(f"Násobení: {a * b}")
print(f"Dot product: {torch.dot(a, b)}")

# Matrix multiplication
m1 = torch.randn(3, 4)
m2 = torch.randn(4, 5)
result = torch.mm(m1, m2)
print(f"\nMatrix mul (3x4) @ (4x5) = {result.shape}")

# Reshaping
t = torch.arange(12)
print(f"\nOriginal: {t}")
print(f"Reshaped (3x4):\n{t.reshape(3, 4)}")
print(f"View (2x6):\n{t.view(2, 6)}")

# Důležité pro batch processing
batch = torch.randn(32, 3, 224, 224)  # batch, channels, height, width
print(f"\nImage batch shape: {batch.shape}")
EOF

cat > "$LEARN/02-pytorch-basics/autograd.py" << 'EOF'
#!/usr/bin/env python3
"""PyTorch Autograd - automatická derivace."""
import torch

print("=== Autograd ===\n")

# Tensor s requires_grad=True sleduje operace
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}")

# Výpočet
y = x ** 2 + 3 * x + 1
print(f"y = x² + 3x + 1 = {y}")

# Zpětná propagace
z = y.sum()  # Skalár pro backward
z.backward()

# Gradienty: dy/dx = 2x + 3
print(f"Gradienty (dy/dx = 2x + 3): {x.grad}")
# Pro x=2: 2*2+3=7, pro x=3: 2*3+3=9 ✓

# Praktický příklad: Lineární regrese
print("\n=== Mini Lineární Regrese ===")
torch.manual_seed(42)

# Data
X = torch.randn(100, 1)
y_true = 3 * X + 2 + torch.randn(100, 1) * 0.1

# Parametry
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.1

for epoch in range(100):
    # Forward
    y_pred = X * w + b
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward
    loss.backward()
    
    # Update (bez gradientů)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Reset gradientů
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\nNaučené: w={w.item():.2f} (skutečné 3), b={b.item():.2f} (skutečné 2)")
EOF

cat > "$LEARN/02-pytorch-basics/simple_nn.py" << 'EOF'
#!/usr/bin/env python3
"""První neuronová síť v PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=== Neuronová Síť ===\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Generování dat (XOR problém - nelze řešit lineárně)
torch.manual_seed(42)
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Definice sítě
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Model, loss, optimizer
model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

print(f"Model:\n{model}\n")

# Training
X, y = X.to(device), y.to(device)

for epoch in range(1000):
    # Forward
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Testování
print("\n=== Výsledky ===")
model.eval()
with torch.no_grad():
    predictions = model(X)
    for i in range(len(X)):
        pred = predictions[i].item()
        actual = y[i].item()
        print(f"Input: {X[i].tolist()} -> Pred: {pred:.3f}, Actual: {actual}")
EOF

#===============================================================================
# 03 - TRANSFORMERS & LLM
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/03-transformers-llm"
cat > "$LEARN/03-transformers-llm/README.md" << 'EOF'
# 03 - Transformers & LLM

## Co se naučíš
- Hugging Face transformers
- Práce s lokálními LLM
- Text generation, embeddings

## Soubory
- `hf_basics.py` - Hugging Face základy
- `local_llm.py` - Lokální LLM s Ollama
- `embeddings.py` - Text embeddings
EOF

cat > "$LEARN/03-transformers-llm/hf_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Hugging Face Transformers základy."""
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

print("=== Hugging Face Transformers ===\n")

# 1. Sentiment Analysis (stáhne malý model)
print("1. Sentiment Analysis")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
])
for r in results:
    print(f"  {r['label']}: {r['score']:.3f}")

# 2. Text Generation (malý GPT-2)
print("\n2. Text Generation")
generator = pipeline("text-generation", model="distilgpt2")
text = generator("Artificial intelligence is", max_length=30, num_return_sequences=1)
print(f"  {text[0]['generated_text']}")

# 3. Question Answering
print("\n3. Question Answering")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "Python is a programming language created by Guido van Rossum in 1991."
question = "Who created Python?"
answer = qa(question=question, context=context)
print(f"  Q: {question}")
print(f"  A: {answer['answer']} (score: {answer['score']:.3f})")

# 4. Fill Mask
print("\n4. Fill Mask")
fill = pipeline("fill-mask", model="distilbert-base-uncased")
result = fill("Machine learning is a type of [MASK] intelligence.")
print(f"  Top prediction: {result[0]['token_str']} ({result[0]['score']:.3f})")

# 5. Named Entity Recognition
print("\n5. Named Entity Recognition")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
entities = ner("Bill Gates founded Microsoft in Seattle.")
for e in entities:
    print(f"  {e['word']}: {e['entity_group']}")

print("\n✓ Všechny modely běží lokálně!")
EOF

cat > "$LEARN/03-transformers-llm/local_llm.py" << 'EOF'
#!/usr/bin/env python3
"""Lokální LLM s Ollama (bez API klíčů)."""
import requests
import json

OLLAMA_URL = "http://localhost:11434"

def check_ollama():
    """Zkontroluj jestli Ollama běží."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def list_models():
    """Seznam lokálních modelů."""
    r = requests.get(f"{OLLAMA_URL}/api/tags")
    models = r.json().get("models", [])
    return [m["name"] for m in models]

def generate(model: str, prompt: str) -> str:
    """Generování textu."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return r.json()["response"]

def chat(model: str, messages: list) -> str:
    """Chat s modelem."""
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False}
    )
    return r.json()["message"]["content"]

if __name__ == "__main__":
    print("=== Lokální LLM s Ollama ===\n")
    
    if not check_ollama():
        print("❌ Ollama neběží!")
        print("Spusťte: ollama serve")
        print("Stáhněte model: ollama pull llama3.2")
        exit(1)
    
    models = list_models()
    print(f"Dostupné modely: {models}")
    
    if not models:
        print("\n❌ Žádné modely! Stáhněte: ollama pull llama3.2")
        exit(1)
    
    model = models[0]
    print(f"\nPoužívám model: {model}")
    
    # Jednoduchá generace
    print("\n--- Generování ---")
    prompt = "Vysvětli co je neuronová síť ve 2 větách."
    response = generate(model, prompt)
    print(f"Q: {prompt}")
    print(f"A: {response}")
    
    # Chat
    print("\n--- Chat ---")
    messages = [
        {"role": "user", "content": "Ahoj! Jak se máš?"},
    ]
    response = chat(model, messages)
    print(f"User: {messages[0]['content']}")
    print(f"AI: {response}")
EOF

cat > "$LEARN/03-transformers-llm/embeddings.py" << 'EOF'
#!/usr/bin/env python3
"""Text Embeddings - základ pro RAG."""
from sentence_transformers import SentenceTransformer
import numpy as np

print("=== Text Embeddings ===\n")

# Malý ale kvalitní model
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Model: all-MiniLM-L6-v2")
print(f"Embedding size: 384\n")

# Dokumenty
documents = [
    "Python is a programming language used for AI.",
    "Machine learning uses data to train models.",
    "Deep learning is based on neural networks.",
    "Cats are popular pets.",
    "Dogs are loyal companions."
]

# Vytvoření embeddingů
embeddings = model.encode(documents)
print(f"Embeddings shape: {embeddings.shape}")

# Query
query = "What is artificial intelligence?"
query_embedding = model.encode(query)

# Výpočet podobnosti (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nQuery: '{query}'")
print("\nPodobnosti:")

similarities = []
for i, doc in enumerate(documents):
    sim = cosine_similarity(query_embedding, embeddings[i])
    similarities.append((sim, doc))

# Seřazení podle podobnosti
similarities.sort(reverse=True)
for sim, doc in similarities:
    print(f"  {sim:.3f}: {doc}")

print("\n✓ Nejpodobnější dokumenty jsou o AI/ML!")
EOF

#===============================================================================
# 04 - RAG SYSTÉM
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/04-rag-system"
cat > "$LEARN/04-rag-system/README.md" << 'EOF'
# 04 - RAG Systém

## Co se naučíš
- Retrieval Augmented Generation
- Vector store (ChromaDB)
- LangChain pro RAG

## Soubory
- `simple_rag.py` - Jednoduchý RAG systém
- `chroma_basics.py` - ChromaDB vector store
EOF

cat > "$LEARN/04-rag-system/simple_rag.py" << 'EOF'
#!/usr/bin/env python3
"""Jednoduchý RAG systém - vše lokálně."""
from sentence_transformers import SentenceTransformer
import chromadb
import requests

print("=== RAG Systém ===\n")

# 1. Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Vector store
client = chromadb.Client()
collection = client.create_collection("documents")

# 3. Naše znalostní báze
knowledge_base = [
    "Python byl vytvořen Guido van Rossumem v roce 1991.",
    "PyTorch je framework pro deep learning od Facebooku.",
    "TensorFlow vytvořil Google v roce 2015.",
    "Transformers architektura byla představena v paperu 'Attention is All You Need' v 2017.",
    "GPT znamená Generative Pre-trained Transformer.",
    "BERT je model od Google pro porozumění textu.",
    "LLM jsou velké jazykové modely trénované na miliardách tokenů.",
    "RAG kombinuje retrieval s generováním textu.",
    "Vector database ukládají embeddingy pro rychlé vyhledávání.",
    "Fine-tuning adaptuje předtrénovaný model na specifický úkol."
]

# 4. Indexování dokumentů
print("Indexuji dokumenty...")
embeddings = embedder.encode(knowledge_base).tolist()
collection.add(
    documents=knowledge_base,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(knowledge_base))]
)
print(f"Indexováno {len(knowledge_base)} dokumentů\n")

# 5. RAG funkce
def rag_query(question: str, top_k: int = 3) -> str:
    """RAG: najdi relevantní dokumenty a vygeneruj odpověď."""
    # Retrieval
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    context = "\n".join(results['documents'][0])
    
    # Generování (s Ollama)
    try:
        prompt = f"""Kontext:
{context}

Otázka: {question}

Odpověz stručně na základě kontextu:"""
        
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=30
        )
        return r.json()["response"]
    except:
        return f"[LLM nedostupný]\n\nNalezený kontext:\n{context}"

# 6. Test
questions = [
    "Kdo vytvořil Python?",
    "Co je PyTorch?",
    "Co znamená GPT?"
]

for q in questions:
    print(f"Q: {q}")
    answer = rag_query(q)
    print(f"A: {answer}\n")
EOF

cat > "$LEARN/04-rag-system/chroma_basics.py" << 'EOF'
#!/usr/bin/env python3
"""ChromaDB základy."""
import chromadb
from chromadb.utils import embedding_functions

print("=== ChromaDB Základy ===\n")

# Vytvoření klienta (in-memory)
client = chromadb.Client()

# Nebo persistent
# client = chromadb.PersistentClient(path="/tmp/chroma_db")

# Embedding function (Sentence Transformers)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Vytvoření kolekce
collection = client.create_collection(
    name="my_collection",
    embedding_function=ef
)

# Přidání dokumentů
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries.",
    "Python is great for data science.",
    "Neural networks can learn complex patterns.",
]

collection.add(
    documents=documents,
    metadatas=[{"source": f"doc_{i}"} for i in range(len(documents))],
    ids=[f"id_{i}" for i in range(len(documents))]
)

print(f"Dokumentů v kolekci: {collection.count()}")

# Vyhledávání
print("\n--- Vyhledávání ---")
results = collection.query(
    query_texts=["What is AI?"],
    n_results=2
)

print("Query: 'What is AI?'")
print("Nejpodobnější dokumenty:")
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    print(f"  [{dist:.3f}] {doc}")

# Filtrování podle metadat
print("\n--- Filtrování ---")
results = collection.query(
    query_texts=["programming"],
    n_results=2,
    where={"source": "doc_2"}
)
print(f"S filtrem: {results['documents']}")

# Update
collection.update(
    ids=["id_0"],
    documents=["Updated: A quick fox jumped."]
)
print("\nDokument id_0 aktualizován")

# Delete
collection.delete(ids=["id_3"])
print(f"Dokumentů po smazání: {collection.count()}")
EOF

#===============================================================================
# 05 - FINE-TUNING
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/05-fine-tuning"
cat > "$LEARN/05-fine-tuning/README.md" << 'EOF'
# 05 - Fine-tuning

## Co se naučíš
- LoRA a QLoRA techniky
- PEFT knihovna
- Příprava datasetu

## Soubory
- `lora_intro.py` - Úvod do LoRA
- `prepare_dataset.py` - Příprava dat pro fine-tuning
EOF

cat > "$LEARN/05-fine-tuning/lora_intro.py" << 'EOF'
#!/usr/bin/env python3
"""LoRA (Low-Rank Adaptation) úvod."""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("=== LoRA Fine-tuning ===\n")

# Proč LoRA?
print("""
LoRA (Low-Rank Adaptation):
- Místo fine-tuningu všech vah trénujeme pouze malé "adaptéry"
- Drasticky snižuje počet trénovatelných parametrů
- Šetří paměť GPU
- Rychlejší trénink
- Lze kombinovat více LoRA adaptérů
""")

# Příklad s malým modelem
model_name = "distilbert-base-uncased"
print(f"Model: {model_name}")

# Načtení modelu
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Počet parametrů před LoRA
total_params = sum(p.numel() for p in model.parameters())
print(f"Celkem parametrů: {total_params:,}")

# LoRA konfigurace
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,              # Rank - nižší = méně parametrů
    lora_alpha=32,    # Scaling faktor
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # Které vrstvy adaptovat
)

# Aplikace LoRA
peft_model = get_peft_model(model, lora_config)

# Statistiky
print("\n--- LoRA Statistiky ---")
peft_model.print_trainable_parameters()

# Uložení pouze LoRA vah
# peft_model.save_pretrained("./lora_adapter")
# Velikost: ~MB místo GB!

print("""
Další kroky pro plný fine-tuning:
1. Připrav dataset (viz prepare_dataset.py)
2. Použij Trainer z transformers
3. Trénuj s malým learning rate (1e-4 až 5e-5)
4. Ulož pouze LoRA adaptér
""")
EOF

cat > "$LEARN/05-fine-tuning/prepare_dataset.py" << 'EOF'
#!/usr/bin/env python3
"""Příprava datasetu pro fine-tuning."""
from datasets import Dataset, load_dataset
import json

print("=== Příprava Datasetu ===\n")

# 1. Vytvoření vlastního datasetu
print("1. Vlastní dataset")
data = {
    "text": [
        "This product is amazing!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
        "Best purchase ever!",
        "Complete waste of money."
    ],
    "label": [1, 0, 1, 1, 0]  # 1=positive, 0=negative
}

dataset = Dataset.from_dict(data)
print(f"   Vytvořeno: {len(dataset)} příkladů")
print(f"   Sloupce: {dataset.column_names}")

# 2. Formát pro instruktážní fine-tuning (chat)
print("\n2. Instruktážní formát")
instruction_data = [
    {
        "instruction": "Classify the sentiment of this review.",
        "input": "This product is amazing!",
        "output": "Positive"
    },
    {
        "instruction": "Classify the sentiment of this review.",
        "input": "Terrible experience.",
        "output": "Negative"
    }
]

# Uložení jako JSONL
with open("/tmp/train_data.jsonl", "w") as f:
    for item in instruction_data:
        f.write(json.dumps(item) + "\n")
print("   Uloženo: /tmp/train_data.jsonl")

# 3. Načtení veřejného datasetu
print("\n3. Veřejné datasety z Hugging Face")
try:
    # Malý dataset pro ukázku
    imdb = load_dataset("imdb", split="train[:100]")
    print(f"   IMDB: {len(imdb)} příkladů")
    print(f"   Příklad: {imdb[0]['text'][:100]}...")
except:
    print("   (Vyžaduje internet)")

# 4. Formátování pro různé modely
print("\n4. Formáty pro různé modely")

# Alpaca formát
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# ChatML formát
chatml_template = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

print("   Alpaca formát: pro Llama, Mistral")
print("   ChatML formát: pro modely s chat template")

# Příklad konverze
example = instruction_data[0]
print(f"\n--- Alpaca příklad ---")
print(alpaca_template.format(**example))
EOF

#===============================================================================
# 06 - COMPUTER VISION
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/06-computer-vision"
cat > "$LEARN/06-computer-vision/README.md" << 'EOF'
# 06 - Computer Vision

## Co se naučíš
- Zpracování obrázků
- CNN a klasifikace
- Detekce objektů s YOLO

## Soubory
- `image_basics.py` - Základy zpracování obrázků
- `simple_cnn.py` - CNN pro klasifikaci
- `yolo_detection.py` - YOLO detekce
EOF

cat > "$LEARN/06-computer-vision/image_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Základy zpracování obrázků."""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

print("=== Zpracování Obrázků ===\n")

# Vytvoření testovacího obrázku
img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("/tmp/test_image.png")
print("Vytvořen testovací obrázek: /tmp/test_image.png")

# Transformace pro neuronové sítě
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Standardní velikost pro CNN
    transforms.ToTensor(),               # Převod na tensor [0,1]
    transforms.Normalize(                # ImageNet normalizace
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Aplikace transformace
img = Image.open("/tmp/test_image.png")
tensor = transform(img)
print(f"\nTransformovaný tensor:")
print(f"  Shape: {tensor.shape}")  # [3, 224, 224]
print(f"  Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")

# Batch pro síť
batch = tensor.unsqueeze(0)  # Přidání batch dimenze
print(f"  Batch shape: {batch.shape}")  # [1, 3, 224, 224]

# Data augmentace
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

print("\nData augmentace:")
print("  - RandomHorizontalFlip")
print("  - RandomRotation")
print("  - ColorJitter")
print("  - RandomResizedCrop")

# Ukázka více augmentací
img = Image.open("/tmp/test_image.png")
for i in range(3):
    augmented = aug_transform(img)
    print(f"  Augmented {i+1}: {augmented.shape}")
EOF

cat > "$LEARN/06-computer-vision/simple_cnn.py" << 'EOF'
#!/usr/bin/env python3
"""CNN pro klasifikaci obrázků."""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("=== CNN Klasifikace ===\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Transformace
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST dataset (automaticky se stáhne)
print("\nStahuji MNIST dataset...")
train_dataset = datasets.MNIST(
    root='/tmp/data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='/tmp/data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28->14
        x = self.pool(self.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel architecture:\n{model}")

# Training (jen 2 epochy pro ukázku)
print("\nTrénink...")
for epoch in range(2):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}")
    
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

# Evaluace
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()

accuracy = 100 * correct / len(test_dataset)
print(f"\nTest accuracy: {accuracy:.2f}%")
EOF

cat > "$LEARN/06-computer-vision/yolo_detection.py" << 'EOF'
#!/usr/bin/env python3
"""YOLO detekce objektů."""
from ultralytics import YOLO
import numpy as np
from PIL import Image

print("=== YOLO Detekce ===\n")

# Stažení YOLOv8 nano (nejmenší, rychlý)
print("Načítám YOLOv8 nano model...")
model = YOLO('yolov8n.pt')

# Vytvoření testovacího obrázku
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
Image.fromarray(img).save('/tmp/test_detect.jpg')

# Detekce
print("Spouštím detekci...")
results = model('/tmp/test_detect.jpg')

# Výsledky
print(f"\nVýsledky detekce:")
for r in results:
    boxes = r.boxes
    print(f"  Nalezeno objektů: {len(boxes)}")
    
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            print(f"    - {name}: {conf:.2f}")

# YOLO třídy
print(f"\nYOLO rozpoznává {len(model.names)} tříd:")
print(f"  {list(model.names.values())[:10]}...")

# Pro skutečné použití:
print("""
Pro detekci na vlastních obrázcích:
    results = model('your_image.jpg')
    results[0].save('result.jpg')  # Uloží s bounding boxy

Pro video:
    results = model('video.mp4')

Pro webcam:
    results = model(source=0, show=True)
""")
EOF

#===============================================================================
# 07 - AUDIO & SPEECH
#===============================================================================
sudo -u "$USER_REAL" mkdir -p "$LEARN/07-audio-speech"
cat > "$LEARN/07-audio-speech/README.md" << 'EOF'
# 07 - Audio & Speech

## Co se naučíš
- Speech-to-text (Whisper)
- Text-to-speech
- Audio zpracování

## Soubory
- `whisper_stt.py` - Speech to text
- `audio_basics.py` - Základy zpracování zvuku
EOF

cat > "$LEARN/07-audio-speech/whisper_stt.py" << 'EOF'
#!/usr/bin/env python3
"""Whisper Speech-to-Text (lokálně)."""
import torch

print("=== Whisper Speech-to-Text ===\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Použijeme faster-whisper (efektivnější)
try:
    from faster_whisper import WhisperModel
    
    print("\nNačítám Whisper tiny model...")
    model = WhisperModel("tiny", device=device, compute_type="float16" if device=="cuda" else "int8")
    
    print("""
Model načten! Pro transkripci:

    segments, info = model.transcribe("audio.mp3")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

Podporované formáty: mp3, wav, m4a, flac, ogg

Modely (větší = přesnější, pomalejší):
    tiny   - 39M params, ~10x realtime
    base   - 74M params, ~7x realtime  
    small  - 244M params, ~4x realtime
    medium - 769M params, ~2x realtime
    large  - 1550M params, ~1x realtime
""")
    
except ImportError:
    print("faster-whisper není nainstalován.")
    print("Zkusím openai-whisper...")
    
    import whisper
    
    print("\nNačítám Whisper tiny model...")
    model = whisper.load_model("tiny", device=device)
    
    print("""
Model načten! Pro transkripci:

    result = model.transcribe("audio.mp3")
    print(result["text"])

Pro segmenty s časem:
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f}s] {segment['text']}")
""")
EOF

cat > "$LEARN/07-audio-speech/audio_basics.py" << 'EOF'
#!/usr/bin/env python3
"""Základy zpracování zvuku."""
import numpy as np
import soundfile as sf

print("=== Audio Základy ===\n")

# Generování testovacího zvuku (sinusovka)
sample_rate = 22050
duration = 2.0
frequency = 440  # A4

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Uložení
sf.write('/tmp/test_audio.wav', audio, sample_rate)
print(f"Vytvořen testovací zvuk: /tmp/test_audio.wav")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Duration: {duration} s")
print(f"  Frequency: {frequency} Hz (A4)")

# Načtení
data, sr = sf.read('/tmp/test_audio.wav')
print(f"\nNačteno:")
print(f"  Shape: {data.shape}")
print(f"  Sample rate: {sr}")
print(f"  Duration: {len(data)/sr:.2f} s")

# Librosa pro pokročilejší analýzu
try:
    import librosa
    import librosa.display
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sr)
    print(f"\nMel spectrogram shape: {mel_spec.shape}")
    
    # MFCC features (užitečné pro speech)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    print(f"MFCC shape: {mfcc.shape}")
    
except ImportError:
    print("\nlibrosa není nainstalována pro pokročilé features")

print("""
Další možnosti:
- librosa: analýza, MFCC, spectrogramy
- soundfile: načítání/ukládání
- pydub: editace, konverze formátů
- torchaudio: PyTorch integrace
""")
EOF

chown -R "$USER_REAL:$USER_REAL" "$LEARN"

#===============================================================================
# README
#===============================================================================
cat > "$LEARN/README.md" << 'EOF'
# 🎓 AI Developer Learning Path

Kompletní průvodce pro AI vývojáře - vše lokálně bez placených účtů!

## Struktura

```
AI-Learning/
├── 01-python-ai/         # Python, NumPy, Pandas, vizualizace
├── 02-pytorch-basics/    # Tensory, autograd, první síť
├── 03-transformers-llm/  # Hugging Face, Ollama, embeddings
├── 04-rag-system/        # RAG, ChromaDB, vector search
├── 05-fine-tuning/       # LoRA, PEFT, příprava dat
├── 06-computer-vision/   # CNN, YOLO, zpracování obrázků
└── 07-audio-speech/      # Whisper, audio processing
```

## Jak začít

```bash
# Aktivuj AI prostředí
conda activate ai

# Přejdi do složky
cd ~/AI-Learning/01-python-ai

# Spusť příklad
python numpy_basics.py
```

## Pořadí studia

1. **01-python-ai** - Pokud neznáš NumPy/Pandas
2. **02-pytorch-basics** - Základy deep learning
3. **03-transformers-llm** - Moderní NLP a LLM
4. **04-rag-system** - Praktické AI aplikace
5. **05-fine-tuning** - Customizace modelů
6. **06-computer-vision** - Práce s obrázky
7. **07-audio-speech** - Práce se zvukem

## Požadavky

- Conda environment `ai`
- Pro LLM: běžící Ollama (`ollama serve`)
- GPU doporučeno ale ne nutné

## Lokální modely

```bash
# Ollama modely
ollama pull llama3.2      # 2GB, všeobecný
ollama pull codellama     # Pro kód
ollama pull nomic-embed-text  # Embeddings

# Hugging Face (stahují se automaticky)
# distilbert, distilgpt2, all-MiniLM-L6-v2
```

Vše běží lokálně bez API klíčů! 🚀
EOF

ok "========== AI LEARNING PATH VYTVOŘEN =========="
log "Složka: ~/AI-Learning/"
log "Začni: cd ~/AI-Learning/01-python-ai && python numpy_basics.py"

