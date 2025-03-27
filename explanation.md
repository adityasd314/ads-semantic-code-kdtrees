# Semantic Code Search & Clustering Tool

## Overview

This is a **command-line tool** that allows developers to:

1. **Generate embeddings** for a given code repository.
2. **Search for code snippets** similar to a given query.
3. **Cluster similar code functions** based on embedding distance.

The tool is designed for **semantic search in codebases** using **machine learning-based embeddings** and efficient **data structures** for fast retrieval and clustering.

## ✨ Features

**AST-based Function Extraction:** Uses Tree-Sitter to extract functions/methods.
**State-of-the-art Embeddings:** Uses `sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net`.
**Efficient Search:** Uses **MinHeap** to retrieve Top-K results.
**Fast Nearest Neighbor Search:** Supports **KDTree**, **Suffix Tree**, and **PyTorch TopK**.
**Clustering with Similarity Matrix:** Uses **Agglomerative Clustering** to group similar functions.
**Clean UI for Query Results:** Allows direct navigation to the matched function in an editor.

## 🔧 Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/semantic-code-search.git
cd semantic-code-search
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Install Tree-Sitter (if not already installed)**

```bash
pip install tree-sitter
```

## 🚀 Usage

### **1. Generate Code Embeddings**

```bash
python main.py generate --repo /path/to/repo
```

🔹 This will:
Parse the repo using **Tree-Sitter** and extract function/method nodes.
Generate **BERT embeddings** for extracted functions.
Store the embeddings in the `.embeddings` directory.

---

### **2. Search for Similar Code Snippets**

```bash
python main.py search --query "find maximum subarray"
```

🔹 This will:
Compute the **query embedding**.
Compare it against **stored embeddings** using **cosine similarity**.
Use a **MinHeap** to efficiently fetch **Top-K** matches.
Display results in a **clean UI**, with links to the matched function.

---

### **3. Cluster Similar Code Functions**

```bash
python main.py cluster --method kdtree
```

🔹 This will:Normalize embeddings before clustering.Support different **data structures** for nearest-neighbor search:

- `KDTree` (default)
- `Suffix Tree`
- `PyTorch TopK`
  Construct a **similarity matrix** based on embedding distances.
  Perform **Agglomerative Clustering** to group similar functions.

---

## 🔬 How It Works

### **Step 1: Code Parsing using Abstract Syntax Tree (AST)**

- We use **Tree-Sitter** to parse source code into an **AST (Abstract Syntax Tree)**.
- We extract only relevant **function/method** definitions for embedding generation.

### **Step 2: Embedding Generation**

- We use `sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net` to generate **semantic embeddings**.
- These embeddings are stored in `.embeddings` for **fast retrieval**.

### **Step 3: Search Functionality**

- Query embeddings are **compared with stored embeddings** using **cosine similarity**.
- A **MinHeap** is used to efficiently retrieve **Top-K** results.

### **Step 4: Clustering Code Snippets**

- We support **three data structures** for **fast nearest-neighbor search**:
  1. **KDTree** – Used for fast multi-dimensional lookups.
  2. **Suffix Tree** – Useful for substring-based searches.
  3. **PyTorch TopK** – Optimized for GPU-based retrieval.
- A **similarity matrix** is constructed for clustering.
- **Agglomerative Clustering** is applied to **group similar functions**.

---

## 🛠 Supported Data Structures

| **Feature**                 | **Data Structure Used**                       |
| --------------------------------- | --------------------------------------------------- |
| **Search**                  | `MinHeap`                                         |
| **Nearest Neighbor Search** | `KDTree`, `Suffix Tree`, `PyTorch TopK`       |
| **Clustering**              | `Similarity Matrix`, `Agglomerative Clustering` |

---

## 🖥 Query UI

- The search results **display matched functions** in a **clean UI**.
- Clicking a result allows **jumping to the function in the editor** instantly.

---

## 🔗 References

- [Tree-Sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [BERT-Based Code Search](https://huggingface.co/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## Contributors

👤 **adityasd314** 
