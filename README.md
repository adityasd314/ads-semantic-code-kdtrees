# Semantic Code Search & Clustering Tool

## Overview

This tool enables **semantic search in codebases** using **machine learning-based embeddings** and efficient **data structures** for fast retrieval and clustering.

### Features

- **AST-based Function Extraction**: Uses Tree-Sitter to extract functions/methods.
- **Machine Learning Embeddings**: Uses `sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net`.
- **Efficient Search Mechanism**: Uses **MinHeap** to retrieve Top-K results.
- **Fast Nearest Neighbor Search**: Supports **KDTree**, **Suffix Tree**, and **PyTorch TopK**.
- **Clustering with Similarity Matrix**: Uses **Agglomerative Clustering** to group similar functions.
- **Query UI**: Provides a clean UI that allows jumping to the matched function in the editor.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/semantic-code-search.git
cd semantic-code-search
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tree-Sitter (if not already installed)

```bash
pip install tree-sitter
```

## Usage

### Command-Line Interface

The tool provides several functionalities through a command-line interface (CLI). The commands available are:

#### 1. Generate Code Embeddings

```bash
python main.py -d --path-to-repo /path/to/repo
```

This command will:

- Parse the repository using **Tree-Sitter** to extract function/method nodes.
- Generate **BERT-based embeddings** for extracted functions.
- Store the embeddings in the `.embeddings` directory for future searches.

#### 2. Search for Similar Code Snippets

```bash
python main.py -p /path/to/repo --query "find maximum subarray"
```

This command will:

- Compute the **query embedding**.
- Compare it against **stored embeddings** using **cosine similarity**.
- Use a **MinHeap** to efficiently fetch **Top-K** matches.
- Display results in a **clean UI**, allowing direct navigation to the matched function.

#### 3. Cluster Similar Code Functions

```bash
python main.py -c --path-to-repo /path/to/repo --cluster-data-structure KDTrees
```

This command will:

- Normalize embeddings before clustering.
- Use the specified **data structure** for nearest-neighbor search.
- Construct a **similarity matrix** based on embedding distances.
- Apply **Agglomerative Clustering** to group similar functions.

## CLI Options

### **General Arguments**

| Option                             | Description                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `-p, --path-to-repo PATH`        | Path to the Git repository to search or embed.                                                              |
| `-m, --model-name-or-path MODEL` | Model to use for embedding generation (default:`sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net`). |
| `-d, --embed`                    | Generate or update embeddings for the codebase.                                                             |
| `-b, --batch-size BS`            | Batch size for embeddings generation (default: 32).                                                         |
| `-x, --file-extension EXT`       | Filter results by file extension (e.g., "py" for Python files).                                             |
| `-n, --n-results N`              | Number of results to return for search (default: 5).                                                        |
| `-e, --editor {vscode,vim}`      | Editor to open selected search results (default: vscode).                                                   |

### **Clustering Options**

| Option                                                                        | Description                                                        |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `-c, --cluster`                                                             | Enable clustering mode.                                            |
| `--cluster-max-distance THRESHOLD`                                          | Maximum distance for clustering (default: 0.2).                    |
| `--cluster-data-structure {KDTrees,torch.topk,suffix-trees,inverted-index}` | Data structure to use for clustering.                              |
| `--cluster-min-lines SIZE`                                                  | Ignore clusters with snippets smaller than this size (default: 0). |
| `--cluster-min-cluster-size SIZE`                                           | Ignore clusters smaller than this size (default: 2).               |
| `--cluster-ignore-identical`                                                | Ignore identical code/exact duplicates.                            |

## How It Works

### Step 1: Code Parsing using Abstract Syntax Tree (AST)

- Uses **Tree-Sitter** to parse source code into an **AST (Abstract Syntax Tree)**.
- Extracts only relevant **function/method** definitions for embedding generation.

### Step 2: Embedding Generation

- Utilizes `sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net` to generate **semantic embeddings**.
- Stores embeddings in `.embeddings` for **fast retrieval**.

### Step 3: Search Functionality

- Query embeddings are **compared with stored embeddings** using **cosine similarity**.
- A **MinHeap** is used to efficiently retrieve **Top-K** results.

### Step 4: Clustering Code Snippets

- Supports **three data structures** for **fast nearest-neighbor search**:
  - **KDTree** – Optimized for fast multi-dimensional lookups.
  - **Suffix Tree** – Useful for substring-based searches.
  - **PyTorch TopK** – Optimized for GPU-based retrieval.
- Constructs a **similarity matrix** for clustering.
- Applies **Agglomerative Clustering** to **group similar functions**.

## Data Structures Used

| Feature                           | Data Structure Used                                 |
| --------------------------------- | --------------------------------------------------- |
| **Search**                  | `MinHeap`                                         |
| **Nearest Neighbor Search** | `KDTree`, `Suffix Tree`, `PyTorch TopK`       |
| **Clustering**              | `Similarity Matrix`, `Agglomerative Clustering` |

## Query UI

- Displays matched functions in a structured interface.
- Clicking a result allows **jumping directly to the function in the editor**.

## References

- [Tree-Sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [BERT-Based Code Search](https://huggingface.co/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net)

## License

This project is licensed under the **MIT License**.
