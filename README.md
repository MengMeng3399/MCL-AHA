# MCL-AHA
# Multi-level Contrastive Learning with Adaptive Hypergraph Augmentation for Cloud API Recommendation
⭐ This code has been completely released ⭐

⭐ Overall framework of the MCL-AHA model⭐ 
<img width="5344" height="3113" alt="flowchart" src="https://github.com/user-attachments/assets/096a05fa-7e57-42f1-921c-69a90053bd18" />

Overall framework of the MCL-AHA model. MCL-AHA first constructs an invocation graph, a Mashup hypergraph, and a cloud API hypergraph. Node representations are learned via GNN and HGNN. Subsequently, multi-level contrastive learning is performed, comprising hypergraph contrastive learning (HCL) and cross-view contrastive learning (CVCL).

⭐ The HGA dataset refers to our previous work:[ https://github.com/528Lab/CAData](https://github.com/528Lab/CAData)⭐ 

⭐ The PWA dataset refers to: [https://github.com/kkfletch/API-Dataset](https://github.com/kkfletch/API-Dataset)⭐ 


### Environment

- **Python**: 3.9.16  
- **PyTorch**: 2.0.1  
- **Transformers**: 4.3x.x  
- Common dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `tqdm`


## BERT Embeddings (Textual Features)

We extract textual embeddings for Mashups/APIs using `bert_embedder.py` (HuggingFace `AutoTokenizer` / `AutoModel`).  
The extracted embeddings are used as the unstructured textual features in our model (Eq. (4)).

### Default settings
- Backbone: `bert-base-uncased`
- Tokenization: padding + truncation with `--max-length 128`
- Layer: `--layer last`
- Pooling: `--pooling mean` (masked mean pooling over non-padding tokens using the attention mask)
  - `--exclude-special-tokens` optionally excludes `[CLS]/[SEP]` from mean pooling
- Inference-only: BERT is used in inference mode (no fine-tuning) for embedding extraction
- Empty text: outputs a zero vector (`--on-empty zero`)

### Output files
The script produces:
- `bert_mashup_des.json`
- `bert_api_des.json`

Each file is a JSON mapping from **entity key** to embedding list (float32).  
> Note: Please ensure the key field used during embedding extraction matches the field used in training (e.g., `MashupName` / `ApiName`).

### Example
```bash
python bert_embedder.py \
  --mashup-json data/mashup.json \
  --api-json data/api.json \
  --model-name bert-base-uncased \
  --layer last \
  --pooling mean \
  --max-length 128 \
  --batch-size 64 \
  --seed 42 \
  --on-empty zero
