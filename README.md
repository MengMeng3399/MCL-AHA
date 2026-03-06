# MCL-AHA
# Multi-level Contrastive Learning with Adaptive Hypergraph Augmentation for Cloud API Recommendation
⭐ This code has been completely released ⭐

⭐ Overall framework of the MCL-AHA model⭐ 
<img width="5344" height="3113" alt="flowchart" src="https://github.com/user-attachments/assets/096a05fa-7e57-42f1-921c-69a90053bd18" />

Overall framework of the MCL-AHA model. MCL-AHA first constructs an invocation graph, a Mashup hypergraph, and a cloud API hypergraph. Node representations are learned via GNN and HGNN. Subsequently, multi-level contrastive learning is performed, comprising hypergraph contrastive learning (HCL) and cross-view contrastive learning (CVCL).

⭐ The HGA dataset refers to our previous work:[ https://github.com/528Lab/CAData](https://github.com/528Lab/CAData)⭐ 

⭐ The PWA dataset refers to: [https://github.com/kkfletch/API-Dataset](https://github.com/kkfletch/API-Dataset)⭐ 


## Training and Optimization Details

To ensure full reproducibility, we provide the specific training configurations, hyperparameters, and optimization details utilized in our experiments.

### Hyperparameters for Contrastive Learning and Augmentation
- **Contrastive Temperature ($\tau$)**: 
  - Cross-view contrastive loss temperature: `0.6`
  - Hypergraph masked contrastive loss temperature: `0.5`
- **Augmentation Parameters (Eq. 16)**: 
  - Overall perturbation strength ($\mu_e$): `0.2`
  - Scaling parameter / Cutoff threshold ($\mu_t$): `1.0`
  - *(Note: Based on empirical tuning, combining these two parameters yields a robust structural masking probability bound of `0.2`, which ensures optimal performance while maintaining computational efficiency in our implementations).*

### Training Configurations & Early Stopping
- **Max Epochs**: The model is trained for a maximum of `400` epochs.
- **Early Stopping**: We employ an early stopping strategy to prevent overfitting. The training process is terminated if the primary evaluation metric (e.g., NDCG) on the validation set does not improve for `50` consecutive epochs. 
- **Batch Size**: `4096` for training and `2048` for testing (configurable via `--batch_size` and `--test_batch_size`).

### Optimization Details
- **Optimizer**: `Adam` (Adaptive Moment Estimation).
- **Learning Rate Scheduler**: The learning rate is kept constant throughout the training process (i.e., no scheduler is applied). The optimal initial learning rates are tuned via grid search (e.g., `1e-3` for PWA and `3e-4` for HGA).
- **L2 Regularization ($\lambda$)**: Applied to model embeddings to prevent overfitting, configured via the `--l2` argument (default is `1e-5`).
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



