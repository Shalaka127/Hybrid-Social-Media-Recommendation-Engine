





#  Hybrid Social Media Recommendation Engine

### Project Summary
This notebook presents a **hybrid recommendation engine** for social media posts, combining **collaborative and content-based methods**. It simulates a user–content ecosystem where engagement data and post attributes jointly drive recommendation quality.

### Purpose
To demonstrate a flexible recommendation framework capable of adapting to both **user behavior patterns** and **content similarity**, using a fully synthetic dataset for experimentation.

### Components
- **Collaborative Filtering (CF):**  
  Uses matrix factorization via TruncatedSVD to model user–item interaction patterns.
- **Content-Based Filtering (CBF):**  
  Extracts textual and categorical post features using TF-IDF and one-hot encoding.
- **Hybrid Integration:**  
  Final recommendation score is computed as  
  `Hybrid = α * CF_score + (1 - α) * CBF_score`,  
  where α defaults to 0.55 for balanced weighting.

### Synthetic Dataset
- 300 users × 800 posts  
- Post attributes: `platform`, `topic`, `title`, and `content`  
- User ratings (1–5 scale), biased by preferred topic  
- Generated dynamically for full reproducibility

### Notebook Workflow
1. Generate users, posts, and interaction data
2. Create sparse rating matrix for CF
3. Train SVD model (latent factor extraction)
4. Encode post metadata and text content
5. Combine CF + CBF for hybrid scoring
6. Visualize rating distribution and top-N recommendations

### Example Output
- List of top recommended posts per user  
- Topic and platform frequency visualizations  
- Insight into hybrid blending effects

### Requirements
```bash
pip install numpy pandas scikit-learn matplotlib scipy
