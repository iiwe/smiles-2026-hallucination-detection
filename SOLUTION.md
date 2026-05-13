**Architecture**
To prevent severe overfitting on a small dataset with high-dimensional embeddings, a dimensionality reduction pipeline combined with a minimal linear classifier was chosen
- Features are first normalized using `StandardScaler`, and then reduced down to 50 dimensions using `UMAP(n_components=50)`
- Logistic Regression to the absolute minimum: `input_dim (50) -> 1`
- `self._net = nn.Sequential(nn.Linear(input_dim, 1))` logistic regression
- This extreme simplification prevents the model from memorizing the training data
- Returns a single logit which pairs perfectly with `BCEWithLogitsLoss`

**`fit()`**
- Dimensionality reduction is fitted using class labels. This forces umap to actively push truthful and hallucinated representations apart during compression
- `pos_weight` is dynamically calculated based on the negative/positive ratio and passed to `BCEWithLogitsLoss`
- `Adam(..., lr=1e-3, weight_decay=1.0)`: Heavy L2 regularization (`1.0`) is used to strictly penalize large weights keeping the linear classifier stable
- `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)` gracefully decays the learning rate if the loss plateaus.
- `epochs = 200` provides enough iterations for the small 50-dimensional space to converge

**Data Splitting**
- `test_size=0.15` is first reserved to create an untouched, global test set (`idx_test`) that remains the same across all folds
- The remaining 85% of the data is passed to `StratifiedKFold(n_splits=5)` creating 5 separate train/validation splits

**Feature Aggregation**
- Instead of taking just the last layer, it sweeps through the upper half of the model's layers with a step of 3 (`range(num_layers - 1, num_layers // 2 - 1, -3)`). Upper layers contain high-level semantic information
- It only extracts the last 30 valid tokens (`real_positions[-30:]`). Since the LLM's actual generated answer sits at the very end of the sequence, focusing only on these tokens strips away the noise of the prompt
- The last 30 tokens are combined for each selected layer and combine the results into a single flat vector

**Geometric Features**
- L2-norms of the last token on the last three layers (to check activation magnitudes)
- Cosine similarity between adjacent top layers (to measure representation drift)
