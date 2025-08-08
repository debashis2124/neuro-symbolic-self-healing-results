## 📊 Experimental Results and Performance Analysis

We evaluate our **NeuroSymbolic Self-Healing framework** along three axes:

1. **Anomaly Detection Quality**
2. **Recovery Efficacy**
3. **End-to-End Resilience under Disruption**

Our full implementation is open-source and available on GitHub.  
📎 *[Add your GitHub repository link here]*

---

### ⚙️ Experimental Setup

We evaluated our neurosymbolic risk-scoring pipeline on a real-world **DoD contracts dataset** (~120,000 records). We defined a binary high-risk label:

- `label = 1` if `obligation > $150,000` or if `modification > 0`

We removed zero-variance features and standard-scaled the inputs.  
The data was stratified into:

- **60% training**
- **20% validation**
- **20% test**

We preserved index order for downstream recovery analysis.

---

### 📈 Training Dynamics and Score Distributions

- The **LSTM autoencoder** quickly learns benign supply chain patterns.
  - Its reconstruction loss drops from **> 0.03** to **< 0.005** within 15 epochs.
  - Both training and validation accuracy improve from ~70% to **>88%** with minimal overfitting.
  - *(See: `figures/figure_2a_autoencoder_loss.png` and `figure_2b_autoencoder_accuracy.png`)*

- The **symbolic rule engine**, though interpretable, suffers from overlapping score distributions between benign and attack flows, achieving only ~79–80% accuracy.
  - *(See: `figures/figure_3_score_distribution.png`)*

- When **neural and symbolic scores are fused** into a hybrid metric:
  - The attack score distribution shifts significantly.
  - Accuracy improves to ~87–89%
  - ROC-AUC improves to ~94–95%
  - *(See: `figures/figure_4_risk_score.png`)*

> **Hybrid fusion improves both precision and separation between benign and malicious instances.**

---

### 🤖 Detection Performance

- We trained a **64-unit LSTM Autoencoder** on benign samples.
- The **neural risk score** is defined as the reconstruction loss:  
  \( f_{NN}(x) = \text{MSE}(x, \hat{x}) \)

- The **symbolic score** \( g_{SR}(x) \) is computed from:
  - High obligation amount
  - High modification count
  - Contract Type (Award/IDV)
  - Top-2 PSC categories

- The scores are fused into:  
  \[
  R(x) = 0.6 \cdot f_{NN}(x) + 0.4 \cdot g_{SR}(x)
  \]

- We input `R(x)` + 3 raw features into an **MLP** for final classification.

- **Results (Table 1):**
  - Neural-Only ROC-AUC: **0.9560**
  - Symbolic-Only ROC-AUC: **0.8826**
  - NS Variants (Raw, Aug, Adv, Trans): All exceed **0.999** on synthetic and **>0.99** on real data

> **Insight:** Fusion of neural reconstruction error with symbolic rules greatly boosts detection quality.

---

### ♻️ Recovery Efficiency

We tested recovery across 3,000 samples using a symbolic planner.  
We measured the number of **compromised nodes** successfully rerouted.

- **Table 2 Summary**:
  - Neural-Only recovers: **45.2%**
  - Symbolic-Only: **29.3%**
  - NS–Raw, NS–Aug, NS–Adv: **~38–47%**
  
> **Note**: NS methods prioritize constraint-compliant rerouting, slightly lowering raw recovery numbers but improving safety and feasibility.

- *(See: `figures/compromised_recovery.png`)*

---

### 💥 Resilience Under Progressive Disruption

To evaluate system resilience, we simulate increasing levels of **node failures** and observe the resulting **network throughput** after rerouting.

- **Neuro-symbolic models** maintain **>90% throughput** even with 20% failures
- **Neural-Only and Symbolic-Only** baselines fall **below 80%**

> **Conclusion**: Integrating learned scores with formal symbolic planning maintains operational robustness even under aggressive fault scenarios.

- *(See: `figures/figure_training_preformance.png`, `figure_validation_preformance.png`, etc.)*

---

## 📎 Citation

```bibtex
@article{your_citation,
  title={Neuro-Symbolic Self-Healing for Secure Logistics},
  author={Your Name et al.},
  journal={...},
  year={2025}
}
```