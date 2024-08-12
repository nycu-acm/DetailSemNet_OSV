# DetailSemNet

DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration, ECCV 2024

## Abstract

Offline signature verification (OSV) is a frequently utilized technology in forensics. This paper proposes a new model, **DetailSemNet**, for OSV. Unlike previous methods that rely on holistic features for pair comparisons, our approach underscores the significance of fine-grained differences for robust OSV. We propose to match local structures between two signature images, significantly boosting verification accuracy. Furthermore, we observe that without specific architectural modifications, transformer-based backbones might naturally obscure local details, adversely impacting OSV performance. To address this, we introduce a **Detail-Semantics Integrator**, leveraging feature disentanglement and re-entanglement. This integrator is specifically designed to enhance intricate details while simultaneously expanding discriminative semantics, thereby augmenting the efficacy of local structural matching. We evaluate our method against leading benchmarks in offline signature verification. Our model consistently outperforms recent methods, achieving state-of-the-art results with clear margins. The emphasis on local structure matching not only improves performance but also enhances the model's interpretability, supporting our findings. Additionally, our model demonstrates remarkable generalization capabilities in cross-dataset testing scenarios. The combination of generalizability and interpretability significantly bolsters the potential of **DetailSemNet** for real-world applications.



## Environments

```text
pip install -r requirements.txt
```

