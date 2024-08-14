# DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration

Meng-Cheng Shih, Tsai-Ling Huang, Yu-Heng Shih, Hong-Han Shuai, Hsuan-Tung Liu, Yi-Ren Yeh, and Ching-Chun Huang, “DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration,” European Conference on Computer Vision (ECCV), 2024.

## Abstract

Offline signature verification (OSV) is a frequently utilized technology in forensics. This paper proposes a new model, **DetailSemNet**, for OSV. Unlike previous methods that rely on holistic features for pair comparisons, our approach underscores the significance of fine-grained differences for robust OSV. We propose to match local structures between two signature images, significantly boosting verification accuracy. Furthermore, we observe that without specific architectural modifications, transformer-based backbones might naturally obscure local details, adversely impacting OSV performance. To address this, we introduce a **Detail-Semantics Integrator**, leveraging feature disentanglement and re-entanglement. This integrator is specifically designed to enhance intricate details while simultaneously expanding discriminative semantics, thereby augmenting the efficacy of local structural matching. We evaluate our method against leading benchmarks in offline signature verification. Our model consistently outperforms recent methods, achieving state-of-the-art results with clear margins. The emphasis on local structure matching not only improves performance but also enhances the model's interpretability, supporting our findings. Additionally, our model demonstrates remarkable generalization capabilities in cross-dataset testing scenarios. The combination of generalizability and interpretability significantly bolsters the potential of **DetailSemNet** for real-world applications.

![overview](https://github.com/nycu-acm/DetailSemNet_OSV/blob/main/fig/overview.png)

## Proposed Method

DetailSemNet is a novel approach for Offline Signature Verification (OSV) that emphasizes local patch features and structural matching. Unlike previous methods that rely on holistic features, DetailSemNet incorporates a Detail-Semantics Integrator (DSI) to enhance the model's ability to capture both detailed and semantic information. The DSI splits features into Semantic and Detail components, processing them separately through different branches: SemanticsAttend, SalientConv, and DetailConv. This design allows the model to retain more detailed information while expanding discriminative semantics. Additionally, the authors propose Structural Matching, which aligns local patch tokens to improve the model's ability to capture local discriminative features. The combination of DSI and Structural Matching enables DetailSemNet to perform more accurate comparisons between signature pairs, significantly boosting verification accuracy.

![patch](https://github.com/nycu-acm/DetailSemNet_OSV/blob/main/fig/patch.png "Three samples from the ChiSig dataset. Signature (a) originates from a different individual than signatures (b) and (c). At first glance, these signatures appear remarkably similar when viewed holistically. However, detailed analysis at the patch level reveals distinct differences between them, which are aspects frequently overlooked in previous methodologies.")

![DSI](https://github.com/nycu-acm/DetailSemNet_OSV/blob/main/fig/DSI.png "We employ filters to extract Low-frequency (LF), low-plus-middle frequency (LMF), and low-plus-high frequency (LHF) images. Our model captures both semantic pattern (low-frequency) and stroke structure and style detail (high-frequency) for improved verification. Leveraging high-frequency data enhances performance, unlike the baseline transformer model, which solely relies on low-frequency patterns and does not benefit from high-frequency features.")

## Conclusion

In this paper, we introduce **DetailSemNet**, a novel model for Offline Signature Verification (OSV) that emphasizes local patch features in Structural Matching, a shift from traditional holistic approaches. **DetailSemNet** also incorporates the Detail-Semantics Integrator (DSI) to enhance structural matching, effectively capturing detailed and semantic aspects. Our results demonstrate that **DetailSemNet** outperforms existing methods in both single-dataset and cross-dataset scenarios, highlighting its strong generalization capability and potential for real-world application. These findings indicate the effectiveness of combining the DSI module with Structural Matching in OSV models, positioning **DetailSemNet** as a significant advancement in forensic technology.

## Environments

```text
pip install -r requirements.txt
```

