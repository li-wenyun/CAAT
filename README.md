<h1 align="center">Efficient Adversarial Training via Criticality-Aware Fine-Tuning</h1>


### Abstract:

Vision Transformer (ViT) models have achieved remarkable performance across various vision tasks, with scalability being a key advantage when applied to large datasets. This scalability enables ViT models to exhibit strong generalization capabilities. However, as the number of parameters increases, the robustness of ViT models to adversarial examples does not scale proportionally. Adversarial training (AT), one of the most effective methods for enhancing robustness, typically requires fine-tuning the entire model, leading to prohibitively high computational costs, especially for large ViT architectures. In this paper, we aim to robustly fine-tune only a small subset of parameters to achieve robustness comparable to standard AT. To accomplish this, we introduce Criticality-Aware Adversarial Training (CAAT), a novel method that adaptively allocates resources to the most robustness-critical parameters, fine-tuning only selected modules. Specifically, CAAT efficiently identifies parameters that contribute most to adversarial robustness. It then leverages parameter-efficient fine-tuning (PEFT) to robustly adjust weight matrices where the number of critical parameters exceeds a predefined threshold. Extensive experiments on three widely used adversarial learning datasets demonstrate that CAAT outperforms state-of-the-art lightweight AT methods with fewer trainable parameters, e.g, about 1.58\% improvement in mean Top-1 accuracy over FullLoRA-AT with approximately 43.9\% of its parameters.

------


## Getting started on CAAT:

### Install dependency:

We have tested our code on both Torch 1.8.0, and 1.10.0. Please install the other dependencies with the following code in the home directory:

```
pip install -r requirements.txt
```



#### Get parameter criticality:

```console
bash configs/vtab_mae_caat_lora_criticality.sh
bash configs/vtab_mae_caat_adapter_criticality.sh
```







#### PEFT with CAAT:

We have provided the following training code:


```console
bash configs/vtab_supervised_caat_lora.sh
bash configs/vtab_supervised_caat_adapter.sh
```






