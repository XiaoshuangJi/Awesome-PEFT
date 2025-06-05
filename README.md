# Awesome-PEFT
### Papers
|Paper Title|Proposed Method|Venue and Year|Materials|
|------------------------------|------------------------------|------------------------------|------------------------------|
|Parameter-Efficient Transfer Learning for NLP|Adapter tuning|ICML 2019|[[PUB](https://proceedings.mlr.press/v97/houlsby19a.html)][[CODE](https://github.com/google-research/adapter-bert)]|
|Simple, Scalable Adaptation for Neural Machine Translation|Single Adapter|EMNLP 2019|[[PUB](https://aclanthology.org/D19-1165/)]|
|Exploring Versatile Generative Language Model Via Parameter-Efficient Transfer Learning|VLM|EMNLP 2020 Findings|[[PUB](https://aclanthology.org/2020.findings-emnlp.41/)]|
|AdapterFusion: Non-destructive task composition for transfer learning|AdapterFusion|EACL 2021|[[PUB](https://aclanthology.org/2021.eacl-main.39/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)]|
|AdapterDrop: On the efficiency of adapters in transformers|AdapterDrop|EMNLP 2021|[[PUB](https://aclanthology.org/2021.emnlp-main.626/)]|
|Prefix-Tuning: Optimizing Continuous Prompts for Generation|Prefix tuning|ACL 2021|[[PDF](https://xiangli1999.github.io/pdf/prefix_tuning.pdf)][[CODE](https://github.com/XiangLi1999/PrefixTuning)]|
|The Power of Scale for Parameter-Efficient Prompt Tuning| Prompt tuning|EMNLP 2021|[[PUB](https://aclanthology.org/2021.emnlp-main.243/)]|
|Parameter-Efficient Transfer Learning with Diff Pruning|DiffPruning|ACL 2021|[[PUB](https://aclanthology.org/2021.acl-long.378/)]|
|Compacter: Efficient low-rank hypercomplex adapter layers|Compacter|NeurIPS 2021|[[PDF](https://proceedings.neurips.cc/paper/2021/file/081be9fdff07f3bc808f935906ef70c0-Paper.pdf)][[CODE](https://github.com/rabeehk/compacter)]|
|Training Neural Networks with Fixed Sparse Masks|Fish-Mask|NeurIPS 2021|[[PDF](https://proceedings.neurips.cc/paper/2021/file/cb2653f548f8709598e8b5156738cc51-Paper.pdf)][[CODE](https://github.com/varunnair18/FISH)]|
|Towards a Unified View of Parameter-Efficient Transfer Learning|MAM Adapter, ParallelAdapter|ICLR 2022|[[PDF](https://xuezhemax.github.io/assets/publications/pdfs/iclr2022_towards.pdf)][[CODE](https://github.com/jxhe/unify-parameter-efficient-tuning)]|
|UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning|UniPELT|ACL 2022|[[PUB](https://aclanthology.org/2022.acl-long.433/)]|
|P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks|P-tuning v2|ACL 2022|[[PUB](https://aclanthology.org/2022.acl-short.8/)]|
|BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models|BitFit|ACL 2022|[[PUB](https://aclanthology.org/2022.acl-short.1/)]|
|FedPara: Low-rank Hadamard Product for Communication-Efficient Federated Learning|LoHa|ICLR 2022|[[PDF](https://openreview.net/pdf?id=d71n4ftoCBy)][[CODE](https://github.com/South-hw/FedPara_ICLR22)]|
|LoRA: Low-Rank Adaptation of Large Language Models|LoRA|ICLR 2022|[[PDF](https://openreview.net/pdf?id=nZeVKeeFYf9)][[CODE](https://github.com/microsoft/LoRA)]|
|Efficient Fine-Tuning of BERT Models on the Edge|FAR|ISCAS 2022|[[PUB](https://ieeexplore.ieee.org/abstract/document/9937567)][[PDF](https://arxiv.org/pdf/2205.01541)]|
|SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters|SparseAdapter|EMNLP 2022 Findings|[[PUB](https://aclanthology.org/2022.findings-emnlp.160/)][[CODE](https://github.com/Shwai-He/SparseAdapter)]|
|Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning|(IA)<SUP>3<SUP/>|NeurIPS 2022|[[PUB](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html)][[CODE](https://github.com/r-three/t-few)]|
|Adamix: Mixture-of adapter for parameter-efficient tuning of large language models|Adamix|-----|[[PDF](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/Mixture_of_Adapters-628fa6a57efd3.pdf)][[CODE](https://github.com/microsoft/AdaMix)]|
|Adaptive budget allocation for parameter-efficient fine-tuning|AdaLoRA|ICLR 2023|[[PDF](https://par.nsf.gov/servlets/purl/10471451)][[CODE](https://github.com/QingruZhang/AdaLoRA)]|
|QLoRA: Efficient Finetuning of Quantized LLMs|QLoRA|NeurIPS 2023|[[PUB](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)][[CODE](https://github.com/artidoro/qlora)]|
|Krona: Parameter efficient tuning with kronecker adapter|KronA|NeurIPS 2023 Workshop|[[PDF](https://neurips2023-enlsp.github.io/papers/paper_61.pdf)]|
|Controlling Text-to-Image Diffusion by Orthogonal Finetuning|OFT|NeurIPS 2023|[[PUB](https://oft.wyliu.com/)]|
|Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning|MPT|ICLR 2023|[[PUB](https://zhenwang9102.github.io/mpt.html)]|
|GPT Understands, Too|P-tuning|AI Open 2024|[[PUB](https://www.sciencedirect.com/science/article/pii/S2666651023000141)]|
|LoRA+: Efficient Low Rank Adaptation of Large Models|LoRA+|ICML 2024|[[PDF](https://openreview.net/pdf?id=NEv8YqBROO)][[CODE](https://github.com/nikhil-ghosh-berkeley/loraplus)]|
|Vector-based Random Matrix Adaptation|VeRA|ICLR 2024|[[PUB](https://dkopi.github.io/vera/)]|
|Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning|LN tuning|ICLR 2024|[[PDF](https://openreview.net/pdf?id=YR3ETaElNK)]|
|Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization|BOFT|ICLR 2024|[[PUB](https://boft.wyliu.com/)]|
|Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation|LyCORIS|ICLR 2024|[[PDF](https://arxiv.org/pdf/2309.14859)][[CODE](https://github.com/KohakuBlueleaf/LyCORIS)]|
|LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models|LoftQ|ICLR 2024|[[PDF](https://openreview.net/pdf?id=LzPWWPAdY4)][[CODE]](https://github.com/yxli2123/LoftQ)|
|ReLoRA: High-Rank Training Through Low-Rank Updates|ReLoRA|ICLR 2024|[[PDF](https://openreview.net/pdf?id=DLJznSp6X3)][[CODE](https://github.com/guitaricet/relora)]|
|DoRA: Weight-Decomposed Low-Rank Adaptation|DoRA|ICML 2024|[[PDF](https://openreview.net/pdf?id=3d5CIRG1n2)][[CODE](https://github.com/NVlabs/DoRA)]|
|RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation|RoSA|ICML 2024|[[PDF](https://openreview.net/pdf?id=FYvpxyS43U)][[CODE](https://github.com/IST-DASLab/RoSA)]|
|Parameter-Efficient Fine-Tuning with Discrete Fourier Transform|FourierFT|ICML 2024|[[PDF](https://arxiv.org/pdf/2405.03003)][[CODE](https://github.com/Chaos96/fourierft)]|
|MEFT:Memory-Efficient Fine-Tuning through Sparse Adapter|MEFT|ACL 2024|[[PDF](https://arxiv.org/pdf/2406.04984)][[CODE](https://github.com/CURRENTF/MEFT)]|
|Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning|COLA|ICML 2024|[[PDF](https://arxiv.org/pdf/2401.04151)]|
|HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning|HydraLoRA|NeurIPS 2024|[[PDF](https://openreview.net/pdf?id=qEpi8uWX3N)][[CODE](https://github.com/Clin0212/HydraLoRA)]|
|S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity|S2FT|NeurIPS 2024|[[PUB](https://infini-ai-lab.github.io/S2FT-Page/)]|
|VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks|VB-LoRA|NeurIPS 2024|[[PDF](https://arxiv.org/pdf/2405.15179)][[CODE](https://github.com/leo-yangli/VB-LoRA)]|
|Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation|HRA|NeurIPS 2024|[[PDF](https://arxiv.org/pdf/2405.17484)][[CODE](https://github.com/DaShenZi721/HRA)]|
|PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models|PiSSA|NeurIPS 2024|[[PDF](https://arxiv.org/pdf/2404.02948)][[CODE](https://github.com/GraphPKU/PiSSA)]|
|CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning|CorDA|NeurIPS 2024|[[PDF](https://openreview.net/pdf?id=Gi00NVru6n)][[CODE](https://github.com/iboing/CorDA)]|
|LoRA-GA: Low-Rank Adaptation with Gradient Approximation|LoRA-GA|NeurIPS 2024|[[PDF](https://openreview.net/pdf?id=VaLAWrLHJv)][[CODE](https://github.com/Outsider565/LoRA-GA)]|
|ReFT: Representation Finetuning for Language Models|ReFT|NeurIPS 2024|[[PDF](https://arxiv.org/pdf/2404.03592)][[CODE](https://github.com/stanfordnlp/pyreft)]|
|One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation|EVA|NeurIPS 2024 Workshop|[[PDF](https://openreview.net/pdf?id=X6AOzi82oo)]|
|SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors|SVFT|NeurIPS 2024 Workshop|[[PDF](https://openreview.net/pdf?id=DOUskwCqg5)]|
|ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models|ALoRA|NAACL 2024|[[PDF](https://aclanthology.org/2024.naacl-long.35.pdf)]|
|X-LoRA: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design|X-LoRA|APL Machine Learning 2024|[[PUB](https://pubs.aip.org/aip/aml/article/2/2/026119/3294581/X-LoRA-Mixture-of-low-rank-adapter-experts-a)][[PDF](https://arxiv.org/pdf/2402.07148)]|
|MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning|MiLoRA|NAACL 2025|[[PDF](https://arxiv.org/pdf/2406.09044)][[CODE](https://github.com/sufenlp/MiLoRA)]|
|HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models|HiRA|ICLR 2025|[[PDF](https://openreview.net/pdf?id=TwJrTz9cRS)][[CODE](https://github.com/hqsiswiliam/hira)]|
|BeamLoRA: Beam-Constraint Low-Rank Adaptation|BeamLoRA|ACL 2025|[[PDF](https://arxiv.org/pdf/2502.13604)]|
|LoRA-Pro: Are Low-Rank Adapters Properly Optimized?|LoRA-Pro|ICLR 2025|[[PDF](https://openreview.net/pdf?id=gTwRMU3lJ5)]|


### PEFT with other techs
|Paper Title|Venue and Year|Materials|
|------------------------------|------------------------------|------------------------------|
|Efficient Federated Learning for Modern NLP|MobiCom 2023|[[PUB](https://dl.acm.org/doi/abs/10.1145/3570361.3592505)][[PDF](https://www.caidongqi.com/pdf/AutoFedNLP.pdf)][[CODE](https://github.com/UbiquitousLearning/FedAdapter)]|
|FedPETuning: When Federated Learning Meets the Parameter-Efficient Tuning Methods of Pre-trained Language Models|ACL 2023 Findings|[[PUB](https://aclanthology.org/2023.findings-acl.632/)][[CODE](https://github.com/SMILELab-FL/FedPETuning)]|
|Fine-Tuning Language Models with Just Forward Passes|NeurIPS 2023|[[PDF](https://arxiv.org/pdf/2305.17333)][[CODE](https://github.com/princeton-nlp/MeZO)]|
|SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models|NeurIPS 2023|[[PDF](https://openreview.net/pdf?id=06quMTmtRV)]|
|LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition|COLM 2024|[[PDF](https://arxiv.org/pdf/2307.13269)][[CODE](https://github.com/sail-sg/lorahub)]|
|FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations|NeurIPS 2024|[[PDF](https://openreview.net/pdf?id=TcCorXxNJQ)][[CODE](https://github.com/ATP-1010/FederatedLLM)]|
|Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources|NeurIPS 2024|[[PDF](https://arxiv.org/pdf/2402.11505)]|
|Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning|ICLR 2024|[[PDF](https://openreview.net/pdf?id=EvDeiLv7qc)][[CODE](https://github.com/Cohere-Labs-Community/parameter-efficient-moe)]|
|FwdLLM: Efficient Federated Finetuning of Large Language Models with Perturbed Inferences|USENIX ATC 2024|[[PUB](https://www.usenix.org/conference/atc24/presentation/xu-mengwei)]|
|Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark|ICML 2024|[[[PUB](https://sites.google.com/view/zo-tutorial-aaai-2024/)][PDF](https://arxiv.org/pdf/2402.11592)][[CODE](https://github.com/ZO-Bench/ZO-LLM)]|
### Libraries
* AdapterHub:
  * Address: https://adapterhub.ml/
  * Description: A Unified Library for Parameter-Efficient and Modular Transfer Learning
* PEFT:
  * Address: https://huggingface.co/docs/peft/index
  * Description: State-of-the-art Parameter-Efficient Fine-Tuning.
* LLM-Adapters:
  * Address: https://github.com/AGI-Edgerunners/LLM-Adapters
  * Description: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models
<!--## Star History

<a href="https://github.com/XiaoshuangJi/Awesome-PEFT&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=XiaoshuangJi/Awesome-PEFT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=XiaoshuangJi/Awesome-PEFT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=XiaoshuangJi/Awesome-PEFT&type=Date" />
  </picture>
</a>-->
