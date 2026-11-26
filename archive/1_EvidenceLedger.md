# Phostream

**1\. PAPER IDENTIFICATION**

&nbsp;

* **Title:** PhoStream: Benchmarking Real-World Streaming for Omnimodal Assistants in Mobile Scenarios.  
* **Authors:** Xudong Lu, Huankang Guan, Yang Bo, Jinpeng Chen, Xintong Guo, Shuhan Li, Fang Liu, Peiwen Sun, Xueying Li, Wei Zhang, Xue Yang, Rui Liu, Hongsheng Li.  
* **Publication year and venue:** 2026, arXiv preprint.  
* **Primary research problem:** Current multimodal large language models fail to operate as continuous mobile assistants because they cannot reliably decide when to respond to streaming audio-visual inputs in dynamic environments.

**2\. EXISTING APPROACH** The paper introduces an evaluation framework and dataset rather than proposing a new compute-saving model architecture. It tests how existing proprietary and open-source models process continuous environmental data.

&nbsp;

* **Input modalities:** Video and audio.  
* **Inputs format:** Continuous streaming video updated at 1-second intervals.  
* **Processing type:** Interactive and streaming.  
* **Model architecture:** The framework acts as an inference wrapper around existing models (e.g., Gemini 3 Pro, Qwen3-Omni), enforcing a sliding memory window to manage context.

**3\. VISUAL-PROCESSING MECHANISM** A. **When is a frame acquired?** The stream provides visual updates every 1 second. B. **When is expensive visual encoding performed?** Not established; the paper tests existing models as black boxes but feeds them updates continuously every second. C. **Does the method select or discard frames?** It discards older frames by dropping history outside a 60-second sliding window. D. **Does it select or discard tokens?** Not established. E. **Does it change resolution, crops, or spatial detail?** Not established. F. **Does it retain or discard historical visual information?** It retains the most recent 60 seconds of visual history and discards older data. G. **What determines these decisions?** A predefined temporal boundary. H. **Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** The temporal memory policy is fixed.

&nbsp;

**4\. COMPUTATIONAL BENEFIT**

&nbsp;

* **Resource reduced:** The pipeline targets memory usage reduction by restricting active visual context to a strict 60-second window.  
* **Quantitative results:** The paper does not provide computational benchmarks, focusing entirely on task-accuracy metrics.

**5\. TASK PERFORMANCE**

&nbsp;

* **Tasks and datasets evaluated:** The benchmark tests Instant, Backward, and Forward visual question answering using the PhoStream dataset, which contains 5,572 QA pairs across mobile scenarios.  
* **Metrics reported:** LLM-as-a-Judge scores (scaled 0-100), Early Response rates, No Response rates, and Partly Correct rates.  
* **Impact of altered visual processing:** The authors ablate the audio modality. Removing audio decreases accuracy on Instant and Backward tasks. Conversely, removing audio slightly improves Forward task performance by mitigating the models' tendency to answer prematurely.

**6\. LIMITATIONS**

&nbsp;

* **A. Explicitly acknowledged:** The authors emphasize this is an evaluation benchmark, not a deployable software system intended for live user interaction.  
* **B. Supported by experiments:** Evaluated models exhibit a severe "Early Response" bias on Forward tasks, frequently predicting answers before the necessary visual evidence appears in the stream.  
* **C. Inferred:** The fixed 60-second visual memory window inherently prevents models from succeeding on queries that require aggregating visual evidence separated by long delays.  
* **D. Unestablished:** The paper does not establish algorithmic solutions to fix the timing inaccuracies or reduce the computational cost of the continuous 1-second polling.

&nbsp;

**7\. EVIDENCE TABLE**

&nbsp;

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| Models consistently answer prematurely on proactive tasks. | Experimental Result | Table 3&nbsp; | Validates the need for dynamic, query-dependent processing delays. |
| Visual context is artificially restricted to 60 seconds. | Implementation Choice | Section 3.4 / Figure 5&nbsp; | Highlights a baseline heuristic for bounding memory. |
| Audio inclusion increases premature guessing. | Ablation Study | Table 4&nbsp; | Demonstrates how multi-modal inputs impact visual decision timing. |
| The pipeline streams updates at a fixed 1-second interval. | Implementation Choice | Section 3.4&nbsp; | Serves as a baseline fixed-rate acquisition strategy. |

&nbsp;

**8\. ONE-PARAGRAPH SUMMARY** This paper introduces PhoStream, a benchmark designed to evaluate multimodal models acting as mobile assistants within continuous audio-visual streams. It tests whether models can accurately deduce *when* to process information and respond to open-ended queries based on changing context. By evaluating existing models with a fixed 1-second visual update interval and a 60-second sliding memory window, the authors show that current architectures exhibit a critical "Early Response" bias, habitually guessing answers before the required visual cues manifest. While the paper does not develop adaptive compute mechanisms, its findings directly support our research goal: models urgently need dynamic, query-dependent policies to manage visual processing efficiently without prematurely forcing inference.

&nbsp;

# FastV

&nbsp;

**1\. PAPER IDENTIFICATION**

&nbsp;

* **Title:** FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference.  
* **Authors:** Liang Chen, Haozhe Zhao, Tianyu Liu, Shuang Bai, Junyang Lin, Chang Zhou, Baobao Chang.  
* **Publication year and venue:** 2024, arXiv preprint.  
* **Primary research problem:** Large Vision-Language Models (LVLMs) incur high computational costs during inference because they process thousands of visual tokens, yet the attention mechanism in the model's deep layers assigns extremely low importance to these visual tokens compared to text.

**2\. EXISTING APPROACH**

&nbsp;

FastV is a plug-and-play inference acceleration technique that selectively drops visual tokens inside the transformer layers of an LVLM. By identifying that the model extracts most necessary visual features in the earliest layers and ignores the raw visual tokens later, FastV ranks visual tokens by their attention scores at a specific early layer and completely discards the lowest-scoring tokens for all subsequent layers, avoiding unnecessary computations.

&nbsp;

* **Input modalities:** Image and Text, as well as Video (processed as a sequence of multiple image frames).  
* **Input format:** Offline images and pre-recorded videos.  
* **Processing format:** Offline inference for static queries.  
* **Model architecture:** Autoregressive transformer-based Large Vision-Language Models (e.g., LLaVA, QwenVL-Chat) consisting of a pre-trained visual encoder (like CLIP-ViT) connected to a Large Language Model decoder.

**3\. VISUAL-PROCESSING MECHANISM**

&nbsp;

* **A. When is a frame acquired?** *Explicitly stated:* Frames are acquired offline before the inference sequence begins. *Cannot be established:* How the system acquires frames in a live, real-time camera stream.  
* **B. When is expensive visual encoding performed?** *Explicitly stated:* Visual encoding (via CLIP-ViT) is performed once at the very beginning of the forward pass before the tokens enter the LLM.  
* **C. Does the method select or discard frames?** *Explicitly stated:* It does not discard entire frames; it maps all provided images or video frames into tokens.  
* **D. Does it select or discard tokens?** *Explicitly stated:* Yes, it discards (prunes) a fixed percentage (R%) of visual tokens at a specific transformer layer (layer K).  
* **E. Does it change resolution, crops, or spatial detail?** *Explicitly stated:* FastV itself does not alter input resolution. However, the paper proves that using FastV allows models to process higher-resolution inputs at the same computational cost as lower-resolution baselines.  
* **F. Does it retain or discard historical visual information?** *Explicitly stated:* For video, all frames are fed into the context window as tokens, and FastV drops uninformative tokens across all frames based on attention, rather than sequentially discarding temporal history.  
* **G. What determines these decisions?** *Explicitly stated:* The pruning decision is determined by the average attention score a visual token receives from all other tokens in the designated filtering layer.  
* **H. Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The pruning ratio is a fixed heuristic (e.g., exactly 50% dropped at layer 2), but the selection of *which* tokens to drop is query-dependent because text instruction tokens influence the visual attention scores. *Cannot be established:* Dynamic policies based on scene changes or model uncertainty.

**4\. COMPUTATIONAL BENEFIT**

&nbsp;

* **Resource reduced:** FastV reduces theoretical FLOPs, inference latency, and the number of tokens processed by the Feed-Forward Network (FFN) and Multi-Head Attention (MHA) modules in the deep layers.  
* **Quantitative results:** FastV achieves a 45% reduction in FLOPs for LLaVA-1.5-13B. In latency tests on A-OKVQA, the processing time for LLaVA-13B dropped from 0.539 seconds per example to 0.341 seconds, making it as fast as a 7B parameter model.

**5\. TASK PERFORMANCE**

&nbsp;

* **Evaluated tasks and datasets:** Image captioning (Nocaps, Flickr30k), visual question answering (A-OKVQA, OCR-VQA), embodied reasoning (PCA-Bench), fine-grained multimodal reasoning (MMMU, MME, MMVet, SeedBench, SciQA-IMG), and Video QA (TGIF, MSVD, MSRVTT).  
* **Reported metrics:** CIDEr, Accuracy, Rouge-L, and specific benchmark sub-scores (Perception, Cognition, Action).  
* **Impact of altered processing:** Dropping 50% of visual tokens after layer 2 results in virtually no loss in accuracy or temporal understanding across benchmarks. Furthermore, for Video QA tasks, performance actually improves slightly because pruning filters out redundant temporal information.

**6\. LIMITATIONS**

&nbsp;

* **A. Explicitly acknowledged:** FastV causes slightly more performance degradation on InstructBLIP compared to LLaVA because InstructBLIP’s Q-Former architecture already compresses visual tokens prior to the LLM, meaning further pruning risks immediate information loss.  
* **B. Supported by experiments:** Pruning visual tokens randomly instead of ranking them by attention causes significant performance drops. Attempting to prune text tokens (system prompts or instructions) breaks the model entirely.  
* **C. Potential limitations inferred:** Because FastV uses a hard-coded pruning percentage (e.g., always 50%), it may fail on highly complex, dense visual scenes where more than 50% of the visual tokens contain necessary, non-redundant details.  
* **D. Important questions not established:** The paper does not establish how token-pruning behaves during a continuous, real-time conversational stream where the user's intent and the camera's visual scene update continuously.

**7\. EVIDENCE TABLE**

&nbsp;

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| Deep layers assign an average attention score to image tokens that is only 0.21% of the score assigned to system prompts. | Quantitative Observation | Page 2, Section 1&nbsp; | Highlights that retaining dense visual information throughout full processing depth is mathematically wasteful. |
| FastV reduces FLOPs by 45% with nearly no performance loss for a 13B model. | Benchmark Result | Page 1, Figure 1; Page 9, Table 1&nbsp; | Demonstrates the immense computational trade-off available via adaptive visual retention. |
| Inference latency per example drops from 0.539s to 0.341s (LLaVA-13B on A-OKVQA). | Latency Measurement | Page 10, Table 4&nbsp; | Proves that altering visual processing directly impacts real-time response speed. |
| Video QA performance actually improves slightly with token pruning. | Benchmark Result | Page 11, Table 6&nbsp; | Suggests that discarding excess visual tokens reduces contextual interference. |
| Random visual token pruning or text token pruning significantly degrades accuracy. | Ablation Study | Page 12, Table 7&nbsp; | Proves visual reduction must be query-driven and attention-informed, not arbitrary. |

&nbsp;

**8\. ONE-PARAGRAPH SUMMARY**

&nbsp;

The FastV paper demonstrates that Large Vision-Language Models (LVLMs) process visual information highly inefficiently, directing the vast majority of deep-layer attention to text prompts rather than image tokens. To exploit this, the authors introduce a heuristic mechanism that prunes a fixed percentage of visual tokens early in the decoding process based on their received attention scores, effectively bypassing subsequent self-attention and feed-forward computations. This approach dramatically reduces inference latency and FLOPs without compromising accuracy across spatial and temporal reasoning tasks. For our research into real-time adaptive visual inference, FastV proves that deep visual retention is largely redundant once early cross-modal grounding occurs. However, because its pruning policy is rigidly fixed and evaluated entirely offline, it leaves unexplored how an interactive system might dynamically adjust its visual acquisition and retention in response to continuous scene changes and shifting user uncertainty.

&nbsp;

# IVTP

1. PAPER IDENTIFICATION  
* **Title:** IVTP: Instruction-guided Visual Token Pruning for Large Vision-Language Models.  
* **Authors:** Kai Huang, Hao Zou, Ye Xi, BoChen Wang, Zhen Xie, and Liang Yu.  
* **Publication year and venue:** 2024 (implied by references to recent LVLMs), Alibaba Group; unverified publication venue (likely an arXiv preprint or conference submission).  
* **Primary research problem:** Large Vision-Language Models (LVLMs) suffer from high computational costs and token budget occupancy due to lengthy visual token sequences, making practical application difficult.  
2. EXISTING APPROACH

The paper introduces Instruction-guided Visual Token Pruning (IVTP), a two-stage mechanism to drop redundant visual tokens during inference. First, a Group-wise Token Pruning (GTP) module uses attention rollout within the frozen visual encoder to discard tokens with low visual informative content. Second, IVTP extends this pruning into the early layers of the Large Language Model (LLM) by using a "pseudo CLS token." This pseudo token is generated using a frozen CLIP text encoder to aggregate semantic information from the user's text instruction, allowing the LLM to selectively drop visual tokens that are irrelevant to the specific query.

&nbsp;

* **Input modalities:** Image and Text.  
* **Input arrival format:** Static images and offline text queries.  
* **Visual processing type:** Offline/inference (designed as a plug-and-play module for pre-trained models during inference, though evaluated in retraining modes as well).  
* **Model architecture:** LVLMs based on a frozen Visual Transformer (ViT), a projection layer, a decoder-only LLM (specifically LLaVA-1.5 built on Vicuna), and an auxiliary frozen CLIP text encoder to process instructions.  
3. VISUAL-PROCESSING MECHANISM

A. **When is a frame acquired?** *Explicitly stated:* A single image frame is acquired and processed at the beginning of the pipeline. *Cannot be established:* How or when frames would be acquired in a real-time, continuous video stream.

B. **When is expensive visual encoding performed?** *Explicitly stated:* At the initial forward pass through the frozen visual encoder (ViT) before the tokens enter the LLM.

C. **Does the method select or discard frames?** *Explicitly stated:* No, it operates on static single images, not sequential video frames.

D. **Does it select or discard tokens?** *Explicitly stated:* Yes, it discards patch tokens hierarchically at the topmost layer of predefined "groups" in both the ViT and the LLM.

E. **Does it change resolution, crops, or spatial detail?** *Explicitly stated:* It reduces fine-grained spatial detail by permanently discarding regions (patches) of the image deemed uninformative. It does not dynamically alter the initial input resolution or crop the image prior to tokenization.

F. **Does it retain or discard historical visual information?** *Explicitly stated:* It discards visual information that is not visually salient or semantically relevant to the text prompt. *Cannot be established:* How it handles historical visual information over time or across multiple turns.

G. **What determines these decisions?** *Explicitly stated:* In the ViT, decisions are based on aggregated attention weights (attention rollout) between patch tokens and the visual CLS token. In the LLM, decisions are based on the attention between the remaining visual tokens and a text-derived "pseudo CLS token".

H. **Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The policy is *query-dependent* (guided by the textual instruction via the CLIP encoder) and *heuristic* (using attention rollout scores and predefined drop ratios). *Cannot be established:* Policies based on uncertainty or dynamic scene changes.

&nbsp;

4. COMPUTATIONAL BENEFIT  
* **Targeted resources:** Number of visual tokens, and total model computational complexity (measured in TFLOPs).  
* **Quantitative results:** IVTP reduces the number of visual tokens by 88.9% (from 576 to 64 tokens). This token reduction decreases the total computational complexity (TFLOPs) by over 46%.  
5. TASK PERFORMANCE  
* **Tasks and datasets:** Evaluated on 12 standard multimodal benchmarks including VQAv2, GQA, VisWiz, SQA, VQA-T, POPE, MME, MMMU, and SEED.  
* **Metrics reported:** Average accuracy (%) and TFLOPs.  
* **Impact of altered processing:** Dropping 88.9% of visual tokens results in only an average 1.0% accuracy drop across the 12 benchmarks. When scaling the token reduction more aggressively (e.g., down to 16 tokens), IVTP significantly outperforms baseline pruning methods by approximately 5%, proving that query-guided retention better preserves necessary visual semantics.  
6. LIMITATIONS

A. **Explicitly acknowledged:** When the visual token count is reduced too aggressively (e.g., to 16 tokens), relying solely on token relevance and uniqueness is inadequate, leading to noticeable performance degradation despite the instruction-guided mechanism. B. **Supported by experiments:** Extending purely visual pruning methods (like TopK or ToMe) directly into the LLM without textual instruction guidance fails to yield significant performance improvements, demonstrating that representation similarity alone is insufficient for sparse token retention. C. **Potential limitations inferred:** The method requires computing similarities between CLIP text embeddings and visual CLS tokens to filter the instruction. If a user query contains novel vocabulary, heavy slang, or complex reasoning steps that do not map cleanly to the CLIP text encoder's latent space, the generation of the "pseudo CLS token" may fail, resulting in the accidental pruning of critical visual evidence. D. **Unestablished questions:** The paper does not establish how this method performs on continuous video streams, how real-time latency fluctuates dynamically, or how token pruning impacts multi-turn conversations where the relevance of visual patches might change based on follow-up questions.

&nbsp;

7. EVIDENCE TABLE

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| IVTP reduces visual tokens by 88.9% and TFLOPs by \>46% with only a 1.0% accuracy drop. | Experimental Result | Page 1, Abstract; Page 9, Sec 4.1; Page 10, Table 1&nbsp; | Quantifies the substantial computational savings possible through aggressive, adaptive token retention. |
| Query-guided token pruning within the LLM outperforms purely visual pruning methods. | Ablation Study | Page 12, Table 6 and Table 7&nbsp; | Demonstrates that visual processing decisions *must* respond to the user's query for optimal efficiency. |
| Group-wise attention rollout yields more stable token importance scores than single-layer attention. | Ablation Study | Page 12, Table 4&nbsp; | Provides a specific heuristic mechanism for deciding which visual information should be retained. |
| The LLM accounts for the vast majority of the LVLM's computational complexity. | Complexity Analysis | Page 13, Table 9&nbsp; | Highlights that token reduction is most impactful when applied *before* or *early within* the LLM phase. |

&nbsp;

8. ONE-PARAGRAPH SUMMARY

The IVTP paper demonstrates that the computational cost of Large Vision-Language Models can be drastically reduced by adaptively pruning visual tokens based on both visual salience and query relevance. By implementing a two-stage process—dropping visually uninformative tokens in the ViT and semantically irrelevant tokens in the LLM via a text-guided "pseudo CLS token"—the authors achieve a 46% compute reduction while retaining 99% of baseline task accuracy. For our research into real-time adaptive visual inference, this paper proves that dynamic, instruction-guided retention of visual information is vastly superior to static or purely visual downsampling. However, because IVTP is evaluated exclusively on static images and single-turn queries, it leaves open the challenge of how to adapt these query-dependent pruning heuristics for continuous, real-time multimodal data streams where user intent and scene context evolve dynamically.

&nbsp;

# ELLSA

**1\. PAPER IDENTIFICATION**

&nbsp;

* **Title:** END-TO-END LISTEN, LOOK, SPEAK AND ACT  
* **Authors:** Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang  
* **Publication year and venue:** 2026, arXiv preprint  
* **Primary research problem:** Existing embodied AI models suffer from a modality divide, operating either as disembodied conversationalists or as "deaf and mute" actors incapable of real-time, full-duplex multimodal interactions (e.g., speaking while acting, responding to action barge-ins).

**2\. EXISTING APPROACH**

&nbsp;

The paper introduces ELLSA, which utilizes a Self-Attention Mixture-of-Experts (SA-MoE) architecture. Instead of relying on a single dense model that suffers from modality interference, ELLSA routes different modalities to specialized experts (a Speech Expert and an Action Expert) that process interleaved multimodal sequences within discrete time blocks. These distinct experts interact seamlessly by sharing a unified attention backbone and Key-Value (KV) cache.

&nbsp;

* **Input modalities:** Speech, vision (images/video frames), text, and action (proprioception).  
* **Input format:** Inputs arrive as continuous streams of audio, video frames, and text instructions.  
* **Processing format:** The model operates in a streaming, interactive, full-duplex manner, processing inputs and outputs concurrently within fixed 1-second time blocks.  
* **Model architecture:** SA-MoE pairs a Speech Expert (Mamba encoder \+ LLaMA-3.1-8B-Instruct) and an Action Expert (Emu3-VisionTokenizer \+ Emu3-Base) connected via a shared self-attention mechanism.

**3\. VISUAL-PROCESSING MECHANISM**

&nbsp;

* **A. When is a frame acquired?** *Explicitly stated:* Visual frames (a front view and a gripper view) are acquired sequentially once per fixed 1-second time block.  
* **B. When is expensive visual encoding performed?** *Explicitly stated:* Encoding happens at each 1-second block as the image tokens are interleaved into the multimodal sequence between the speech and text tokens.  
* **C. Does the method select or discard frames?** *Explicitly stated:* It does not discard frames dynamically; it processes the designated frames for every time block.  
* **D. Does it select or discard tokens?** *Cannot be established:* The paper does not mention adaptively discarding visual tokens; it relies on the Emu3-VisionTokenizer output for each frame.  
* **E. Does it change resolution, crops, or spatial detail?** *Cannot be established:* There is no mention of dynamic resolution or cropping policies.  
* **F. Does it retain or discard historical visual information?** *Explicitly stated:* It deliberately discards historical visual and action information older than two seconds to constrain the sequence length.  
* **G. What determines these decisions?** *Explicitly stated:* The decisions are determined by hardcoded architectural constraints (fixed 1-second blocks and a rigid 2-second history window) rather than dynamic scene understanding.  
* **H. Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The visual acquisition and retention policy is entirely fixed.

**4\. COMPUTATIONAL BENEFIT**

&nbsp;

* **Resource reduced:** The architecture primarily attempts to reduce training computation (via modality-specific expert initialization) and memory overhead (by strictly limiting visual history to two seconds). A secondary goal is latency reduction via the streaming full-duplex design.  
* **Quantitative results:** By replacing a 1-second time block with a 0.48-second time block, ELLSA reduces per-block Speech-to-Speech latency from 854ms to 455ms, and Speech-to-Action latency from 786ms to 428ms. The SA-MoE architecture also drastically outperforms a single dense model under the same training constraints (e.g., 74.7% vs 62.7% on Llama Q.), demonstrating higher parameter and data efficiency.

**5\. TASK PERFORMANCE**

&nbsp;

* **Tasks and datasets evaluated:** Speech interaction (Llama Q., Web Q., TriviaQA, AlpacaEval) and Robotic Manipulation (LIBERO, CALVIN). The authors also evaluate custom full-duplex tasks: Dialogue/Action turn-taking, Defective instruction rejection, Speaking-while-acting, Context-grounded VQA, and Action barge-ins.  
* **Metrics reported:** Accuracy (%), Task Success Rate (%), GPTScore (1-5 scale), and Latency (ms).  
* **Impact of altered processing:** Altering the visual/action temporal processing window from 1.0 second to a finer-grained 0.48 seconds severely degrades action performance (e.g., LIBERO SPATIAL success rate drops from 95.4% to 91.0%) because the shorter time blocks disrupt the temporal coherence of action sequences. Furthermore, when forced to perform concurrent visual-action tasks and speech-generation tasks (speaking-while-acting), task success drops across complex benchmarks (e.g., LIBERO LONG drops from 84.4% to 73.2%) due to cross-modal distraction.

**6\. LIMITATIONS**

&nbsp;

* **A. Explicitly acknowledged:** The model is constrained to simulated environments and lacks advanced conversational dynamics like user backchanneling. Performance declines when simultaneously speaking and acting due to distraction.  
* **B. Supported by experiments/ablations:** High-frequency visual/action sampling (0.48s time blocks) breaks action continuity and lowers task success rates, indicating the model's sensitivity to temporal processing alterations.  
* **C. Potential limitations inferred:** The fixed 2-second visual memory window inherently prevents the model from succeeding on tasks requiring long-term visual recall or tracking slow environmental changes over extended periods.  
* **D. Important questions not established:** The paper does not establish how to dynamically scale visual acquisition rates or token retention based on the user's query complexity, scene uncertainty, or available compute resources.

**7\. EVIDENCE TABLE**

&nbsp;

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| ELLSA utilizes discrete 1-second time blocks for interleaved multimodal perception and generation. | Implementation detail | Page 3, Section 3.1, Figure 1&nbsp; | Highlights a fixed-rate baseline for full-duplex visual-action systems. |
| The system retains only a limited 2-second window of vision and action history to manage context length. | Implementation detail | Page 15, Appendix A.1&nbsp; | Provides a static heuristic for managing visual memory encoding budgets. |
| Reducing the time block from 1s to 0.48s decreases latency but degrades robotic action task success. | Ablation experiment | Page 23, Appendix G.1, Table 9&nbsp; | Demonstrates the trade-off between rapid interaction latency and visual-action coherence. |
| Performance drops across all metrics during simultaneous speaking-while-acting tasks compared to isolated tasks. | Benchmark result | Page 9, Section 5.2.2, Table 4&nbsp; | Validates that concurrent multimodal generation incurs a computational or representational penalty, necessitating adaptive resource allocation. |

&nbsp;

**8\.  ONE-PARAGRAPH SUMMARY**

&nbsp;

The ELLSA paper introduces a full-duplex, streaming architecture that utilizes a Self-Attention Mixture-of-Experts (SA-MoE) mechanism to simultaneously process vision, speech, text, and action streams. By interleaving inputs and outputs into fixed 1-second time blocks and strictly limiting visual history to the most recent two seconds, the model manages complex multimodal interactions—such as responding to verbal interruptions while executing physical tasks—without requiring separate sequential passes. While the paper does not explore adaptive visual token pruning or variable resolution, it provides a crucial baseline for our research by proving that real-time, concurrent multimodal generation is achievable. However, their ablations reveal that reducing the fixed visual processing block from 1 second to 0.48 seconds severely degrades action coherence, underscoring the urgent need for adaptive, query-dependent visual inference policies that can balance interaction latency with robust environmental grounding.

&nbsp;

# VITA-E

**1\. PAPER IDENTIFICATION**

&nbsp;

* **Title:** VITA-E: Natural Embodied Interaction with Concurrent Seeing, Hearing, Speaking, and Acting.  
* **Authors:** Xiaoyu Liu, Chaoyou Fu, Chi Yan, Chu Wu, Haihan Gao, Yi-Fan Zhang, Shaoqi Dong, Cheng Qian, Bin Luo, Xiuyong Yang, Guanwu Li, Yusheng Cai, Yunhang Shen, Deqiang Jiang, Haoyu Cao, Xing Sun, Caifeng Shan, Ran He.  
* **Publication year and venue:** 2025, arXiv preprint.  
* **Primary research problem:** Current Vision-Language-Action (VLA) models operate on rigid, static interaction paradigms that lack the ability to concurrently see, hear, speak, and act, fundamentally preventing seamless, interruptible human-robot collaboration.

**2\. EXISTING APPROACH**

&nbsp;

The paper proposes a dual-model framework named VITA-E to enable real-time concurrency and interruption. It deploys two parallel VLA instances: an "Active Model" focused on executing the current task, and a "Standby Model" acting as an observer ready to process new user requests. The system relies on a "model-as-controller" paradigm where the Vision-Language Model (VLM) generates special control tokens (e.g., \[ACT\], \[HALT\]) to directly drive system-level state transitions, bridging high-level reasoning with a low-level Action Expert.

&nbsp;

* **Input modalities:** Vision, user speech (natural language instructions), and proprioceptive robot states.  
* **Input arrival format:** Inputs arrive as continuous streaming visual data, real-time auditory commands, and real-time robot state data.  
* **Visual processing type:** Interactive and streaming, specifically designed to process visual and auditory cues continuously to allow for immediate task switching or emergency stops.  
* **Model architecture:** A dual-system setup combining a large Vision-Language Model (System-2, for understanding and control token generation) and a Diffusion Transformer Action Expert (System-1, for generating low-level motor kinematics).

**3\. VISUAL-PROCESSING MECHANISM**

&nbsp;

A. **When is a frame acquired?** *Explicitly stated:* Visual input is acquired continuously at each timestep $t$ from a first-person-view static Realsense D455 RGB camera mounted on the robot's head

&nbsp;B. **When is expensive visual encoding performed?** *Explicitly stated:* Visual encoding is performed by the Vision Encoder at each timestep $t$ to supply the VLM and Action Expert with current environmental context.

C. **Does the method select or discard frames?** *Cannot be established:* The paper does not detail any dynamic frame-dropping or selective sampling policy.

D. **Does it select or discard tokens?** *Cannot be established:* There is no mention of visual token pruning or selection within the vision encoder.

E. **Does it change resolution, crops, or spatial detail?** *Explicitly stated:* The system uses a fixed image resolution of $224\times 224$ for both top and wrist images. It does not dynamically alter spatial detail.&nbsp;

F. **Does it retain or discard historical visual information?** *Explicitly stated:* The model strictly discards historical visual information, utilizing a fixed history length of 1\.

G. **What determines these decisions?** *Explicitly stated:* The visual processing parameters (resolution, history length) are determined by fixed architectural hyperparameters rather than dynamic scene analysis.

H. **Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The visual acquisition and retention policy is entirely fixed. However, the broader *interaction* policy (when to act or halt based on the visual/audio stream) is learned and query-dependent, driven by the VLM's generation of control tokens.

&nbsp;

**4\. COMPUTATIONAL BENEFIT**

&nbsp;

* **Resource reduced:** The VITA-E framework does *not* attempt to reduce computational resources. In fact, the authors explicitly acknowledge that their dual-model architecture incurs a higher computational cost than single-model systems.  
* Instead of reducing compute, the method aims to reduce **interaction latency**, eliminating the rigid sequential delays of standard VLA models.  
* **Quantitative results:** By operating models in parallel, the system achieves an average voice response latency of 2.26 seconds while simultaneously executing physical manipulation tasks.

**5\. TASK PERFORMANCE**

&nbsp;

* **Tasks and datasets evaluated:** The model was evaluated in simulation on the Libero benchmark (Spatial, Object, Goal, LONG suites) and in the real world on pick-and-place tasks (picking up a can, picking and placing a toy). Interactive tasks included speech interruption, task switching, and emergency stops.  
* **Metrics reported:** Success Rate (%) and average voice response latency (seconds). Human-evaluated accuracy (%) was used to grade the VLM's generated responses during ablations.  
* **Impact of altered processing:** The dual-model interaction mechanism successfully achieved a 100% success rate on speech interruptions and emergency stops, and a 93.3% success rate on task switching. However, maintaining a frozen VLM to preserve these interactive capabilities resulted in a noticeable performance gap on complex spatial and goal-oriented manipulation tasks in the Libero benchmark compared to end-to-end jointly trained models like GROOT.

**6\. LIMITATIONS**

&nbsp;

A. **Limitations explicitly acknowledged by the authors:**

&nbsp;

* The dual-model architecture requires higher computational resources compared to single-model systems.  
* The system currently lacks the ability to handle long-horizon, multi-stage tasks.  
* The task-switching mechanism is inefficient, relying on a basic "retraction" to a neutral state rather than calculating a smooth trajectory directly between two tasks.

B. **Limitations supported by experiments or ablations:**

&nbsp;

* VITA-E underperforms on the Libero Goal and Spatial task suites compared to baselines because its VLM is kept frozen, depriving it of the benefits of large-scale joint visual-action pre-training.  
* The 6.7% failure rate in task switching is caused by the VLM misinterpreting new user directives, occasionally providing only a verbal response instead of triggering the necessary physical action switch.

C. **Potential limitations inferred from the methodology:**

&nbsp;

* Because the system utilizes a strict history length of 1, it is inherently incapable of solving tasks that require visual memory, such as tracking objects after they become temporarily occluded.

D. **Important questions the paper does not establish:**

&nbsp;

* The paper does not establish how the system handles latency spikes or thermal throttling when running two heavy VLA models concurrently on real robotic edge hardware.  
* It is unclear how visual noise, motion blur during rapid arm movements, or partial occlusions affect the VLM's ability to trigger accurate emergency stops.

**7\. EVIDENCE TABLE**

&nbsp;

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| VITA-E uses two parallel VLA instances (Active and Standby) to process streams, allowing the robot to handle interruptions without dropping ongoing actions. | System Architecture | Page 7, Section 3.3, Figure 4&nbsp; | Demonstrates how behavioral concurrency and low-latency interaction can be achieved architecturally, albeit at a compute cost. |
| The system operates with a fixed history length of 1 and a fixed image resolution of $224\times 224$. | Implementation Detail | Page 14, Appendix 6.2, Table 5&nbsp; | Highlights a rigidly fixed visual processing pipeline, showing a gap where adaptive visual inference could be applied. |
| Keeping the VLM frozen to preserve interactive instruction-following capabilities degrades complex spatial and goal-oriented manipulation success. | Benchmark Result | Page 8, Section 4.1, Figure 5&nbsp; | Illustrates the trade-off between conversational/interactive competence and fine-grained visual-motor performance. |
| The dual-model architecture incurs higher computational resource consumption than single-model systems. | Explicit Limitation | Page 10, Section 5&nbsp; | Validates that achieving real-time interaction via brute-force parallelization scales poorly, underscoring the need for adaptive efficiency. |

**8\. ONE-PARAGRAPH SUMMARY**

&nbsp;

The VITA-E paper introduces a dual-model framework designed to break the rigid, turn-based paradigms of current Vision-Language-Action (VLA) systems, enabling robots to concurrently see, hear, speak, and act while processing real-time interruptions. By utilizing an "Active" and "Standby" model in parallel, and relying on a VLM that generates system-level control tokens (e.g., \[ACT\], \[HALT\]), the system achieves seamless task switching and emergency halting. For our research in adaptive visual inference, this paper is highly relevant because it perfectly illustrates the tension between interactive responsiveness and computational cost: VITA-E achieves fluid interaction through brute-force parallelization rather than adaptive visual processing. Its reliance on a strictly fixed visual resolution and a history length of 1 highlights a significant opportunity for future work to introduce dynamic, query-dependent visual sampling to reduce the heavy computational burden while maintaining this level of concurrency.

&nbsp;

# PruneVid

1. PAPER IDENTIFICATION  
* **Title:** PRUNEVID: VISUAL TOKEN PRUNING FOR EFFICIENT VIDEO LARGE LANGUAGE MODELS  
* **Authors:** Xiaohu Huang, Hao Zhou, Kai Han  
* **Publication year and venue:** 2024, arXiv preprint (cs.CV)  
* **Primary research problem:** Large Language Models (LLMs) experience severe computational bottlenecks and high KV cache memory costs when processing video data due to the inherent redundancy of spatial-temporal tokens and the quadratic complexity of attention mechanisms.  
2. EXISTING APPROACH

PruneVid is a training-free token pruning mechanism designed to reduce visual redundancy in pre-trained Video LLMs. It operates in two stages: first, it identifies and merges static tokens across temporal frames and clusters spatially similar tokens using a density peaks clustering algorithm to eliminate inherent video redundancy. Second, it passes the merged tokens through the initial layers of an LLM and uses the cross-attention scores between the user's text question and the visual tokens at an intermediate layer to select only the most query-relevant visual tokens. The unselected tokens are pruned from the sequence and the KV cache for the remainder of the decoding process.

&nbsp;

* **Input modalities:** Video (frames) and Text (user query).  
* **Input arrival format:** Inputs arrive as complete offline video files and static text queries.  
* **Visual processing type:** Offline pre-filling stage.  
* **Model architecture:** Off-the-shelf Video LLMs consisting of a Vision Encoder and a standard Transformer-based LLM backbone (evaluated on PLLaVA, ST-LLM, and LLaVA-OneVision).  
3. VISUAL-PROCESSING MECHANISM

A. **When is a frame acquired?** *Explicitly stated:* The entire sequence of video frames ($T$ frames) is acquired as a complete batch before any processing begins. *Cannot be established:* How frames would be acquired in a real-time continuous video stream.

B. **When is expensive visual encoding performed?** *Explicitly stated:* Visual encoding is performed on all $T$ frames by the vision encoder at the very beginning of the pipeline, prior to any token merging or pruning.

C. **Does the method select or discard frames?** *Explicitly stated:* It does not discard entire frames. Instead, it segments the video into temporal chunks and merges static tokens within those segments.

D. **Does it select or discard tokens?** *Explicitly stated:* Yes, it spatially and temporally merges tokens, and then explicitly discards (prunes) a fixed percentage of them at an intermediate LLM layer.

E. **Does it change resolution, crops, or spatial detail?** *Explicitly stated:* It reduces spatial detail by clustering spatially similar tokens into a single averaged token representation. *Cannot be established:* Dynamic cropping or changes to the raw input resolution of the frames.

F. **Does it retain or discard historical visual information?** *Explicitly stated:* It discards visual information from the KV cache for the decoding stage if those tokens are deemed irrelevant to the query. *Cannot be established:* How the KV cache is managed across multi-turn, continuous dialogues.&nbsp;

G. **What determines these decisions?** *Explicitly stated:* Spatial and temporal merging are determined by feature cosine similarity against a threshold ($\tau$). Token pruning is determined by ranking the cross-attention scores between question tokens and visual tokens at an intermediate LLM layer ($M$).

H. **Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The pruning policy is *query-dependent* (using question-to-vision attention), *heuristic* (relying on fixed thresholds like $\tau$, merging ratios, and top\-$\alpha$% selection), and based on *scene changes* (using DPC-KNN clustering to group temporal scenes).

&nbsp;

4. COMPUTATIONAL BENEFIT  
* **Targeted resources:** Visual-token count, Total model FLOPs, GPU memory, and Time-To-First-Token (TTFT) latency.  
* **Quantitative results:** PruneVid retains only 15.1% to 17.0% of the original visual tokens (pruning over 80%). This token reduction decreases total LLM FLOPs to 0.20x \- 0.26x (a 74% \- 80% reduction), reduces GPU memory consumption from 20G to 17G on PLLaVA, and yields a TTFT latency speed-up of 1.55x.  
5. TASK PERFORMANCE  
* **Tasks and datasets:** Evaluated on Generic Multi-Choice VideoQA (MVBench), Long-form Multi-Choice VideoQA (Video-MME, EgoSchema), and Text Generation (VideoChatGPT-Bench).  
* **Metrics reported:** Accuracy (%) for multi-choice benchmarks, and GPT-3.5-Turbo scores (ranging 1-5) for Correctness of Information, Detail Orientation, Contextual Understanding, Temporal Understanding, and Consistency.  
* **Impact of altered processing:** Despite discarding over 80% of tokens, accuracy remains competitive and sometimes improves compared to the unpruned baseline (e.g., PLLaVA MVBench accuracy increases from 46.6% to 47.6%). This indicates that removing redundant visual tokens acts as a denoiser, improving the LLM's focus on relevant information.  
6. LIMITATIONS

A. **Limitations explicitly acknowledged by the authors:** Retaining too many tokens (setting selection ratio $\alpha$ too high) introduces irrelevant visual information that degrades accuracy. Furthermore, pruning tokens at too shallow a layer (e.g., layer 5\) degrades performance because the LLM has not yet formed semantic cross-attention patterns. B. **Limitations supported by experiments or ablations:** If the spatial merging ratio ($\beta$) is set to 1.0 (no spatial merging) or set too aggressively low (e.g., 0.125), accuracy drops noticeably across benchmarks. C. **Potential limitations inferred from the methodology:** The system must process all frames through the heavy vision encoder and the first $M$ layers of the LLM before query-guided pruning occurs. This approach limits its utility for real-time streaming interaction, where encoding uninformative frames is a major latency bottleneck. D. **Important questions the paper does not establish:** The paper does not establish how this architecture functions without knowing the full video sequence upfront, nor how the attention-guided pruning handles sequential query shifts in a continuous, open-ended real-time interaction.

&nbsp;

7. EVIDENCE TABLE

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| PruneVid prunes over 80% of visual tokens while maintaining or improving VideoQA accuracy across three architectures. | Benchmark result | Page 8, Table 1&nbsp; | Proves that vast amounts of visual data can be discarded post-encoding without harming downstream visual-task performance. |
| Query-guided pruning reduces FLOPs by \~77%, reduces GPU memory by 3GB, and speeds up Time-To-First-Token by 1.55x. | Efficiency analysis | Page 9, Table 2&nbsp; | Quantifies the computational trade-off and latency benefits of adaptive token pruning. |
| The framework uses a two-stage process: temporal-spatial feature clustering followed by LLM cross-attention ranking. | System Architecture | Page 5, Figure 2&nbsp; | Provides a specific hybrid mechanism (feature-based \+ semantic-based) for reducing visual load. |
| Pruning at layer 10 provides the optimal balance; pruning too early (layer 5\) degrades performance. | Ablation study | Page 10, Figure 3a&nbsp; | Highlights that query-dependent visual decisions require sufficient semantic processing depth to be accurate. |

8. ONE-PARAGRAPH SUMMARY

The PruneVid paper is highly relevant to research on adaptive visual inference because it demonstrates that query-guided token selection drastically reduces multimodal computational overhead while preserving, and occasionally improving, task performance. By combining heuristic spatial-temporal token clustering with semantic pruning based on cross-attention scores in the LLM, the method safely discards over 80% of visual tokens, achieving significant reductions in FLOPs and KV-cache memory footprints. For real-time interaction systems, this proves that dense visual representations contain vast redundancy that can be aggressively discarded once the user's specific intent is known. However, PruneVid operates entirely offline, encoding the full video batch before applying its query-dependent pruning. Adapting this logic for real-time multimodal interaction will require shifting semantic pruning earlier in the pipeline to avoid the bottleneck of heavily encoding uninformative frames in a live continuous stream.

&nbsp;

# VideoChat-Online

**1\. PAPER IDENTIFICATION**

&nbsp;

* **Title:** Online Video Understanding: OVBench and VideoChat-Online  
* **Authors:** Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, Limin Wang  
* **Publication year and venue:** 2025, arXiv (cs.CV)  
* **Primary research problem:** Multimodal Large Language Models (MLLMs) struggle to apply offline video understanding capabilities to real-time, continuous online video streams (e.g., autonomous driving, robotics) due to the infinite influx of visual information and the need to process dynamic, time-dependent contexts efficiently.

**2\. EXISTING APPROACH**

&nbsp;

The paper introduces VideoChat-Online, which utilizes a Pyramid Memory Bank (PMB) to process streaming video. The PMB progressively abstracts older frames from high spatial detail to high temporal detail across hierarchical memory queues. To prevent memory overload, the system continuously evicts redundant frames based on visual similarity, down-samples their spatial resolution, and synchronizes these removals with the LLM's KV Cache.

&nbsp;

* **Input modalities:** Video (streaming frames) and Text (user queries).  
* **Input arrival format:** Inputs arrive continuously as streaming video sequences and interactive dialogue queries.  
* **Visual processing type:** Streaming and interactive.  
* **Model architecture:** An offline-to-online trained 4B-parameter model utilizing an InternViT-300M visual encoder, a Phi-3 language model, and a custom hierarchical Pyramid Memory Bank (PMB).

**3\. VISUAL-PROCESSING MECHANISM**

&nbsp;

A. **When is a frame acquired?** *Explicitly stated:* Frames are continuously sampled from the incoming video stream at varying rates (e.g., 1 fps, 2 fps, 8 fps) corresponding to different hierarchical memory queues.

&nbsp;

B. **When is expensive visual encoding performed?** *Explicitly stated:* Encoding is performed continuously as frames stream in, before they are stored in the memory module and synchronized with the KV Cache (the "Streaming Frame Writing" phase).

&nbsp;

C. **Does the method select or discard frames?** *Explicitly stated:* Yes. When a memory layer reaches capacity, the system evicts the older frame from the most visually similar adjacent frame pair.

&nbsp;

D. **Does it select or discard tokens?** *Explicitly stated:* Yes. The method erases KV Cache tokens corresponding to the timestamps of evicted frames to maintain synchronization between the memory bank and the LLM.

&nbsp;

E. **Does it change resolution, crops, or spatial detail?** *Explicitly stated:* Yes. Evicted frames are spatially down-scaled (via average pooling) to lower resolutions before being passed to the next (older) memory layer, drastically reducing the tokens per frame (e.g., from 256 down to 64, then to 16).

&nbsp;

F. **Does it retain or discard historical visual information?** *Explicitly stated:* It retains historical information but structurally discards its fine-grained spatial detail. Recent frames keep high spatial resolution, while older frames are kept at high temporal density but low spatial resolution.

&nbsp;

G. **What determines these decisions?** *Explicitly stated:* Frame eviction decisions are determined by memory capacity limits (${C}_{i}$) and the cosine similarity between average-pooled adjacent frames.

&nbsp;

H. **Is the policy fixed, heuristic, query-dependent, learned, uncertainty-dependent, or based on scene changes?** *Explicitly stated:* The policy is based on *scene changes* (cosine similarity between adjacent frames) and *heuristics* (fixed memory capacities and fixed resolution down-scaling factors). *Cannot be established:* The visual processing policy is entirely query-agnostic; it pre-fills and compresses memory independently of the user's specific dialogue context.

&nbsp;

**4\. COMPUTATIONAL BENEFIT**

&nbsp;

The method attempts to reduce **KV Cache memory**, **total visual-token count**, and **VRAM usage** to enable real-time processing of infinite video streams.

&nbsp;

* **Total visual tokens:** The inference overhead is strictly bounded to 832 visual tokens total across all memory layers.  
* **VRAM usage:** VideoChat-Online limits VRAM usage to 8.71 GB when processing 1000 video frames, compared to MovieChat (16.90 GB) and Flash-VStream (16.03 GB).

**5\. TASK PERFORMANCE**

&nbsp;

* **Tasks and datasets evaluated:** The proposed OVBench (16 online subtasks including Spatial Perception, Temporal Hallucination, Past Memory, Future Prediction) and multiple offline benchmarks (EgoSchema, MLVU, VideoMME, MVBench, LongVideoBench).  
* **Metrics reported:** Accuracy (%) and VRAM Usage (GB).  
* **Impact of altered processing:** The similarity-based frame eviction and hierarchical resolution scaling allowed the 4B model to achieve 54.9% accuracy on OVBench, outperforming the 7B Flash-VStream online model (31.2%). Ablations reveal that removing the highest-resolution spatial memory layer significantly degrades Spatial Perception, while removing the deepest temporal memory degrades Temporal Perception. Reducing the streaming frame capture rate from 4 fps to uniform sampling drops accuracy by approximately 2%.

**6\. LIMITATIONS**

&nbsp;

A. **Limitations explicitly acknowledged by the authors:** Existing offline MLLMs generalized to online tasks via sliding windows currently outperform native online architectures, highlighting the difficulty of training online models effectively without robust offline-to-online knowledge transfer. B. **Limitations supported by experiments or ablations:** Replacing the similarity-based eviction policy with simple First-In-First-Out (FIFO) queuing degrades OVBench performance from 54.9% to 51.5%, demonstrating the model's heavy reliance on inter-frame similarity dynamics. C. **Potential limitations inferred from the methodology:** Because frame eviction relies strictly on global cosine similarity, the system risks permanently deleting high-resolution details of small but critical objects if they undergo subtle changes between largely similar adjacent frames. Furthermore, because compression happens before the query is known, the model cannot adaptively "zoom in" or retain spatial detail based on user uncertainty. D. **Important questions the paper does not establish:** The paper does not establish how the memory synchronization latency scales during extremely rapid scene changes where numerous frames might trigger simultaneous eviction and KV cache deletion.

&nbsp;

**7\. EVIDENCE TABLE**

&nbsp;

| Finding | Evidence type | Page/section/figure/table | Relevance to our research |
| :---- | :---- | :---- | :---- |
| PMB progressively scales down spatial resolution ($Re{s}_{i+1}$) for older frames to prioritize temporal capacity. | System Architecture | Page 5 / Section 4.1 / Figure 3&nbsp; | Demonstrates a heuristic strategy for determining *how much* visual information to retain over time. |
| Frame eviction is triggered by memory capacity and executed by finding the highest cosine similarity between adjacent frames. | Implementation Detail | Page 5 / Section 4.1&nbsp; | Provides a scene-change-based mechanism for deciding *when* visual information should be processed/discarded. |
| The system strictly caps the total inference computational overhead at 832 visual tokens across all memory layers. | Hyperparameter Setting | Page 12 / Appendix 1.1&nbsp; | Quantifies the computational constraints required for real-time interactive streaming. |
| VRAM usage drops to 8.71 GB, outperforming state-of-the-art online models by nearly 50% while maintaining higher accuracy. | Benchmark Result | Page 12 / Appendix Table 7&nbsp; | Illustrates the trade-off between architectural compression mechanisms and hardware cost. |
| Decreasing the frame sampling rate (fps) steadily degrades overall accuracy on online tasks. | Ablation Study | Page 13 / Appendix Figure 7&nbsp; | Highlights the performance impact of initial visual acquisition rates. |

**8\.  ONE-PARAGRAPH SUMMARY**

&nbsp;

This paper introduces VideoChat-Online, highlighting structural methods for processing infinite video streams through a Pyramid Memory Bank (PMB). The system demonstrates that bounding computational cost in real-time interactions requires aggressive, asymmetric visual retention: it preserves high spatial resolution for recent frames while down-scaling older frames to prioritize dense temporal continuity. Visual processing decisions are driven by scene redundancy, utilizing cosine similarity to evict adjacent frames and directly purging corresponding tokens from the LLM's KV cache to maintain strict hardware limits (8.71 GB VRAM). While highly efficient, the visual inference pipeline remains entirely heuristic and query-agnostic, operating continuously in the background rather than adapting to the user's specific dialogue context. This highlights a critical gap and opportunity for future research to design adaptive pipelines where visual retention policies shift dynamically in response to user intent and uncertainty.

&nbsp;

&nbsp;

Conclusions:

**Phostream**

Your assessment of PhoStream is entirely accurate. It is an evaluation framework, not a novel architecture. It will be highly valuable later for benchmarking your implementation, particularly because it enforces strict response windows for "instant" versus "future" questions, quantifying exactly why real-time adaptive processing is necessary.

&nbsp;

**FastV**

There is a slight misconception here regarding what is being pruned. FastV does not conclude that semantic information only exists in the initial *frames* of a video. Instead, it identifies an "inefficient visual attention" phenomenon within the models themselves. After the first few decoder layers, the LLM stops paying attention to the dense visual tokens and hyper-focuses on structural "anchor tokens" (like the initial system prompt tokens). FastV simply **prunes the deep visual tokens that the LLM is already ignoring**. It demonstrates that models naturally over-encode visual data, and aggressively dropping that data mid-computation saves significant FLOPs without hurting performance.

&nbsp;

**IVTP (Instruction-guided Visual Token Pruning)**

While it operates on static images, discounting it misses the broader architectural mechanism. IVTP's value lies in **using the user's explicit instruction to dictate *which* visual tokens are retained**. In a real-time streaming pipeline, applying this logic means you can dynamically prune visual tokens that do not match the current dialogue context before processing subsequent frames, directly addressing the question of how to adapt visual retention to the user's query.

&nbsp;

**ELLSA (End-to-end Listen, Look, Speak and Act)**

You correctly identified the limitations of its brute-force approach (1 fps, 2-second memory). However, ELLSA serves as a crucial negative baseline. It proves that attempting continuous multimodal interaction *without* adaptive compression forces the system into an aggressively tiny context window to avoid latency bottlenecks, particularly when deploying on constrained edge hardware. It validates the necessity of your research.

&nbsp;

**VITA-E**

This paper is highly relevant to embodied and autonomous systems, operating much like a ROS 2 service call rather than a continuous topic subscription. VITA-E does not passively ingest a continuous video stream. It utilizes a **visual trigger mechanism where the language model actively decides *when* to poll the vision module** based on the conversation context. This "pull" mechanism—rather than a camera "pushing" frames continuously—directly answers your question of how visual processing decisions should respond to the dialogue context.

&nbsp;

**PruneVid**

The merging mechanism here is not based on the age of the frames. PruneVid utilizes **spatial-temporal clustering to group visually similar tokens**, and then uses the user's text question to score and prune irrelevant visual tokens inside the LLM. Its major limitation is that it operates offline, requiring the entire video batch before pruning. However, it successfully demonstrates how to combine heuristic pixel-similarity pruning with query-dependent semantic pruning.

&nbsp;

**VideoChat-Online**

While it shares the goal of reducing visual load with PruneVid, its approach is fundamentally the opposite. PruneVid is offline and query-dependent; VideoChat-Online is **streaming and query-agnostic**. It processes frames live and uses a Pyramid Memory Bank to compress them purely based on age and visual similarity, progressively degrading the spatial resolution of older frames to prioritize dense temporal continuity. It ignores the user's query during compression, focusing solely on bounding the hardware memory footprint for infinite streams.

&nbsp;

&nbsp;