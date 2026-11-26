# Research Proposal: Adaptive Visual Inference in Real-Time Multimodal Interaction

## 1\. Candidate 1: Early-Layer Variance Gating (H18)

**A. Refined Hypothesis**

* **Independent variable:** The presence and sensitivity threshold of the early-layer temporal variance gate.  
* **Dependent variables:** Deep visual-encoder invocations, total ViT processing time per second of video, and transient-event recall (task accuracy on rapidly changing visual details).  
* **Controlled variables:** Camera acquisition frame rate (fixed at 5 Hz), baseline Vision-Language Model architecture, and video resolution.  
* **Mechanism:** Semantic stillness in a scene correlates with high spatial-feature stability in the early layers of a Vision Transformer (ViT). Bypassing deep-layer attention when early layers remain static prevents redundant computation without missing semantic updates.  
* **Conditions for success:** The prediction should hold during standard conversational or egocentric interactions where the background is largely static and scene changes are distinct.  
* **Conditions for failure:** The prediction will fail if critical semantic updates involve very small pixel-area changes (e.g., reading updating text on a monitor) that do not cross the early-layer variance threshold.

**B. Proposed Method** Acquire frames continuously at a high rate (e.g., 5 Hz). Pass the frame only through the first $K$ blocks (e.g., Blocks 1-3) of a frozen ViT. Compute the cosine similarity of the block 3 output embeddings against the block 3 embeddings of the last fully encoded frame. If the similarity exceeds threshold $\tau$, halt the forward pass, discard the frame, and reuse the KV cache representations of the previous frame. If similarity is below $\tau$, process the frame through the remaining blocks and update the LLM's visual context.

**C. Experimental Setup**

* **Dataset:** PhoStream's benchmark subset (real-world mobile streaming scenarios).  
* **Input format:** Continuous video streaming updated at 5 Hz.  
* **Annotation requirements:** None. Relies on standard VQA ground truths.  
* **Model components:** An open-weight multimodal model with a standard ViT encoder (e.g., Qwen-VL or LLaVA).  
* **Implementation work:** Write a custom C++ inference wrapper (e.g., via LibTorch) to intercept the ViT forward pass at block 3, calculate cosine similarity, and conditionally route the control flow. This approach leverages core principles of embedded system design to tightly manage hardware-level memory bandwidth during execution.  
* **Important confounders:** The computational overhead of extracting block 3 tensors and computing their similarity might exceed the FLOPs saved by skipping blocks 4-24.

**D. Baselines**

1. **Continuous fixed-rate encoding (5 Hz):** The unmodified model. Required to establish the upper bound for task accuracy/event recall and the baseline for processing time.  
2. **Lower fixed-rate encoding (e.g., 1 Hz or 0.5 Hz):** A uniform subsampling baseline tuned to match the average encoder invocations of the gated model. This isolates whether the gating mechanism's selection of frames outperforms blindly reducing the frame rate.

**E. Evaluation Metrics**

* **Primary (Hypothesis Testing):** Number of frames passed to the expensive encoder (measures selection efficiency), Task accuracy (measures grounding preservation), Vision-encoder processing time (measures latency reduction).  
* **Secondary (Diagnostics):** Transient-event recall (measures if the gate missed quick changes) and Peak memory usage.

**F. Expected Observations (If Correct)** Vision-encoder processing time will drop by \>60% during visually static conversational segments. Task accuracy will match the 5 Hz continuous baseline and strictly outperform the matched lower fixed-rate baseline.

**G. Observations That Would Falsify It**

1. No meaningful reduction in end-to-end response latency because the overhead of calculating the cosine similarity gate is too high.  
2. An unacceptable increase in missed-event rate, proving that early ViT layers do not reliably capture fine-grained semantic updates.

**H. Feasibility Risks** High memory bandwidth requirements to pull block 3 embeddings into GPU memory to calculate similarity could cause a bottleneck, particularly in real-time edge deployments.

**I. Remaining Novelty Questions** While it is established that early layers determine spatial relevance for static images, it remains unverified whether this translates to reliable temporal gating across a continuous streaming context.

**J. Concise Research-Problem Statement** Investigating whether early-layer feature variance in Vision Transformers can serve as a temporal gate to bypass deep visual encoding, reducing computation in continuous multimodal streams without degrading transient-event recall.

---

## 2\. Candidate 2: Rigid Acquisition with Active Clarification (H16)

**A. Refined Hypothesis**

* **Independent variable:** The interactive policy (Passive silence-filling vs. Active `[CLARIFY]` emission).  
* **Dependent variables:** Task accuracy on out-of-frame/ambiguous queries, and Early Response rate.  
* **Controlled variables:** The visual acquisition rate, rigidly fixed to 1 fps.  
* **Mechanism:** Asking the user to provide better visual evidence resolves ambiguity with higher accuracy than attempting to hardware-adapt the camera rate or relying on internal hallucination.  
* **Conditions for success:** The user is present, willing to interact, and the missing visual evidence is physically acquirable.  
* **Conditions for failure:** Rapid tasks where the time taken to verbally negotiate evidence exceeds the safe reaction window (e.g., autonomous driving, fast robotic manipulation).

**B. Proposed Method** Lock the visual acquisition hardware to a rigid, predictable 1 fps stream to eliminate controller latency. Train the LLM to map high visual uncertainty (e.g., target object occluded or out-of-frame) to a specific `[CLARIFY]` control token. Emitting this token interrupts the model's reasoning and triggers text-to-speech to actively ask the user a clarifying question (e.g., "Can you point the camera left?").

**C. Experimental Setup**

* **Dataset:** A modified subset of PhoStream's Forward tasks combined with standard VQA, augmented so that the target object is deliberately cropped out or heavily occluded in the initial frames.  
* **Annotation requirements:** Training data must be constructed where ambiguous inputs are labeled with a `[CLARIFY]` response rather than an answer.  
* **Model components:** A full-duplex capable VLM (e.g., Qwen-Omni architecture).  
* **Evaluation protocol:** Requires a simulated "User-in-the-loop" script. If the model emits `[CLARIFY]`, the script pauses the timer, updates the visual stream to reveal the occluded object, and resumes.  
* **Implementation work:** Supervised fine-tuning (SFT) to teach the `[CLARIFY]` behavior and the development of the simulated-user evaluation harness in C++.  
* **Important confounders:** The model may learn to optimize for safety by spamming the `[CLARIFY]` token on all queries, requiring a strict penalty function during evaluation.

**D. Baselines**

1. **Passive fixed-rate baseline (1 fps):** The standard model that attempts to answer based on available evidence. Required to measure the baseline Early Response and hallucination rates.  
2. **Passive Wait baseline:** A model trained to emit a `[WAIT]` token and hold silence until the object appears, rather than actively asking. Required to test if active dialogue outperforms passive waiting.

**E. Evaluation Metrics**

* **Primary (Hypothesis Testing):** Task accuracy on ambiguous tasks, Early Response rate (measures hallucination reduction), Clarification rate (measures token spamming).  
* **Secondary (Diagnostics):** End-to-end response latency (including the time taken to verbally negotiate).

**F. Expected Observations (If Correct)** The model will demonstrate high success on out-of-frame tasks where the passive baseline fails. The Early Response rate will drop to near zero on Forward predictive tasks.

**G. Observations That Would Falsify It**

1. The model optimizes for safety by spamming the `[CLARIFY]` token on easily answerable queries, demonstrating it cannot accurately calibrate its own visual confidence threshold.  
2. The interaction delay introduced by conversational backchanneling renders the system subjectively unusable compared to passive waiting.

**H. Feasibility Risks** Designing a reward/penalty mechanism during SFT that accurately discourages `[CLARIFY]` spam without entirely suppressing the behavior is highly complex.

**I. Remaining Novelty Questions** While wait-controllers exist for audio streams, actively leveraging full-duplex capability to resolve visual ambiguity via explicit user negotiation remains uninvestigated in the documented literature.

**J. Concise Research-Problem Statement** Investigating whether coupling a rigid visual acquisition rate with an active full-duplex clarification policy resolves ambiguous visual queries more accurately than passive evidence-gathering, without inducing unacceptable interaction delays or over-clarification loops.