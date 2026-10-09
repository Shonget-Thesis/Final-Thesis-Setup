**A Tanh-Based Teacher-Student Velocity Blending Approach for Flow-Matching Text-to-Speech Inference**

Chiong, H., Limpahan, M. V., Locsin, R., Sajol, R. J., Eder, M. S.



1. **INTRODUCTION**



**1.1 BACKGROUND OF THE STUDY**

In recent years, deep generative modeling for speech synthesis has shifted from classical parametric and autoregressive frameworks toward non-autoregressive Continuous Normalizing Flows (CNFs) trained via Flow Matching (Lipman et al., 2023; Mehta et al., 2024). To streamline generative trajectories, Optimal-Transport Conditional Flow Matching (OT-CFM) constructs straight-line probability paths between Gaussian noise priors and target acoustic mel-spectrogram distributions, minimizing vector field curvature and enabling accurate acoustic reconstruction (Eskimez et al., 2024; Lipman et al., 2023; Mehta et al., 2024). This mathematical foundation underpins modern zero-shot Text-to-Speech (TTS) architectures, including Glow-WaveGAN 2, YourTTS, NaturalSpeech 2, NaturalSpeech 3, Seed-TTS, and CosyVoice 2, which clone and synthesize the voice of an unseen target speaker using only a brief reference audio prompt without requiring speaker-dependent fine-tuning or model retraining (Anastassiou et al., 2024; Casanova et al., 2022; Du et al., 2024; Ju et al., 2024; Lei et al., 2022; Shen et al., 2023).



This zero-shot generative capability holds immense value for assistive Information and Communication Technology (ICT), particularly in special education tools for neurodivergent learners with Autism Spectrum Disorder (ASD) (Sahito et al., 2024). Autistic children often present social-communicative difficulties and auditory sensory sensitivities, responding significantly better to familiar, comforting voices, such as parents or special education teachers than to generic, robotic synthetic voices (Cassidy et al., 2016; Sahito et al., 2024). Empirical evaluations of interactive mobile applications, such as the *FillMeApp&#x20;*&#x65;ducational learning game, demonstrate that gamified visual interfaces significantly enhance student motivation and extend focus spans up to 8 to 10 minutes per session compared to traditional visual aids (Eder et al., 2016). Deploying zero-shot voice cloning within mobile assistive software allows platforms to dynamically synthesize personalized instructional feedback in a familiar caretaker's voice, fostering predictable, low-anxiety learning environments (Cassidy et al., 2016; Eder et al., 2016; Sahito et al., 2024).



Despite the high acoustic naturalness of continuous flow-matching architectures, their practical deployment on resource-constrained mobile edge hardware is severely hindered by high inference latency (Du et al., 2024; Mehta et al., 2024; Ye et al., 2024). Generating continuous audio representations requires numerically solving Ordinary Differential Equations (ODEs) across dozens of sequential discretization steps, where each step necessitates evaluating a full neural network forward pass (Eskimez et al., 2024; Mehta et al., 2024). This high count of Neural Function Evaluations (NFEs) creates an intractable computational bottleneck for interactive assistive applications, where cloud API network jitter or processing lag breaks the child’s fragile cognitive attention loop (Eder et al., 2016; Sahito et al., 2024; Sofronievski et al., 2022; Wu et al., 2025). Consequently, hybrid teacher-guided sampling frameworks have been established, where a multi-step teacher flow model executes initial ODE steps to anchor global prosodic structure, after which a distilled student model completes terminal acoustic refinement in very few steps (Li et al., 2026; Ye et al., 2024; Zhang et al., 2026).



To eliminate latency penalties without incurring the massive computational cost of retraining multi-million parameter networks, research has increasingly prioritized training-free inference mechanisms (Monsefi et al., 2026; Zhang et al., 2026). Training-free inference refers to test-time algorithmic strategies that manipulate or regularize sampling trajectories, velocity fields, or latent states strictly during execution, bypassing the immense computational footprint, energy demands, and data requirements of fine-tuning or re-distilling model weights (Monsefi et al., 2026; Yao et al., 2025). In multi-stage teacher-student sampling, applying training-free trajectory controls offers a computationally lightweight paradigm to optimize inference efficiency and acoustic stability directly at the decoding stage (Li et al., 2026; Monsefi et al., 2026; Zhang et al., 2026).



While teacher-guided sampling reduces total NFEs, existing implementations exhibit a critical structural vulnerability: latent trajectory instability caused by boundary discontinuity at the teacher-student handover (Li et al., 2026; Monsefi et al., 2026; Zhang et al., 2026). In standard teacher-guided execution, the transition between models occurs as an unbuffered, abrupt switch where the multi-step teacher halts ODE integration, extrapolates an unregularized state estimate, and passes it directly into the student model’s discrete reconstruction chain (Li et al., 2026; Monsefi et al., 2026). Because the distilled student is trained to operate across specific learned noise distributions, forcing it to evaluate an intermediate state inherited from an external continuous trajectory induces severe covariate shift and vector field mismatch (Monsefi et al., 2026; Zhang et al., 2026). This vector field disagreement disrupts the probability flow path, creating latent trajectory instability that degrades acoustic naturalness. Thus, the research gap lies in the absence of a training-free inference mechanism capable of ensuring transition continuity and trajectory stability across multi-model generative handovers. 



To resolve this inference-time boundary mismatch without model retraining, this study investigates training-free, trajectory-aware transition control within distilled zero-shot flow-matching speech synthesis while keeping all pretrained neural network weights strictly frozen. By establishing explicit diagnostic probes for latent field disagreement using directional cost and norm distance metrics, this research evaluates trajectory-aware transition control mechanisms, incorporating continuous velocity blending schedules and cushioning manifolds, to absorb boundary shock, eliminate interface turbulence, and preserve latent trajectory continuity across the handover horizon. Through systematic ablation studies and multi-stage decision gates, the investigation benchmarks trajectory-aligned transition procedures against student-only, unbuffered teacher-guided, and teacher-only baselines across primary evaluation metrics, including Word Error Rate (WER), speaker similarity cosine distance, UTMOS naturalness, and Real-Time Factor (RTF) (Li et al., 2026; Mehta et al., 2024; Ye et al., 2024). Ultimately, this research seeks to optimize the quality–efficiency Pareto frontier of distilled zero-shot flow matching, demonstrating a fast, low-latency, and trajectory-stable voice cloning pipeline tailored for on-device deployment in neurodivergent assistive learning platforms



**1.2 PROBLEM STATEMENT**

Accelerating zero-shot Text-to-Speech (TTS) synthesis through multi-stage teacher-student distillation reduces computational latency by compressing multi-step vector integration into very few Neural Function Evaluations (NFEs). However, aggressive step reduction can lead to reduced speech diversity and prosodic variation (Li et al., 2026). Teacher-guided sampling addresses this limitation by using a continuous flow-matching teacher model to establish global prosodic structure before transferring execution to a distilled student generator (Li et al., 2026). Nevertheless, transferring intermediate generation states between distinct generative processes raises concerns regarding state alignment and transition stability. 



**1.2.1 The Handover Boundary in Teacher-Guided Sampling**

Knowledge distillation enables fast acoustic generation, but transitioning between a continuous flow-matching teacher model and a distilled student generator requires careful regulation of the switching boundary (Sevriugov et al., 2026; Zhang & Zhao, 2026). Studies on hybrid generative processes indicate that direct, unbuffered transfers of continuous-flow outputs into iterative sampling processes may introduce boundary misalignment and increase reconstruction error (Ektefaie et al., 2026). In teacher-guided sampling architectures such as DMOSpeech 2, the teacher model performs the initial denoising stages before transferring the generation process to the student model at a predefined switching point (Li et al., 2026). However, DMOSpeech 2 does not introduce a dedicated, training-free mechanism for regulating this transition or evaluating the compatibility of the transferred state with the student’s learned dynamics.



**1.2.2 Potential Latent-State Discontinuity and Trajectory Effects**

The teacher-student handover also raises concerns regarding latent-state continuity across transition boundaries. Because the teacher and student models may follow different generative dynamics, passing an unregularized teacher state directly into the student’s reconstruction chain may introduce latent-state discrepancies (Monsefi et al., 2026; Zhang et al., 2026). Vector-field differences at the transition boundary could induce trajectory drift and compound estimation errors during subsequent sampling steps (Monsefi et al., 2026; Zhang et al., 2026). Furthermore, abrupt perturbations at transition points may disrupt residual trajectory calibration, potentially amplifying discretization errors and affecting overall acoustic naturalness (Trikutam, 2026; Yao et al., 2026). These potential trajectory disruptions warrant investigation into lightweight, training-free trajectory-control mechanisms.



**1.2.3 Application Constraints in Speech Synthesis for Neurodivergent Learners**

Beyond algorithmic trajectory alignment, deploying zero-shot speech synthesis within assistive Information and Communication Technology (ICT) highlights operational challenges for learners with Autism Spectrum Disorder (ASD) (Sahito et al., 2024). Prior studies demonstrate the potential value of interactive digital tools in special education. For example, gamified learning applications such as the *FillMeApp&#x20;*&#x74;ool evaluated by Eder et al. (2016) have been associated with increased motivation and sustained attention among participating students with autism. Additionally, Sahito et al. (2024) reported positive effects of assistive ICT interventions on academic and social outcomes among students with ASD.



However, practical deployment of speech-based assistive applications presents challenges related to personalized voice generation and inference latency. Personalized or familiar voices may be relevant in assistive speech applications, creating a potential use case for zero-shot voice cloning, which can generate speech using a target speaker’s vocal characteristics without requiring speaker-specific retraining (Cassidy et al., 2016). At the same time, interactive educational applications rely on responsive auditory feedback, while cloud-based speech generation may introduce processing delays and network-dependent latency that can affect responsiveness (Du et al., 2024; Wu et al., 2026). These considerations highlight the need for speech synthesis systems that can provide personalized speech while maintaining efficient inference.



Consequently, an application gap exists for a low-latency, trajectory-stable zero-shot TTS engine capable of efficient execution on local mobile hardware to support interactive assistive learning tools for neurodivergent learners.



**1.3 OBJECTIVES OF THE STUDY**

**1.3.1 General Objective**

The general objective of this study is to evaluate a training-free transition-control mechanism within the teacher-student handover of distilled zero-shot Text-to-Speech (TTS) to reduce potential latent-state misalignment and improve generation stability while maintaining inference efficiency under frozen-model constraints.



**1.3.2 Specific Objectives**

Specifically, this study aims to:

**1. Characterize the teacher-student handover boundary.&#x20;**&#x45;valuate the effects of varying transition points on vector-field alignment, acoustic quality, and inference latency in distilled zero-shot TTS.

**2. Evaluate a training-free transition-control mechanism.&#x20;**&#x46;ormulate and test velocity blending schedules or cushioning manifolds to reduce vector-field mismatch and potential trajectory errors between teacher and student models.

**3. Benchmark quality-efficiency trade-offs:&#x20;**&#x43;ompare the trajectory-aligned pipeline with student-only, unbuffered teacher-guided, and teacher-only baselines using Word Error Rate (WER), speaker similarity cosine distance, UTMOS acoustic naturalness, and Real-Time Factor (RTF) under frozen-model constraints.

**4. Assess deployment feasibility in assistive ICT:&#x20;**&#x45;valuate the low-latency execution and voice-cloning capabilities of the optimized zero-shot TTS pipeline for potential integration into mobile assistive learning tools for neurodivergent learners with Autism Spectrum Disorder (ASD).





2. **REVIEW OF RELATED LITERATURES**



**Outline**

1.  Literatures for the problem
   1. General Problem
   2. Application Problem
   3. Theoretical Problem
2. Literatures for the solution
   1. Mathematical foundations
   2. Existing formulas
   3. Properties and proof of theorized solution
3. Literatures for the application
   1. Existing applications
4. Literatures for the evaluation
   1. Evaluation metric foundation