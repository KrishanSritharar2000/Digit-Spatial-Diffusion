# Digit-Spatial-Diffusion

Thesis: 👉 [Krishan-Sritharar-Thesis](../main/Krishan_Sritharar_Thesis.pdf)

In recent years, the growing field of artificial intelligence has opened up exciting opportunities
in the realm of image synthesis, with a particular interest in the potential
of text-to-image generation. Among the multitude of techniques available, diffusion
models have emerged as a particularly promising approach, offering the capability to
generate highly realistic images by sampling from complex probability distributions
over time. However, the complexity of spatially and semantically accurate image
generation from textual prompts remains a significant challenge within the field.


This project addresses these complexities by introducing a novel approach that integrates
spatial knowledge into image generation using diffusion models. We developed
a new methodology that extends traditional diffusion models by integrating
with scene graph, which are a non-ambiguous, easily-specified tool for representing
spatial relationships and semantics within a scene. Our unique two-stage generation
process is composed of an annotator model and a conditioned diffusion model,
which leverages the powers of Stable Diffusion and ControlNet to dictate the spatial
orientation of objects to the diffusion process.


We use a custom dataset comprising images of MNIST digits arranged in grid formations,
serving as an ideal sandbox to explore our proposed methodology’s effectiveness.
This choice of dataset focuses attention on the spatial layouts of objects,
eliminating any other variables which may interfere with the results. We train four
main methods of models: a baseline, a hand-crafted scene graph representation, and
two control models designed using ControlNet, represented as typed digits and dots.


Evaluation of each method was performed with our custom design accuracy metrics
and digit classifier. We observed object and relationship count accuracy, as well as
relationship integrity and thoroughly evaluated all the models we trained, providing
insights into surprising areas of performance. We deduced the dots approach as the
superior concept since it outperformed the baseline results by nearly 20%.


By highlighting the importance of spatial understanding in image generation and
by providing a methodology for fulfilling this request, this work makes a significant
contribution to the field and paves the way for more sophisticated applications that
will reshape the landscape of AI-driven image generation.
