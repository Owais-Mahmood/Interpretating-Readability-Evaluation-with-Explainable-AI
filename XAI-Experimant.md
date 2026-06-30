## XAI methods we may use and compare

We will compare the methods at three levels: **token attribution**, **linguistic edit and concept explanations**, and **causal or mechanistic explanations**.

| XAI family                              | Methods                                                                                                                 | What they explain                                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Gradient based attribution**          | Integrated Gradients, Layer Gradient × Activation, Gradient × Input or saliency                                         | Which source and simplified tokens most influenced each predicted strategy                                      |
| **Perturbation based attribution**      | Occlusion, Leave One Token Out, Value Zeroing                                                                           | How the prediction changes when a token or aligned span is removed or neutralised                               |
| **Shapley and local surrogate methods** | SHAP, KernelSHAP, LIME                                                                                                  | Approximate the contribution of each token or feature to the prediction                                         |
| **Attention based explanation**         | Raw attention, attention rollout or aggregation, source to target attention, gradient weighted label specific attention | Which source and target tokens interact and whether attention differs across strategy labels                    |
| **Relevance propagation**               | AttnLRP                                                                                                                 | Propagates relevance through transformer attention and feed forward components                                  |
| **Human edit grounded explanation**     | Attribution overlap with observed deletions, additions, substitutions, reorderings and aligned source target spans      | Whether model explanations correspond to changes actually made by human simplifiers                             |
| **Pattern grounded explanation**        | Attribution overlap with linguistic simplification patterns, dependency relations and strategy annotations              | Whether the classifier relies on linguistically meaningful patterns                                             |
| **Counterfactual explanation**          | Controlled token changes, span replacement, deletion, addition, reordering and minimal counterfactual examples          | What smallest change would remove, introduce or change a predicted strategy                                     |
| **Pattern intervention explanation**    | Deliberately insert or remove a simplification pattern and measure the prediction change                                | Tests whether the identified linguistic pattern is causally important                                           |
| **Concept based explanation**           | CLARITY inspired rationale → concept → prediction analysis                                                              | Connects tokens to interpretable concepts and then to strategy predictions                                      |
| **Probing based explanation**           | Linear probes, MLP probes, syntactic probes, dependency probes and CEFR or readability probes                           | Determines what linguistic knowledge is encoded in model representations                                        |
| **Explanation summarisation**           | ExSUM and structured explanation summaries                                                                              | Produces compact global summaries of the patterns associated with each strategy                                 |
| **Delta explanation**                   | Δ explanations between the original and simplified sentence representations                                             | Explains what changed internally when moving from the complex sentence to its simplification                    |
| **Regional and cohort explanation**     | Cohort explanations, regional explanations and subgroup discovery                                                       | Identifies explanation patterns shared by languages, strategies, domains or sentence groups                     |
| **Cross lingual explanation**           | Cross language attribution and concept consistency                                                                      | Tests whether the same strategy is explained similarly in Arabic, Catalan, English, Spanish, French and Italian |
| **Mechanistic explanation**             | Circuit tracing, component attribution and Attributor style analysis                                                    | Identifies the layers, attention heads and internal components responsible for a strategy prediction            |
| **Graph based explanation**             | Explanation graph or logic graph comparison                                                                             | Compares token → pattern → concept → strategy reasoning structures across methods and models                    |
| **Uncertainty aware explanation**       | Conformal prediction and explanation analysis by confidence level                                                       | Distinguishes reliable explanations from explanations attached to uncertain predictions                         |

## Core head to head comparison

The principal quantitative comparison should contain:

1. **Integrated Gradients**
2. **Layer Gradient × Activation**
3. **Occlusion or Leave One Token Out**
4. **SHAP**
5. **Value Zeroing**
6. **AttnLRP**
7. **Raw attention**, used only as a baseline
8. **Gradient weighted label specific attention**
9. **Human edit grounded attribution**
10. **Pattern intervention explanations**

