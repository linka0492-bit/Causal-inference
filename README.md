# Causal Inference for Mathematical and Physical Models

## 1. Scientific problem

The project studies how a known mathematical model of a physical system
can be **validated, compared with alternative models, and simplified
using causal inference**.

The central pipeline is

\[ `\text{physical law}`{=tex} `\rightarrow`{=tex}
`\text{differential equation / PDE}`{=tex} `\rightarrow`{=tex}
`\text{functional decomposition}`{=tex} `\rightarrow`{=tex}
`\text{graph }`{=tex}G `\rightarrow`{=tex} `\text{model family}`{=tex}
`\rightarrow`{=tex} `\text{observational fitting}`{=tex}
`\rightarrow`{=tex} `\text{model selection}`{=tex} `\rightarrow`{=tex}
`\text{interventions}`{=tex} `\rightarrow`{=tex}
`\text{edge validation}`{=tex}. \]

The important point is that a physical equation is treated not merely as
a formula, but as a **structured functional model**. Its structure is
represented by a graph whose vertices may be observable physical
quantities as well as latent intermediate computational quantities.

The final scientific question is:

> Can the structure of a mathematical physical model be supported by
> data, and can interventions determine which of its dependencies are
> genuinely necessary and which are redundant?

------------------------------------------------------------------------

## 2. Example supplied in the theoretical materials: forced Van der Pol oscillator

The provided presentation uses the forced Van der Pol oscillator as a
basic example:

\[ `\frac{d^2x}{dt^2}`{=tex} -`\mu`{=tex}(1-x\^2)`\frac{dx}{dt}`{=tex}
+x-A`\sin`{=tex}(`\omega `{=tex}t)=0. \]

Here

-   (x(t)) is the state;
-   (`\dot `{=tex}x=dx/dt);
-   (`\ddot `{=tex}x=d^2x/dt^2);
-   (`\mu`{=tex}) is the nonlinear parameter;
-   \(A\) is the forcing amplitude;
-   (`\omega`{=tex}) is the forcing angular frequency;
-   \(t\) is time.

Introducing

\[
X_1=`\ddot `{=tex}x,`\qquad `{=tex}X_2=x,`\qquad `{=tex}X_3=`\dot `{=tex}x,
\]

the equation can be written as

\[ X_1(t) =
`\mu`{=tex}(1-X_2(t)\^2)X_3(t)-X_2(t)+A`\sin`{=tex}(`\omega `{=tex}t).
\]

The supplied material explicitly represents this model as a
causal/functional graph with observable nodes (X_1,X_2,X_3,t) and latent
nodes (V_1,`\ldots`{=tex},V_6). fileciteturn0file1L10-L12
fileciteturn0file1L25-L47

This example is important because it shows that the graph is **not only
a graph of physical variables**. It also describes how the mathematical
expression is constructed from elementary operations.

------------------------------------------------------------------------

## 3. Functional graph and latent computational nodes

Suppose

\[ G=(V,E) \]

is the directed graph of a mathematical model.

There are two conceptually different kinds of nodes.

### Observable nodes

These correspond to physical quantities that can be measured:

\[ X_1(t),X_2(t),X_3(t),t. \]

For the Van der Pol example:

\[
X_1=`\ddot `{=tex}x,`\qquad `{=tex}X_2=x,`\qquad `{=tex}X_3=`\dot `{=tex}x.
\]

### Latent nodes

The nodes

\[ V_1,`\ldots`{=tex},V_6 \]

represent intermediate results of elementary mathematical operations.
The supplied presentation uses the dictionary

\[ `\mathcal `{=tex}D=
{`\sin`{=tex}(`\cdot`{=tex}),(`\cdot`{=tex})\^2,+,`\times`{=tex}}. \]

More generally,

\[ `\mathcal `{=tex}D= {f_1,f_2,`\ldots`{=tex},f_K} \]

is a fixed dictionary of admissible elementary functions.

Thus the graph represents a computational composition such as

\[ X_2`\rightarrow `{=tex}X_2\^2 `\rightarrow 1`{=tex}-X_2\^2
`\rightarrow `{=tex}(1-X_2\^2)X_3 \]

together with

\[ t,`\omega`{=tex}`\rightarrow `{=tex}`\sin`{=tex}(`\omega `{=tex}t),
\]

followed by addition of the resulting terms.

The presentation defines the causal model associated with a graph as

\[ f_G(w):x`\mapsto`{=tex}`\hat `{=tex}x, \]

and gives a parameterized Van der Pol expression of the form

\[ X_1(t) = w_1(w_4-w_5X_2\^2)`\times `{=tex}X_3(t) -w_2X_3(t)
+w_3`\sin`{=tex}(w_6t), \]

with the exact powers/parent assignments determined by the graph shown
in the supplied material. fileciteturn0file1L64-L76

The important abstraction is

\[ `\boxed{\text{graph}+\text{dictionary}+\text{parameters}
\Longrightarrow \text{mathematical expression}.}`{=tex} \]

------------------------------------------------------------------------

## 4. From differential equations to structural models

A general differential equation can be represented schematically as

\[
`\mathcal `{=tex}F(u,`\partial`{=tex}\_tu,`\nabla `{=tex}u,`\nabla`{=tex}\^2u,`\ldots`{=tex};`\theta`{=tex})=0.
\]

After isolating one quantity, for example,

\[ `\partial`{=tex}\_tu =
F(u,`\nabla `{=tex}u,`\nabla`{=tex}\^2u,`\ldots`{=tex};`\theta`{=tex}),
\]

the right-hand side can be decomposed into elementary operations.

For a PDE, a computational graph may therefore contain:

-   fields (u,v,p,`\ldots`{=tex});
-   spatial derivatives;
-   temporal derivatives;
-   products;
-   sums;
-   nonlinear functions;
-   physical parameters;
-   forcing terms.

For example, a nonlinear PDE term

\[ (u`\cdot`{=tex}`\nabla`{=tex})u \]

can itself be represented as a subgraph involving (u),
(`\nabla `{=tex}u), multiplication and summation.

The same principle applies to equations such as Navier--Stokes and
Maxwell's equations. The particular physics changes, but the abstract
construction remains:

\[ `\text{PDE}`{=tex} `\rightarrow`{=tex}
`\text{elementary functional decomposition}`{=tex} `\rightarrow`{=tex}
G. \]

------------------------------------------------------------------------

## 5. Structural causal interpretation

A structural causal model is commonly written as

\[ X_i=f_i(X\_{`\operatorname{pa}`{=tex}(i)},U_i), \]

where (`\operatorname{pa}`{=tex}(i)) are the parents of (X_i) in the
graph and (U_i) represents external or latent factors.

In this project, the structural model has an additional
mathematical-physics interpretation:

-   some nodes are physical observables;
-   some nodes are latent computational intermediates;
-   the functions are constrained by a dictionary;
-   the final function represents a differential equation or PDE.

Therefore the graph describes both **structure** and **mechanism**.

This distinction is crucial. A statistical correlation between two
measured variables is not automatically an edge in the physical model.
The causal interpretation must ultimately be tested by intervention.

------------------------------------------------------------------------

## 6. Observational data

Let the available dataset be

\[ D={z^{(1)},`\ldots`{=tex},z^{(N)}}. \]

For an ODE this may contain

\[ (X_1(t_i),X_2(t_i),X_3(t_i),t_i),
`\qquad `{=tex}i=1,`\ldots`{=tex},N. \]

For a PDE it may contain measurements over space and time, for example

\[ {u(x_i,t_j),v(x_i,t_j),p(x_i,t_j),`\ldots`{=tex}}. \]

For a fixed graph (G), let

\[ `\mathcal `{=tex}M_G=
{f_G(`\cdot`{=tex};w):w`\in`{=tex}`\mathcal `{=tex}W_G} \]

be the family of models having that structure.

Parameter estimation is then

\[ `\hat `{=tex}w_G =
`\arg`{=tex}`\max`{=tex}\_{w`\in`{=tex}`\mathcal `{=tex}W_G}
p(D`\mid `{=tex}G,w), \]

or, equivalently, minimization of a suitable loss.

For example,

\[ `\hat `{=tex}w_G = `\arg`{=tex}`\min`{=tex}*w `\sum`{=tex}*{i=1}\^{N}
`\left[
y^{(i)}-f_G(x^{(i)};w)
\right]`{=tex}\^2. \]

Thus there are two different problems:

1.  **parameter fitting for a fixed graph**;
2.  **selection of the graph itself**.

------------------------------------------------------------------------

## 7. Alternative models

Let

\[ `\mathcal `{=tex}G={G_1,`\ldots`{=tex},G_M} \]

be the allowed set of candidate graph structures.

Different graphs may represent models that differ in:

-   the presence or absence of an edge;
-   edge direction;
-   intermediate operations;
-   nonlinearities;
-   polynomial degree;
-   forcing terms;
-   decomposition into latent functions.

The model-selection problem is

\[ `\hat `{=tex}G =
`\arg`{=tex}`\max`{=tex}\_{G`\in`{=tex}`\mathcal `{=tex}G}
`\operatorname{Score}`{=tex}(G,D). \]

The definition of (`\mathcal `{=tex}G) and the dictionary is essential.
A statement such as "the model is the most probable" has meaning only
relative to a specified model space and probability model.

------------------------------------------------------------------------

## 8. Bayesian formulation

If a prior distribution over graphs is given,

\[ p(G), \]

then

\[ p(G`\mid `{=tex}D) = `\frac{p(D\mid G)p(G)}{p(D)}`{=tex}. \]

The preferred model is

\[ `\boxed{
G^\ast=
\arg\max_{G\in\mathcal G}p(G\mid D)
}`{=tex}. \]

The marginal likelihood is

\[ p(D`\mid `{=tex}G) = `\int`{=tex}
p(D`\mid `{=tex}G,w)p(w`\mid `{=tex}G),dw. \]

Hence

\[ G\^`\ast`{=tex} = `\arg`{=tex}`\max`{=tex}\_G p(D`\mid `{=tex}G)p(G).
\]

If a Bayesian treatment is not used, the same model-selection role can
be played by an information criterion or another score. For example,

\[ `\mathrm{BIC}`{=tex} = -2`\log`{=tex}`\hat `{=tex}L+k`\log `{=tex}N.
\]

The causal-chamber material gives BIC with a Gaussian likelihood as an
example of observational causal-structure scoring. fileciteturn2file0

------------------------------------------------------------------------

## 9. Why observational fit is insufficient

Suppose

\[ G_1`\neq `{=tex}G_2 \]

but

\[ p\_{G_1}(D)`\approx `{=tex}p\_{G_2}(D). \]

Then observational data may not distinguish the mechanisms.

Two models can have nearly identical

\[ p(Y`\mid `{=tex}X) \]

while predicting different effects under intervention:

\[ p\_{G_1}(Y`\mid `{=tex}do(X=x)) `\neq`{=tex}
p\_{G_2}(Y`\mid `{=tex}do(X=x)). \]

Therefore a good observational fit does not establish causal
correctness.

This is the fundamental reason interventions are part of the project.

------------------------------------------------------------------------

# 10. Causal inference and interventions

An intervention

\[ do(X=x) \]

means that the mechanism generating (X) is replaced by an external
assignment

\[ X:=x. \]

In a structural model

\[ X=f_X(`\operatorname{pa}`{=tex}(X),U_X), \]

the intervention replaces that equation with

\[ X=x. \]

Other structural mechanisms remain unchanged.

This is different from conditioning:

\[ p(Y`\mid `{=tex}X=x) \]

is observational, whereas

\[ p(Y`\mid `{=tex}do(X=x)) \]

is interventional.

In general,

\[ p(Y`\mid `{=tex}X=x)`\neq `{=tex}p(Y`\mid `{=tex}do(X=x)). \]

The causal-chamber materials use exactly this intervention-based
interpretation: an edge (X`\to `{=tex}Y) means that an intervention on
(X) changes the distribution of subsequent measurements of (Y).
fileciteturn5file0

------------------------------------------------------------------------

## 11. "Breaking" the model

The phrase "break the model" means performing a controlled intervention
that deliberately violates the normal mechanism of a variable.

The procedure is:

1.  choose a variable or mechanism;
2.  force it to a prescribed value or regime;
3.  observe the downstream response;
4.  compare the response with the prediction of candidate models.

For a proposed edge

\[ X`\rightarrow `{=tex}Y, \]

a causal effect is suggested when, for some (x_1`\neq `{=tex}x_2),

\[ p(Y`\mid `{=tex}do(X=x_1)) `\neq`{=tex} p(Y`\mid `{=tex}do(X=x_2)).
\]

The strength of the evidence depends on the intervention set, noise,
sample size and model assumptions.

------------------------------------------------------------------------

# 12. Model validation under intervention

Let

\[ `\mathcal `{=tex}I={I_1,`\ldots`{=tex},I_K} \]

be the set of admissible interventions.

For each intervention (I_k), the candidate model predicts

\[ P_G(`\cdot`{=tex}`\mid `{=tex}I_k). \]

The experiment produces an empirical distribution

\[ `\widehat `{=tex}P(`\cdot`{=tex}`\mid `{=tex}I_k). \]

A general interventional discrepancy is

\[ `\Delta`{=tex}*{`\mathrm{int}`{=tex}}(G) = `\sum`{=tex}*{k=1}\^{K}
`\lambda`{=tex}\_k d`\left`{=tex}(
`\widehat `{=tex}P(`\cdot`{=tex}`\mid `{=tex}I_k),
P_G(`\cdot`{=tex}`\mid `{=tex}I_k) `\right`{=tex}), \]

where (d) is an appropriate distance and
(`\lambda`{=tex}\_k`\ge0`{=tex}).

For deterministic physical simulations, (d) may instead compare
trajectories, fields or residuals.

A model is interventional-adequate if its predicted response remains
consistent with the experimentally observed response over the chosen
intervention set.

------------------------------------------------------------------------

# 13. Edge ablation and redundant edges

Let

\[ G=(V,E) \]

be the selected graph and let

\[ e=(X`\rightarrow `{=tex}Y)`\in `{=tex}E. \]

Remove the edge:

\[ G\^{-e} = (V,E`\setminus`{=tex}{e}). \]

The original and reduced models are then compared.

Define

\[ `\Delta`{=tex}*e = `\sum`{=tex}*{I`\in`{=tex}`\mathcal `{=tex}I}
`\lambda`{=tex}*I d `\left`{=tex}( P_G(`\cdot`{=tex}`\mid `{=tex}I),
P*{G\^{-e}}(`\cdot`{=tex}`\mid `{=tex}I) `\right`{=tex}). \]

If

\[ `\Delta`{=tex}\_e\<`\varepsilon`{=tex}, \]

for a pre-specified tolerance (`\varepsilon`{=tex}), then the edge is a
candidate for being redundant.

The important qualification is:

> An edge is redundant **relative to the specified model class, data and
> intervention set**.

It is not valid to conclude from finite experimental evidence that the
physical effect is absolutely nonexistent.

------------------------------------------------------------------------

# 14. Observational versus interventional equivalence

Two models can satisfy

\[ G_1`\sim`{=tex}\_{`\mathrm{obs}`{=tex}}G_2 \]

if they produce the same observational distribution.

But they may fail to satisfy

\[ G_1`\sim`{=tex}\_{`\mathrm{int}`{=tex},`\mathcal `{=tex}I}G_2 \]

if some intervention in (`\mathcal `{=tex}I) produces different
predictions.

Therefore

\[ `\text{observational equivalence}`{=tex}
`\not`{=tex}`\Rightarrow`{=tex}
`\text{interventional equivalence}`{=tex}. \]

Interventions shrink the set of models compatible with the evidence.

This is the key causal advantage over ordinary fitting.

------------------------------------------------------------------------

# 15. Connection with causal discovery

Classical causal discovery attempts to infer

\[ D`\rightarrow`{=tex}`\widehat `{=tex}G. \]

The present project has a related but different goal.

Here there is already a physically motivated model

\[ G\_{`\mathrm{physics}`{=tex}}, \]

and the question is whether the data and interventions support that
structure:

\[ G\_{`\mathrm{physics}`{=tex}} `\stackrel{?}{\approx}`{=tex}
G\_{`\mathrm{data}`{=tex}}. \]

Thus the task is best understood as **causal validation and causal model
selection for mathematical physical models**, rather than only generic
causal discovery.

The supplied causal-chamber work illustrates standard observational and
interventional causal-discovery tasks and evaluates graph recovery by
precision and recall. fileciteturn2file0

------------------------------------------------------------------------

# 16. Connection with symbolic regression

Symbolic regression usually seeks

\[ D`\rightarrow `{=tex}f \]

for a mathematical expression (f).

Here the search is structured by

\[ D `\rightarrow`{=tex} (G,w) `\rightarrow`{=tex} f_G(w). \]

The graph determines the structure and the dictionary determines the
allowed elementary operations.

Thus the project can be viewed as a form of **structured symbolic
regression with causal/interventional validation**.

The distinction is important:

\[ `\text{symbolic fit}`{=tex} `\neq`{=tex}
`\text{causal validation}`{=tex}. \]

A formula can fit data well and still contain unnecessary dependencies.

------------------------------------------------------------------------

# 17. Generalization to PDEs

Let a physical PDE be

\[ `\mathcal `{=tex}F\^`\ast[u]=0`{=tex}. \]

After decomposition,

\[ `\mathcal `{=tex}F\^`\ast`{=tex} =
f_n`\circ `{=tex}f\_{n-1}`\circ`{=tex}`\cdots`{=tex}`\circ `{=tex}f_1,
\]

which gives a graph

\[ G^`\ast`{=tex}=(V,E^`\ast`{=tex}). \]

For each candidate

\[ G`\in`{=tex}`\mathcal `{=tex}G \]

define

\[ `\mathcal `{=tex}M_G =
{`\mathcal `{=tex}F_G(`\cdot`{=tex};w):w`\in`{=tex}`\mathcal `{=tex}W_G}.
\]

The observational task is

\[ `\hat `{=tex}w_G = `\arg`{=tex}`\max`{=tex}\_w p(D`\mid `{=tex}G,w),
\]

followed by

\[ `\hat `{=tex}G\_{`\mathrm{obs}`{=tex}} = `\arg`{=tex}`\max`{=tex}\_G
p(G`\mid `{=tex}D). \]

Interventions then generate modified systems

\[ do(I_k), \]

which are solved or experimentally measured to obtain

\[ `\widehat `{=tex}P(`\cdot`{=tex}`\mid `{=tex}I_k). \]

The model is validated through

\[ P\_{`\hat `{=tex}G}(`\cdot`{=tex}`\mid `{=tex}I_k) `\approx`{=tex}
`\widehat `{=tex}P(`\cdot`{=tex}`\mid `{=tex}I_k). \]

Finally each edge is ablated and tested.

This framework is applicable to PDEs such as Navier--Stokes or Maxwell
equations provided that their variables and differential operators are
represented in the chosen graph/dictionary formalism.

------------------------------------------------------------------------

# 18. Complete mathematical formulation

Given:

\[ D={z\^{(i)}}\_{i=1}\^{N}, \]

a set of candidate graphs

\[ `\mathcal `{=tex}G={G_1,`\ldots`{=tex},G_M}, \]

a dictionary

\[ `\mathcal `{=tex}D, \]

a graph prior

\[ p(G), \]

and an intervention set

\[ `\mathcal `{=tex}I, \]

define for each graph

\[ `\mathcal `{=tex}M_G=
{f_G(`\cdot`{=tex};w):w`\in`{=tex}`\mathcal `{=tex}W_G}. \]

### Parameter fitting

\[ `\hat `{=tex}w_G = `\arg`{=tex}`\max`{=tex}\_w p(D`\mid `{=tex}G,w).
\]

### Model selection

\[ `\boxed{
\hat G
=
\arg\max_{G\in\mathcal G}
p(G\mid D)
}`{=tex} \]

or an explicitly defined alternative score.

### Interventional validation

For every

\[ I`\in`{=tex}`\mathcal `{=tex}I, \]

compare

\[ P\_{`\hat `{=tex}G}(`\cdot`{=tex}`\mid `{=tex}I) \]

with

\[
`\widehat `{=tex}P\_{`\mathrm{data}`{=tex}}(`\cdot`{=tex}`\mid `{=tex}I).
\]

### Edge ablation

For each

\[ e`\in `{=tex}E\_{`\hat `{=tex}G}, \]

construct

\[ `\hat `{=tex}G\^{-e} = (V,E\_{`\hat `{=tex}G}`\setminus`{=tex}{e}).
\]

Then compute

\[ `\Delta`{=tex}*e = `\sum`{=tex}*{I`\in`{=tex}`\mathcal `{=tex}I}
`\lambda`{=tex}*I d `\left`{=tex}(
P*{`\hat `{=tex}G}(`\cdot`{=tex}`\mid `{=tex}I),
P\_{`\hat `{=tex}G\^{-e}}(`\cdot`{=tex}`\mid `{=tex}I) `\right`{=tex}).
\]

If

\[ `\Delta`{=tex}\_e\<`\varepsilon`{=tex}, \]

the edge is considered redundant relative to the experimental setting.

------------------------------------------------------------------------

# 19. What the diploma should establish

The scientific argument should have three separate layers.

### 19.1. Observational adequacy

The selected model explains the available sample:

\[ P\_{`\hat `{=tex}G}(D)`\approx `{=tex}P\_{`\mathrm{data}`{=tex}}(D).
\]

### 19.2. Model preference

Among the allowed alternatives,

\[ `\hat `{=tex}G = `\arg`{=tex}`\max`{=tex}\_G
`\operatorname{Score}`{=tex}(G,D). \]

If a Bayesian formulation is used, this becomes

\[ `\hat `{=tex}G = `\arg`{=tex}`\max`{=tex}\_G p(G`\mid `{=tex}D). \]

### 19.3. Causal/interventional adequacy

The selected structure predicts responses to interventions:

\[ P\_{`\hat `{=tex}G}(Y`\mid `{=tex}do(X=x)) `\approx`{=tex}
P\_{`\mathrm{data}`{=tex}}(Y`\mid `{=tex}do(X=x)). \]

Only the combination of these levels provides strong evidence that the
mathematical structure is not merely a convenient statistical
approximation.

------------------------------------------------------------------------

# 20. What must be fixed rigorously

For the final thesis formulation, the following objects must be
explicitly defined.

### Model space

\[ `\mathcal `{=tex}G. \]

Which graphs are allowed? Are latent nodes allowed? Are cycles allowed?
How is time represented?

### Dictionary

\[ `\mathcal `{=tex}D. \]

Which elementary functions and differential operators may occur?

### Parameter spaces

\[ `\mathcal `{=tex}W_G. \]

### Likelihood / noise model

\[ p(D`\mid `{=tex}G,w). \]

### Prior

\[ p(G),`\qquad `{=tex}p(w`\mid `{=tex}G) \]

if Bayesian model comparison is used.

### Intervention class

\[ `\mathcal `{=tex}I. \]

Which variables can be manipulated and how?

### Comparison metric

\[ d(P,Q). \]

### Redundancy threshold

\[ `\varepsilon`{=tex}. \]

Without these definitions, statements such as "the most probable model"
or "the edge is redundant" remain informal.

------------------------------------------------------------------------

# 21. Important interpretation of "redundant"

The correct claim is not

> "The edge does not exist physically."

The defensible claim is

> "The edge is not required to reproduce the observational and
> interventional behaviour considered, within the specified model class,
> dictionary, noise assumptions and intervention set."

This distinction is essential because a missing detectable effect may be
caused by:

-   finite sample size;
-   measurement noise;
-   weak causal effects;
-   insufficient intervention strength;
-   insufficient intervention coverage;
-   latent confounding;
-   model misspecification;
-   an incomplete dictionary.

The causal-chamber paper makes a related caution: absence of an edge
does not automatically imply absence of every physical causal effect,
since weak effects, latent variables and confounding may remain
undetected. fileciteturn5file0

------------------------------------------------------------------------

# 22. Conceptual pipeline for the thesis

``` text
                 PHYSICAL SYSTEM
                       │
                       ▼
              Differential equation
                       │
                       ▼
              Functional decomposition
                       │
                       ▼
       ┌─────────────────────────────────┐
       │ Graph G                          │
       │ observable nodes                 │
       │ latent computational nodes      │
       │ elementary functions / dictionary│
       └─────────────────────────────────┘
                       │
                       ▼
              Model family M_G
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      observational data   interventional data
             │                   │
             ▼                   ▼
      parameter fitting   intervention prediction
             │                   │
             └─────────┬─────────┘
                       ▼
                model comparison
                       │
                       ▼
                preferred graph
                       │
                       ▼
                  edge ablation
                       │
                       ▼
             intervention comparison
                       │
                 ┌─────┴─────┐
                 ▼           ▼
             necessary     redundant
               edge          edge
```

------------------------------------------------------------------------

# 23. Final statement of the research task

The project can be summarized as follows:

> **Given a mathematical model of a physical system, represent its
> differential equation/PDE as a structured graph of observable
> variables and latent computational operations. Given a dataset and a
> predefined family of alternative graphs/functions, estimate the
> parameters of each candidate and determine which model is most
> strongly supported by the data. Then perform controlled interventions
> and compare the predicted and observed interventional behaviour.
> Finally, remove individual edges and use the interventional response
> to determine which dependencies are necessary and which are
> redundant.**

The core scientific hypothesis is therefore:

\[ `\boxed{
\text{a physically correct mathematical structure should be supported not only by observational fit,
but also by its behaviour under interventions}
}`{=tex} \]

and the ultimate structural objective is

\[ `\boxed{
\text{identify the simplest causally adequate structure within the chosen model class.}
}`{=tex} \]

------------------------------------------------------------------------

## 24. Terminology

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  PDE                                 Partial Differential Equation

  ODE                                 Ordinary Differential Equation

  DAG                                 Directed Acyclic Graph

  SCM                                 Structural Causal Model

  Causal Inference                    Inference about effects of
                                      interventions

  Causal Discovery                    Recovery of causal structure from
                                      data

  Observational data                  Data collected without targeted
                                      intervention

  Interventional data                 Data collected after controlled
                                      intervention

  (do(X=x))                           Intervention setting (X) externally
                                      to (x)

  Latent node                         Unobserved intermediate/model
                                      variable

  Dictionary                          Set of admissible elementary
                                      functions

  Model family (`\mathcal `{=tex}M_G) All parameterized models with graph
                                      (G)

  Model selection                     Choosing among candidate structures

  Marginal likelihood                 (p(D`\mid `{=tex}G)), likelihood
                                      integrated over parameters

  Posterior model probability         (p(G`\mid `{=tex}D))

  Edge ablation                       Removing an edge and testing the
                                      resulting model

  Redundant edge                      Edge whose removal is
                                      indistinguishable under the
                                      specified tests

  Interventional equivalence          Agreement of models under a
                                      specified intervention class
  -----------------------------------------------------------------------

## 25. Sources in the project context

The Van der Pol presentation provides the concrete graph/function
formalism: observable variables (X_1,X_2,X_3), latent variables
(V_1,`\ldots`{=tex},V_6), the elementary-function dictionary, the
adjacency representation and the parameterized (f_G(w)).
fileciteturn0file1L49-L76

The supplied causal-chamber material motivates the interventional
interpretation of causal graphs and gives examples of observational and
interventional causal-structure recovery. fileciteturn5file0

This README intentionally describes the **scientific problem and theory
only**. Implementation details, source-code architecture, notebooks,
experiments and individual repository files are outside its scope.
