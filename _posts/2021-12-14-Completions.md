---
layout: post
title: "Completions"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

(Ref: Kreyszig's "Functional Analysis"; [P. Tamaroff's post](https://math.stackexchange.com/questions/2477496/what-is-the-cauchy-completion-of-a-metric-space/2477613#2477613)) 

Let ${ (X,d) }$ be a metric space. A metric space ${ (\hat{X}, \hat{d}) }$ with the properties  
* ${ (\hat{X}, \hat{d}) }$ is complete   
* There is an embedding ${ (X,d) \overset{T}{\hookrightarrow} (\hat{X}, \hat{d})}$   
* Every ${ p \in \hat{X} }$ is the limit of some Cauchy seq contained in ${ T(X), }$ that is ${ \overline{T(X)} = \hat{X} }$ 

is called a *completion* of ${ (X,d) }.$ 

Eg: ${ \mathbb{R} }$ is a completion of ${ \mathbb{Q} }.$ 

---

**Th**: Let ${ (X,d) }$ be a metric space. It has a completion. Further any two completions of it are isomorphic.
   
**Pf**: [Existence] Consider the complete space ${ \mathcal{B}(X,\mathbb{R}) ,}$ of all bounded functions on set ${ X }$ with sup norm. Fix ${ a \in X }.$ We [have](https://bvenkatakarthik.github.io/EmbedMetric) the Kuratowski embedding  ${ (X, d) \overset{\varphi}{\hookrightarrow} ( \mathcal{B}(X, \mathbb{R}), d _{\text{sup}} ) }$ sending ${ x \mapsto (f _x - f _a) },$ where ${ f _p (q) := d(p,q) }.$ Now ${ (\overline{\varphi(X)} , d _{\text{sup}}) }$ is a valid completion of ${ (X,d) }.$  
  
[Uniqueness] Suppose ${ (\hat{X}, \hat{d}), (\tilde{X}, \tilde{d}) }$ are two completions of ${ (X,d) }.$ We want an isomorphism ${ (\hat{X}, \hat{d}) \to (\tilde{X}, \tilde{d}) }.$   
There are isomorphisms ${ (X, d) \overset{S}{\to} (S (X), \hat{d}) \subseteq (\hat{X}, \hat{d}) }$ and ${ (X, d) \overset{T}{\to} (T (X),  \tilde{d}) \subseteq (\tilde{X}, \tilde{d}) },$ with ${ \overline{S (X)} = \hat{X} }$ and ${ \overline{T (X)} = \tilde{X} }.$   
One can try defining a map ${ \hat{X} \overset{f}{\to} \tilde{X} }$ as follows : Let ${ \hat{x} \in \hat{X} }.$ Pick a seq ${ (S (x _j))  }$ in ${ S (X) }$ converging to ${ \hat{x} }.$ Now ${ (S (x _j) ) }$ is Cauchy, and due to the isomorphisms so are ${ (x _j) }$ and ${ (T (x _j) ) }.$ Set ${ f(\hat{x}) := \lim _{j \to \infty} T (x _j ). }$   
This is a well-defined map : Let ${ \hat{x} \in \hat{X} },$ and suppose ${ (S (x _j) ), }$ ${ (S (y _j) ) }$ are two sequences in ${ S(X) }$ converging to ${ \hat{x} }.$ Like above, ${ (T(x _j)), }$ ${ (T(y _j)) }$ are Cauchy. We should show ${ \lim _{j \to \infty} T(x _j) = \lim _{j \to \infty} T(y _j) .}$ Since ${ \tilde{d}(T(x _j), T(y _j)) }$ ${ = \hat{d} (S(x _j), S(y _j)) \to 0 }$ we have ${ \tilde{d}(\lim T(x _j), \lim T(y _j) ) }$ ${ \leq \tilde{d}(\lim T(x _j), T(x _j) )  }$ ${ + \tilde{d}(T(x _j), T(y _j)) }$ ${ + \tilde{d} (T(y _j), \lim T(y _j))  }$ ${ \to 0 }$ as needed.   
Similarly one checks ${ f }$ is distance preserving and a bijection, which ensures its an isomorphism. 
