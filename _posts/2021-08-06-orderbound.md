---
layout: post
title: "Bounding order of a permutation"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

$ \color{goldenrod}{\text{Th}} $: For any $ \sigma \in S_n $, we have $ \text{ord}(\sigma) \leq (e^{\frac{1}{e}})^n \text{ } ( \leq 1.445^n ). $   
$ \color{goldenrod}{\text{Pf}} $: Let $ \sigma = \sigma_1 \ldots \sigma_k $ be decomposition into disjoint cycles, and $ \| \sigma_i \| =: \ell_i $. Now $ \text{ord}(\sigma) $ $ = \text{LCM}(\ell_1, \ldots, \ell_k) $ $ \leq \ell_1 \ldots \ell_k $ $ \leq \left( \dfrac{\ell_1 + \ldots + \ell_k}{k} \right)^k $ $ = \left( \dfrac{n}{k} \right)^k $ $ = e^{k( \log(n) - \log(k) )}. $   
$ f(x) := x (\log(n) - \log(x)) $ defined on $ \mathbb{R}_ {&gt; 0} $ has $ f'(x) = \log(n) - \log(x) - 1 $. Now $ f' $ is $ &gt; 0 $ on $ (0, \frac{n}{e}) $, $ 0 $ at $ \frac{n}{e} $, and $ &lt; 0 $ on $ (\frac{n}{e}, \infty) $. So $ f $ is increasing on $ (0, \frac{n}{e}] $, reaches a maximum value of $ f(\frac{n}{e}) = \frac{n}{e} $, and decreases on $ [\frac{n}{e}, \infty) $.   
So $ \text{ord}(\sigma) \leq e^{k( \log(n) - \log(k) )} $ $ \leq e^{\frac{n}{e}} .$

---

$\color{goldenrod}{\text{Eg}} $: For $ n = 10 $, we see order of any $ \sigma \in S_{10} $ is $ \leq 40 $. 
