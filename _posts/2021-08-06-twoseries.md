---
layout: post
title: "Two related series"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

$ \color{goldenrod}{\text{Th}} $: Consider sequence $ (a_j) $ in $ \mathbb{R}_ {\geq 0}$.   
Now $ \displaystyle \sum_{j=1}^{\infty} a_j $ converges if and only if $ \displaystyle \sum_{j=1}^{\infty} \dfrac{a_j}{1+a_j} $ converges.   
$ \color{goldenrod}{\text{Pf}}$: $ \underline{\implies} $ Clear as $ \frac{a_j}{1+a_j} \leq a_j $.   
$ \underline{\impliedby} $ As $ \frac{a_j}{1+a_j} \overset{j\to \infty}{\longrightarrow} 0 $, we have $ 0 \leq \frac{a_j}{1+a_j} &lt; \frac{1}{2} $ for all sufficiently large $ j $. (As usual, "$ P(n) $ for all sufficiently large $ n $" is short for "there is an $ N $ such that $ P(n) $ holds for all $ n \geq N $").   
So $ a_j &lt; 1 $ for all large enough $ j $. Hence $ a_j \leq 2 \left( \dfrac{a_j}{1+a_j} \right) $ for all large enough $ j $, as needed.   
> There is nothing special about taking $ \frac{1}{2} $ in the beginning. In general $ \frac{a_j}{1+a_j} = 1 - \frac{1}{1+a_j} $ goes to $ 0 $ as $ j \to \infty $, so $ \frac{1}{1+a_j} $ goes to $ 1 $, that is $ a_j \to 0 $ as $ j \to \infty $. Pick any $ \eta &gt; 0 $.  Now $ a_j &lt; \eta $ for all large enough $ j $. Hence $ a_j \leq (1+\eta) \frac{a_j}{1+a_j} $ for all large enough $ j $. 
