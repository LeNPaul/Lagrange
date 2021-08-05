---
layout: post
title: "Using Partial Summation"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

$\color{goldenrod}{\text{Th}}$: Let $ (g(k)) $ be a sequence in a complete normed space, and $ (a_k) $ a sequence in $ \mathbb{R}_ {\geq 0} $ decreasing to $ 0 $. Suppose seq $ G(k) := g(1) + \ldots + g(k) $ is bounded. Then $ \displaystyle \sum_{k=1}^{\infty} a_k g(k) $ converges.   
$\color{goldenrod}{\text{Pf}} $: To get rid of $ g(k) $s, we can write   
$$ \begin{aligned} \sum_{k=1}^{n} a_k g(k) &= a_1 G(1) + \sum_{k=2}^{n} a_k (G(k) - G(k-1)) \\\\\\ &= \sum_{k=1}^{n} a_k G(k) - \sum_{k=1}^{n-1} a_{k+1} G(k) \\\\\\ &= a_n G(n) + \sum_{k=1}^{n-1} (a_{k} - a_{k+1}) G(k) \end{aligned}$$   
Let $ M &gt; 0 $ be such that $ \| G(k) \| \leq M $ for all $ k $.   
Now for $ m &gt; n $ we have   
$$\begin{aligned} \left| \sum_{k=n+1}^{m} a_k g(k) \right| &= \left| (a_m G(m) - a_n G(n)) + \sum_{k=n}^{m-1} (a_k - a_{k+1}) G(k) \right| \\\\\\ &\leq a_m M + a_n M + \sum_{k=n}^{m-1} (a_k - a_{k+1}) M \\\\\\ &= 2 a_n M &nbsp; ( \to 0 \text{ as } n \to \infty) \end{aligned}$$   
So seq of partial sums of $ \sum a_k g(k) $ is Cauchy, hence convergent. 

----

$\color{goldenrod}{\text{Eg}}$: Consider the complex series $ \displaystyle \sum_{k=1}^{\infty} \dfrac{z^k}{k} .$   
**Zone of absolute convergence** : It converges absolutely for $ \| z \| &lt; 1 $ (because $ \frac{\|z\|^k}{k} \leq \|z\|^k $ and $ \sum \|z\|^k $ converges). If $ \|z\| = 1 $, it cant converge absolutely. If $ \|z\| &gt; 1 $, it cant converge as $ \frac{\|z\|^k}{k} \to \infty $ (write $ \|z\| = 1+h $ and expand).   
**Zone of convergence** : We saw it converges for $ \|z\| &lt; 1 $ and doesnt converge for $ \|z\| &gt; 1 $. For points with $ \|z\| = 1 $ ... It doesnt converge for $ z = 1 $. For points on unit circle other than $ 1 $ it converges (take $ g(k) = z^k $ and $ a_k = \frac{1}{k} $).   
> Putting $ z = (-1) $, we see $\displaystyle \sum_{k=1}^{\infty} \frac{(-1)^k}{k} $ converges.   
Putting $ z = - e^{i} $, and considering imaginary parts of partial sums, we see $ \displaystyle \sum_{k=1}^{\infty} (-1)^k\frac{\sin(k)}{k} $ converges. 
