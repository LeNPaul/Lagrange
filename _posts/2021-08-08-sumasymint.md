---
layout: post
title: "Sum asymptotic to integral"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Consider an integrable decreasing function $ f : [2, \infty) \to \mathbb{R}_{&gt; 0} $. 

$ \int_{2}^{x} f(t) dt = \int_{2}^{3} f + \ldots + \int_{\lfloor x \rfloor -1}^{\lfloor x \rfloor} f + \int_{\lfloor x \rfloor}^{x} f $.   
Using $ f(i+1) \leq \int_{i}^{i+1} f \leq f(i) $, we get $ f(3) + \ldots + f(\lfloor x \rfloor) + \int_{\lfloor x \rfloor}^{x} f $ $ \leq \int_{2}^{x} $ $ \leq f(2) + \ldots + f(\lfloor x \rfloor - 1) + \int_{\lfloor x \rfloor}^{x} f $.   
As $ \int_{\lfloor x \rfloor}^{x} f \leq f(\lfloor x \rfloor) (x - \lfloor x \rfloor) $ $ \leq f(\lfloor x \rfloor) $, we get $ f(3) + \ldots + f(\lfloor x \rfloor)$ $ \leq \int_{2}^{x} f $ $ \leq f(2) + \ldots + f(\lfloor x \rfloor) $, that is $ -f(2) \leq \int_{2}^{x} f - \sum_{2 \leq k \leq x} f(k) \leq 0 .$ 

So $ \sum_{2\leq k \leq x} f(k) = \int_{2}^{x} f + \mathcal{O}(1) .$

If we further have $ \int_{2}^{x} f \to \infty $ as $ x \to \infty $, dividing above eqn by $ \int_{2}^{x} f $ and taking $ x \to \infty $ gives $ \displaystyle \sum_{2 \leq k \leq x} f(k) \sim \int_{2}^{x} f(t) \text{&nbsp;} dt $. 

