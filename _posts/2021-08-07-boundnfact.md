---
layout: post
title: "Bounding n!"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

$ \log(n!) = \log(1) + \ldots + \log(n) $. 

This sum is $ \int_{1}^{2} \log(2) + \ldots + \int_{n-1}^{n} \log(n) $ $ \geq \int_{1}^{2} \log(t)dt + \ldots + \int_{n-1}^{n} \log(t)dt $ $ = \int_{1}^{n} \log(t) dt = n \log(n) - n +1 $. 

Also $ \log(2) + \ldots + \log(n-1) $ $ = \int_{2}^{3} \log(2) + \ldots + \int_{n-1}^{n} \log(n-1) $ $ \leq \int_{2}^{3} \log(t)dt + \ldots + \int_{n-1}^{n} \log(t)dt $ $ \leq \int_{1}^{n} \log(t)dt = n \log(n) - n + 1 $.   
So adding $ \log(n) $,   
$ \log(1) + \ldots + \log(n) \leq \log(n) + (n \log(n) - n + 1). $

Finally,   
$ n \log(n) - n + 1 $ $ \leq \log(1) + \ldots + \log(n)$ $ \leq \log(n) + (n \log(n) - n + 1). $   
Taking exponential,   
$ n^n e^{-n} e \leq n! \leq n^{n+1} e^{-n} e $.

---

Edit [Ok its pointless to avoid pictures] 

![pic](https://i.imgur.com/qsMihf1_d.webp?maxwidth=640&shape=thumb&fidelity=medium) 
