---
layout: post
title: "Reals are complete"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Let $(a_n)$ be a Cauchy sequence of reals. It is bounded [ There is an $N$ such that $ a_N, a_{N+1}, \ldots $ are in $ (a_N - 1, a_N + 1) $. Now $ \max \{ |a_1|, \ldots, |a_{N-1}|, |a_N|+1 \} $ is $ \geq $ each $ | a_n | $ ]. 

So $ \alpha_{j} := \sup\{a_j, a_{j+1}, \ldots \} $ are well-defined, bounded (and decreasing). Therefore they converge, to $ \alpha := \inf\{ \alpha_1, \alpha_2, \ldots \} $. 

Let $\epsilon > 0 $. There is an $ N (=N_{\epsilon}) $ such that $ a_N, a_{N+1}, \ldots $ are in $ (a_N - \epsilon, a_N + \epsilon) $. So $ \alpha_N, \alpha_{N+1}, \ldots $ are in $ [ a_N - \epsilon, a_N + \epsilon ] $, and hence so is $ \alpha $. 

Finally $ a_N, a_{N+1}, \ldots $ and $ \alpha $ are all in $ [ a_N - \epsilon, a_N + \epsilon ] $, ensuring each $ | a_N - \alpha |, |a_{N+1} - \alpha |, \ldots $ is $ \leq 2 \epsilon $. 
