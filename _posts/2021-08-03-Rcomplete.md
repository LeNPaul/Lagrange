---
layout: post
title: "Reals are complete"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---


Let $(a_n)$ be a Cauchy sequence of reals. It is bounded.   
>There is an $N$ such that $ a_N, a_{N+1}, \ldots $ are in $ (a_N - 1, a_N + 1) $. Now $ \max \lbrace \vert a_1 \vert , \ldots, \vert a_{N-1}\vert, \vert a_N \vert+1 \rbrace $ is $ \geq $ each $ \vert a_n \vert .$ 

So $ \alpha_{j} := \sup\lbrace a_j, a_{j+1}, \ldots \rbrace $ are well-defined, bounded (and decreasing). Therefore they converge, to $ \alpha := \inf\lbrace \alpha_1, \alpha_2, \ldots \rbrace $. 

Let $\epsilon > 0 $. There is an $ N (=N_{\epsilon}) $ such that $ a_N, a_{N+1}, \ldots $ are in $ (a_N - \epsilon, a_N + \epsilon) $. So $ \alpha_N, \alpha_{N+1}, \ldots $ are in $ [ a_N - \epsilon, a_N + \epsilon ] $, and hence so is $ \alpha $. 

Finally $ a_N, a_{N+1}, \ldots $ and $ \alpha $ are all in $ [ a_N - \epsilon, a_N + \epsilon ] $, ensuring each $ \vert a_N - \alpha \vert, \vert a_{N+1} - \alpha \vert, \ldots $ is $ \leq 2 \epsilon $. 
