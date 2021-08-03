---
layout: post
title: "Disk containing roots"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Consider $ z^d + a_{d-1} z^{d-1} + \ldots + a_1 z + a_0 $ with each $ a_j \in \mathbb{C} $. Any root $ c \in \mathbb{C} $ must satisfy $ \| c \| \leq 1 + \| a_{d-1} \| + \ldots + \| a_0 \| $ (i.e. must be in the closed disk $ \| z \| \leq 1 + \sum \| a_j \| $)

Proof is easy. Roots with $ \| c \| &lt; 1 $ will obviously satisfy.   
For those with $ \| c \| \geq 1 $ we proceed as :   
$ c^d = -(a_{d-1} c^{d-1} + \ldots + a_0) $, so $ \| c \| = \left| a_{d-1} + \dfrac{a_{d-2}}{c} + \ldots + \dfrac{a_0}{c^{d-1}} \right| $ $ \leq \left| a_{d-1} \right| + \left| \dfrac{a_{d-2}}{c} \right| + \ldots + \left| \dfrac{a_0}{c^{d-1}} \right| $ $ \leq \| a_{d-1} \| + \ldots + \| a_0 \| .$  
