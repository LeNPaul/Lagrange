---
layout: post
title: "Box with limit points"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

[Minor observation] 
> **Notation**: Let ${ (x _j) _{j \geq 1} }$ be a sequence in a normed space. A sequence ${ (x _j) _{ j \in J} }$ with ${ J \subseteq \mathbb{Z} _{\gt 0} }$  infinite is called a subsequence. We say subsequence ${ (x _j) _{j \in J} }$ converges to ${ p }$ as ${ j \in J, j \to \infty }$ if for every ${ \epsilon \gt 0 },$ ${ x _j \in B _{\epsilon} (p) }$ for all but finitely many $j$ in $J.$  

**Obs**: Let ${ (x _j) _{j \geq 1} }$ be a bounded seq in ${ \mathbb{R} ^n }.$ We can write ${ x _j = (x _{1, j}, \ldots, x _{n, j} ) ^t},$ and take ${ \alpha _k := \limsup _{j \to \infty} x _{k, j} }$ and ${ \beta _k := \liminf _{ j \to \infty} x _{k, j} }.$ Let ${ L }$ be the set of limit points of ${ (x _j) }.$   
Now ${ L \subseteq [\beta _1, \alpha _1] \times \ldots \times [\beta _n, \alpha _n] }.$ Further, each of the ${2n}$ faces ${ \lbrace \beta _1 \rbrace \times [\beta _2, \alpha _2] \times \ldots \times [\beta _n, \alpha _n],  }$ ${ \lbrace \alpha _1 \rbrace \times [\beta _2, \alpha _2] \times \ldots \times [\beta _n, \alpha _n]; }$ ${ \ldots ; }$ ${ [\beta _1, \alpha _1] \times \ldots \times [\beta _{n-1}, \alpha _{n-1}] \times \lbrace \beta _n \rbrace, }$ ${  [\beta _1, \alpha _1] \times \ldots \times [\beta _{n-1}, \alpha _{n-1}] \times \lbrace \alpha _n \rbrace }$ has a point of $L.$   
**Pf**: ${ \color{purple}{\text{(a)}} }$ Let ${ p \in L }.$ There is a subsequence ${ (x _j) _{j \in J} , J \subseteq \mathbb{Z} _{\gt 0} }$ converging to ${ p }$ as ${ j \in J, j \to \infty }.$ So each ${ x _{k, j} }$ converges to ${ p _k }$ as ${ j \in J, j \to \infty }.$ So each ${ p _k \in [\beta _k, \alpha _k] },$ ie ${ p \in [\beta _1, \alpha _1] \times \ldots \times [\beta _n, \alpha _n] }.$   
${ \color{purple}{\text{(b)}} }$ We can show a subsequence converging to a point in ${ \lbrace \beta _1 \rbrace \times [\beta _2, \alpha _2] \times \ldots \times [\beta _n, \alpha _n] },$ the same argument extends to other faces. As ${ \beta _1 }$ is the smallest limit point of ${ (x _{1, j}) _{j \geq 1} },$ there is a subsequence ${ (x _{1, j}) _{j \in J ^{(1)} } , J ^{(1)} \subseteq \mathbb{Z} _{\gt 0} }$ converging to ${ \beta _1 }$ as ${ j \in J ^{(1)}, j \to \infty }.$ Now ${ (x _{2, j}) _{j \in J ^{(1)} } }$ has a convergent subsequence ${ (x _{2, j}) _{j \in J ^{(2)} }, J ^{(2)} \subseteq J ^{(1)} .}$ Now ${ (x _{3, j}) _{j \in J ^{(2)} } }$ has a convergent subsequence ${ (x _{3,j}) _{j \in J ^{(3)} }, J ^{(3)} \subseteq J ^{(2)} }$ and so on.   
Finally we get a convergent subsequence ${ (x _j) _{j \in J ^{(n)} } ,}$ with ${ (x _{1, j}) _{j \in J ^{(n)} } }$ converging to ${ \beta _1 }$ as ${ j \in J ^{(n)}, j \to \infty }.$ Its limit lies in ${ \lbrace \beta _1 \rbrace \times [\beta _2,  \alpha _2] \times \ldots \times [\beta _n, \alpha _n] },$ as needed. 
