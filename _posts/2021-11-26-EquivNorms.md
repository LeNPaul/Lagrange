---
layout: post
title: "Equivalent Norms"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

[Ref: [S. G. Johnson's proof](https://math.mit.edu/~stevenj/18.335/norm-equivalence.pdf)] 

Let ${ V }$ be an ${ \mathbb{R}- }$vector space with basis ${ \mathcal{B} = (v _1, \ldots, v _n) },$ and let ${ \lVert \ldots \rVert _{1}, \lVert \ldots \rVert _{2} }$ be two norms on it. Turns out these norms are equivalent. 

If ${ \mathscr{V} \overset{T}{\to} \mathscr{W} }$ is an isomorphism of ${\mathbb{R}-}$vector spaces and ${ \lVert \ldots \rVert _{\mathscr{W}} }$ is a norm on ${ \mathscr{W} },$ notice ${ \lVert v \rVert _{\mathscr{V}} := \lVert T(v) \rVert _{\mathscr{W}} }$ is a norm on ${ \mathscr{V} }.$   
So via the isomorphism ${ \mathbb{R} ^n \to V }$ sending ${ x = (x _1, \ldots, x _n) ^{t} \mapsto \mathcal{B}x = \sum _{1} ^{n} x _j v _j},$ we get norms ${ \vert x \vert _{1} := \lVert \mathcal{B} x \rVert _{1} }$ and ${ \vert x \vert _{2} := \lVert \mathcal{B}x \rVert _{2} }$ on ${ \mathbb{R} ^n }.$ 

We want constants ${ L, M \gt 0 , }$ such that ${ L \lVert v \rVert _{1} \leq \lVert v \rVert _{2} \leq M \lVert v \rVert _{1} }$ for all ${ v \in V  },$ ie such that ${ L \lVert \mathcal{B}x \rVert _{1} \leq \lVert \mathcal{B}x \rVert _{2} \leq M \lVert \mathcal{B}x \rVert _{1} }$ for all ${ x \in \mathbb{R} ^n },$ ie such that ${ L \vert x \vert _{1} \leq \vert x \vert _{2} \leq M \vert x \vert _{1} }$ for all ${ x \in \mathbb{R} ^n }.$
   
So it suffices to show norms ${ \vert \ldots \vert _{1}, \vert \ldots \vert _{2} }$ are equivalent on ${ \mathbb{R} ^n }.$ It suffices to show every norm ${ \lVert \ldots \rVert }$ on ${ \mathbb{R} ^n }$ is equivalent to Euclidean norm ${ \lVert x \lVert _{u} := \sqrt{\sum _{1} ^{n} x _j ^{2} } }.$   

---

Let ${ \lVert \ldots \rVert }$ be a norm on ${ \mathbb{R} ^n }.$   
We want constants ${ L', M' \gt 0, }$ such that ${ L'  \lVert x \rVert _{u} \leq \lVert x \rVert \leq M' \lVert x \rVert _{u} }$ for all ${ x \in \mathbb{R} ^n },$ ie such that ${ L' \leq \left\lVert \frac{x}{\lVert x \rVert _{u}} \right\rVert \leq M' }$ for all nonzero ${ x \in \mathbb{R} ^n },$ ie such that ${ L' \leq \lVert y \rVert \leq M' }$ for all ${ y \in S _{u} := \lbrace p \in \mathbb{R} ^n : \lVert p \rVert _{u} = 1 \rbrace.  }$ 

So we can study ${ y \mapsto \lVert y \rVert }$ as a map  ${ (S _{u}, \lVert .. \rVert _{u}) \overset{\varphi}{\to} \mathbb{R} }.$   
We already have an ${ M' \gt 0 }$ from direct considerations : ${ \lVert x \rVert }$ ${ \leq \sum _{1} ^{n} \vert x _j \vert \lVert e _j \rVert }$ ${ \leq \sqrt{\sum _{1} ^{n} \vert x _j \vert ^2}  {\sqrt{\sum _1 ^n \lVert e _j \rVert ^2} } }$ ${ = \lVert x \rVert _{u} M' }$ for all ${ x \in \mathbb{R} ^n }.$   
This ensures ${ \varphi }$ is Lipschitz  continuous : ${ \vert \varphi(x) - \varphi(y) \vert }$ ${ = \vert \lVert x \rVert - \lVert y \rVert \vert }$ ${ \leq \lVert x - y \rVert }$ ${ \leq M' \lVert x - y \rVert _{u} .}$   
As ${ \varphi }$ is continuous on a compact set, let ${ \varphi }$ attain its maximum and minimum at ${ y _{\text{max}} , y _{\text{min}} \in S _{u} }$ respectively. Now ${ 0 \lt \lVert y _{\text{min}} \rVert \leq \lVert y \rVert  \leq \lVert y _{\text{max}} \rVert }$ for all ${ y \in S _{u} },$ as needed. 

