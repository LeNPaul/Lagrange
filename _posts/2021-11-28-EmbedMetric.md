---
layout: post
title: "Metric spaces vs subsets of normed spaces"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Let ${ (X, d) }$ be a metric space. There is a distance preserving embedding ${ X \hookrightarrow V }$ into a normed space ${ V .}$ 

Consider ${ \mathcal{B}(X, \mathbb{R}) },$ the space of bounded functions on set ${ X },$ with sup norm. Fix ${ a \in X }.$   
Now the map ${ X \hookrightarrow \mathcal{B}(X, \mathbb{R}) }$ sending ${ x \mapsto \varphi(x) := ( f _x - f _a ) },$ where ${ f _p (q) := d(p,q) },$ will work.   
> Well-defined: Let ${ x \in X }.$ Now ${ f _x - f _a \in \mathcal{B}(X, \mathbb{R}) },$ because ${ \vert f _x (t) - f _a (t) \vert }$ ${ = \vert d(x,t) - d(a,t) \vert }$ ${ \leq d(x,a) }.$ 
> Injective: Say ${ f _{x} - f _a = f _{y} - f _{a} }.$ Now ${ d(x,t) = d(y,t) }$ for all ${ t \in X },$ so ${ d(x,y) = 0 ,}$ ie ${ x=y }.$  
> Distance-Preserving: Let ${ x, y \in X }.$ We need ${ d(x,y) = \lVert \varphi (x) - \varphi(y) \rVert _{\infty} },$ ie ${ d(x,y) = \sup _{t \in X} \vert d(x,t) - d(y,t) \vert }.$ As ${ \vert d(x,t) - d(y,t) \vert \leq d(x,y) }$ and equality holds when ${ t = y },$ done. 

This is called [Kuratowski embedding](https://en.m.wikipedia.org/wiki/Kuratowski_embedding). 

---

Each ${ \varphi(x) }$ is a Lipschitz function on ${ (X,d) },$ as ${ \vert \varphi(x) (t _1) - \varphi(x) (t _2) \vert }$ ${ = \vert (f _x - f _a)(t _1) - (f _x - f _a)(t _2) \vert }$ ${ \leq \vert d(x, t _1) - d(x,t _2) \vert + \vert d(a, t _1) - d(a,t _2) \vert  }$ ${ \leq 2 d(t _1, t _2) }.$ 

So ${ \varphi }$ infact gives a distance preserving embedding ${ X \hookrightarrow \mathcal{BL}(X, \mathbb{R}) },$ into the space of bounded  lipschitz functions on ${ X }$ with sup norm.

