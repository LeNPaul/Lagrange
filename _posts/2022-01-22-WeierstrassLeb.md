---
layout: post
title: "Weierstrass Approximation (Lebesgue's Proof)"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Let ${ f \in \mathcal{C}[a,b] }.$   
Weierstrass approximation theorem says there is a sequence of polynomials uniformly converging to ${ f }$ on ${ [a,b] }.$ That is, for every ${ \epsilon \gt 0 }$ there is a polynomial ${ P }$ with ${ \max _{x \in [a,b]} \vert f (x) - P(x) \vert }$ ${ \lt \epsilon }.$ 
  
The proof below is originally due to Lebesgue. Some sources for the proof are books [1, Sec 9.1], [2, Th 5.61] and survey [3, pg 27]. 

---

By uniform continuity, for every ${ \epsilon \gt 0 }$ there is a continuous piecewise-linear map ${ [a,b] \overset{\ell}{\to} \mathbb{R} }$ with ${\max _{x \in [a,b]} \vert f(x) - \ell(x) \vert \lt \epsilon }.$   
> Let ${ \epsilon \gt 0 }.$ There is a ${ \delta \gt 0 }$ such that ${ \vert f(x) - f(y) \vert }$ ${ \lt \epsilon }$ whenever ${ x,y \in [a,b] },$ ${ \vert x - y \vert }$ ${ \lt \delta }.$   
> Pick a partition ${ a = x _0 \lt x _1 \lt \ldots \lt x _{n} = b }$ with ${ \max \vert x _{i+1} - x _{i} \vert }$ ${ \lt \delta }.$ Consider the continuous piecewise-linear map ${ [a,b] \overset{\ell}{\to} \mathbb{R} }$ formed by joining ${ (x _0, f(x _0)), }$ ${ \ldots, }$ ${ (x _n , f(x _n) ) }$ by straight segments.   
> Now for every ${ [x _{i}, x _{i+1}] }$ and intermediate point ${p = x _{i} + \alpha (x _{i+1} - x _{i}) }$ with ${ \alpha \in [0,1] },$ we have ${ \vert f(p) - \ell(p) \vert }$ ${ = \vert f(p) - \lbrace f(x _i) + \alpha (f(x _{i+1}) - f(x _i) ) \rbrace \vert }$ ${ = \vert {\color{purple}{\alpha}} f(p) + {\color{green}{(1-\alpha)}} f(p) - \lbrace {\color{purple}{\alpha}} f(x _{i+1}) + {\color{green}{(1-\alpha)}} f(x _i) \rbrace \vert  }$ ${ \leq \alpha \vert f(p) - f(x _{i+1}) \vert }$ ${ + (1-\alpha) \vert f(p) - f(x _i) \vert }$ ${ \leq \alpha \epsilon + (1-\alpha) \epsilon }$ ${ = \epsilon }.$ 

So it suffices to show the following.

**Th**: If ${ [a,b] \overset{\ell}{\to} \mathbb{R} }$ is a continuous piecewise-linear map, for every ${ \epsilon \gt 0 }$ there is a polynomial ${ P }$ with ${\max _{x \in [a,b]} \vert \ell(x) - P(x) \vert \lt \epsilon }.$   
**Pf**: Say ${ [a,b] \overset{\ell}{\to} \mathbb{R} }$ is continuous, and piecewise-linear w.r.t a partition ${ a = x _0 \lt x _1 \lt \ldots \lt x _{n} = b }.$ Defining ${ \ell(x) }$ to be ${ \ell(x _0) }$ on ${ (-\infty, x _0] }$ and ${ \ell(x _n) }$ on ${ [x _n, \infty) }$ gives a continuous map ${ \mathbb{R} \overset{\ell}{\to} \mathbb{R} }.$   
Call a function ${ \mathbb{R} \overset{f}{\to} \mathbb{R} }$ "nice" if for every ${ M \gt 0 }$ there is a sequence of polynomials converging uniformly to ${ f }$ on ${ [-M, M] }.$ It suffices to show ${ \ell }$ is nice.   
Note the set of all nice functions is closed under addition and multiplication.   
> Say ${ f,g }$ are nice. Then so are ${ f+g, fg }$ because:  Let ${ M \gt 0 }$ be arbitrary. There are sequences ${ (p _n), (q _n) }$ of polynomials such that ${ \lVert p _n - f \rVert }$ ${ \to 0 }$ and ${ \lVert q _n - g \rVert }$ ${ \to 0 },$ where ${ \lVert \ldots \rVert }$ is the sup norm on ${ \mathcal{C}[-M,M] }.$   
> Now even ${ \lVert p _n q _n - fg \rVert }$ ${ = \lVert (\underline{p _n - f} + f)(\underline{q _n - g} + g) - fg \rVert }$ ${ = \lVert (p _n - f)(q _n - g) + (p _n - f)g + f(q _n - g) \rVert }$ ${ \leq \lVert p _n - f \rVert \lVert q _n - g \rVert }$ ${ + \lVert p _n - f \rVert \lVert g \rVert }$ ${ + \lVert f \rVert \lVert q _n - g \rVert }$ ${ \to 0 }$ and ${ \lVert (p _n + q _n) - (f+g) \rVert }$ ${ \leq \lVert p _n - f \rVert }$ ${ + \lVert q _n - g \rVert }$ ${ \to 0 }$ as ${ n \to \infty }.$ 

So it suffices to express ${ \ell }$ as a combination (using ${ +, \times }$) of nice functions. In what follows, ${ m _0 := 0 },$ ${ m _1 := \frac{\ell(x _1) - \ell(x _0)}{x _1 - x _0} },$ ${ \ldots, m _n := \frac{\ell(x _n) - \ell(x _{n-1})}{x _n - x _{n-1}} },$ ${ m _{n+1} := 0 }$ are the slopes of segments of ${ \ell },$ and ${ t ^+ }$ ${ := \frac{1}{2} (t + \vert t \vert) }$ will mean the positive part of ${ t }.$   
The constant function ${ g _0 (t) }$ ${ := \ell(x _0) }$ agrees with ${ \ell(t) }$ on ${ (-\infty, x _0] }.$   
As ${ g _0 (t) }$ is ${ \ell(x _0) }$ on ${ [x _0, x _1] },$ the adjusted function ${ g _1 (t) }$ ${ := g _0 (t) + m _1 (t - x _0) ^+ }$ agrees with ${ \ell(t) }$ on ${ (-\infty, x _0] \cup [x _0, x _1] }$ and is linear (straight) on ${ [x _0, \infty) }.$   
As ${ g _1 (t) }$ is ${ \ell(x _1) + m _1 (t - x _1) }$ on ${ [x _1, x _2] },$ the adjusted function ${ g _2 (t) }$ ${ := g _1 (t) + (m _2 - m _1) (t - x _1) ^+ }$ agrees with ${ \ell(t) }$ on ${ (-\infty, x _1] \cup [x _1, x _2] }$ and is linear on ${ [x _1, \infty) }.$   
Continuing with such adjustments, we get a function ${ g _n (t) }$ ${ := \ell(x _0) }$ ${+ m _1 (t - x _0) ^+ }$ ${ + (m _2 - m _1)(t - x _1) ^+ }$ ${ + \ldots + (m _n - m _{n-1}) (t - x _{n-1}) ^+  }$ which agrees with ${ \ell(t) }$ on ${ (-\infty, x _n] }$ and is linear on ${ [x _{n-1}, \infty) }.$   
Finally as ${ g _n (t) }$ is ${ \ell(x _n) + m _n (t - x _n) }$ on  ${ [x _n, \infty) },$ the adjusted function ${ g _{n+1} (t) }$ ${ := g _n (t) - m _n (t - x _n) ^+ }$ agrees with ${ \ell(t) }$ on ${ \mathbb{R} }.$   
The process is visual: 

![](https://i2.lensdump.com/i/rdchwA.jpg) 

Now ${ \ell(t) }$ ${ = g _{n+1} (t) }$ ${ = \ell(x _0) + \sum _{0} ^{n} (m _{j+1} - m _j) (t - x _j) ^+ }.$ So to show ${ \ell }$ is nice, it suffices to show ${ t \mapsto t ^+ := \frac{1}{2} (t + \vert t \vert) }$ is nice. This is true since ${ \vert t \vert }$ is a nice function.   
> Let ${ M \gt 0 }.$ [We know](https://bvenkatakarthik.github.io/BinomSeriesPositive)  polynomials ${ \sum _{0} ^{n} \binom{1/2}{j} x ^j }$ converge uniformly to ${ \sqrt{1+x} }$ on ${ [-1,1] }.$   
> As ${ {\color{green}{t}} }$ varies in ${ [-M, M] },$ we have ${ \frac{t}{M} \in [-1,1] }$ and ${ {\color{purple}{(\frac{t}{M}) ^2 -1}} }$ ${ \in [ -1, 0] }.$ So ${ \left\vert \sqrt{1+ ({\color{purple}{(\frac{t}{M}) ^2 -1}} ) }- \sum _{0} ^{n} \binom{1/2}{j} ({\color{purple}{(\frac{t}{M}) ^2 -1}} ) ^j \right\vert }$ ${ \leq \max _{x \in [-1,0]} \vert \sqrt{1+x} - \sum _{0} ^{n} \binom{1/2}{j} x ^j \vert }$ whenever ${ {\color{green}{t}} \in [-M, M] }.$   
> That is, ${ \max _{t \in [-M, M]} \left\vert  \frac{\vert t \vert}{M} - \sum _{0} ^{n} \binom{1/2}{j} ((\frac{t}{M}) ^2 -1) ^j \right\vert  }$ ${ \leq \max _{x \in [-1,0]} \vert \sqrt{1+x} - \sum _{0} ^{n} \binom{1/2}{j} x ^j \vert },$ and hence ${ \max _{t \in [-M, M]} \left\vert  \vert t \vert  - M \sum _{0} ^{n} \binom{1/2}{j} ((\frac{t}{M}) ^2 -1) ^j \right\vert  }$ ${  \leq M \max _{x \in [-1,0]} \vert \sqrt{1+x} - \sum _{0} ^{n} \binom{1/2}{j} x ^j \vert }$ ${ \to 0 }$ as ${ n \to \infty }.$

---

**Refs**: 
[1] M. Haase, *Functional analysis : An elementary introduction* 
[2] J. C. Burkill, *A second course in Mathematical Analysis*   
[3] A. Pinkus, Weierstrass and Approximation Theory, *J. Approx. Theory* 




 



