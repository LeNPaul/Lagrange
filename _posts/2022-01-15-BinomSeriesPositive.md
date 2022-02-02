---
layout: post
title: "Binomial Series"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---


(Ref: Stromberg's "Classical Real Analysis")

Let ${ \alpha \in \mathbb{R} }.$ The [generalised binomial theorem](https://math.stackexchange.com/questions/135894/generalised-binomial-theorem-intuition) says ${ (1+x) ^{\alpha} }$ ${ = \sum _{0} ^{\infty} \binom{\alpha}{j} x ^j }$ for ${ \vert x \vert \lt 1 }.$   
> Also, radius of convergence of ${ \sum _{1} ^{\infty} \binom{\alpha}{j} x ^j }$ is ${ \infty }$ when ${ \alpha \in \mathbb{Z} _{\geq 0} }$ (because all but finitely many coeffs are ${ 0 }$) and ${ 1 }$ when ${ \alpha \in \mathbb{R} \setminus \mathbb{Z} _{\geq 0} }$ (because ${ \vert \binom{\alpha}{j+1} / \binom{\alpha}{j} \vert }$ ${ = \vert \frac{ \alpha - j }{ j+1 } \vert }$ ${ \to 1 }$ as ${ j \to \infty }$). 

If ${ \alpha \gt 0 },$ we can show ${ \sum _1 ^{\infty} \binom{\alpha}{j} x ^j }$ converges uniformly on ${ \vert x \vert \leq 1 }.$   

Recall (a part of) Kummer's Test   
**Th**: Let sequence ${ (a _n) \subseteq \mathbb{R} _{\gt 0} }.$ Suppose we can find a sequence ${ (b _n) \subseteq \mathbb{R} _{\gt 0} }$ such that ${ \ell }$ ${ := \varliminf \left( \dfrac{a _n b _n - a _{n+1} b _{n+1}}{a _n} \right) }$ ${ \gt 0 }.$ Then ${ \sum _{1} ^{\infty} a _n  }$ converges.   
**Pf**: Pick an ${ \epsilon }$ ${ \gt 0 }$ such that ${ \ell - \epsilon }$ ${ \gt 0 }.$ There is an ${ N _0 }$ such that ${ \frac{a _n b _n - a _{n+1} b _{n+1}}{a _n} }$ ${ \geq \ell - \epsilon }$ ${ (\gt 0) }$ for all ${ n \geq N _0 }.$   
We have ${ a _n (\ell - \epsilon) \leq a _n b _n - a _{n+1} b _{n+1} }$ for all ${ n \geq N _0 }.$ So ${ (\sum _{N _0 \leq n \leq N} a _n ) (\ell - \epsilon)  }$ ${ \leq a _{N _0} b _{N _0} - a _{N+1} b _{N+1} }$ ${ \leq a _{N _0} b _{N _0} }$ for arbitrary ${ N \geq N _0 }.$   
So the partial sums of ${ \sum _{1} ^{\infty} a _n }$ are bounded above, and hence are convergent. 

**Th**: Let ${ \alpha \gt 0 }.$ Then ${ \sum _{1} ^{\infty} \binom{\alpha}{j} x ^j }$ converges absolutely and uniformly on ${ \vert x \vert \leq 1 }.$   
**Pf**: By Weierstrass M-test, it suffices to show ${ \sum _{1} ^{\infty} \vert \binom{\alpha}{j} \vert }$ converges. When ${ \alpha }$ is a positive integer this holds trivially, so suppose ${ \alpha }$ is positive but not an integer.   
Now ${ a _j }$ ${ := \vert \binom{\alpha}{j} \vert }$ ${ \gt 0 }.$ Also ${ \frac{a _{j+1}}{a _j} }$ ${ = \vert \frac{\alpha - j}{j+1} \vert }$ ${ = \frac{j-\alpha}{j+1} }$ for all ${ j \geq \alpha }.$ Now ${ \left( \frac{j a _j - (j+1) a _{j+1} }{a _j} \right) }$ ${ = \alpha }$ ${ (\gt 0) }$  for all ${ j \geq \alpha }.$ So ${ \sum _{1} ^{\infty} a _j }$ converges by Kummer's test. 

**Cor**: Let ${ \alpha \gt 0 }.$ Then ${ (1+x) ^{\alpha} }$ ${ = \sum _{0} ^{\infty} \binom{\alpha}{j} x ^j }$ for ${ \vert x \vert \leq 1 }.$    
> By uniform convergence, the series ${ \sum _{0} ^{\infty} \binom{\alpha}{j} x ^j }$ is continuous on ${ [-1,1] }.$ So ${ (1+1) ^{\alpha} }$ ${ = \lim _{x \to 1 ^{-}} (1+x) ^{\alpha} }$ ${ = \lim _{x \to 1 ^{-}} ( \sum _{0} ^{\infty} \binom{\alpha}{j}x ^j )  }$ ${ = \sum _{0} ^{\infty} \binom{\alpha}{j} },$ and similarly ${ 0 }$ ${ = \sum _{0} ^{\infty} \binom{\alpha}{j} (-1) ^j }.$ 

**Cor** (Finding a sequence of polynomials converging uniformly to ${ \vert x \vert }$ on ${ [-1,1] }$)   
> Idea: Write ${ \vert x \vert }$ ${ = \sqrt{1+(x ^2 -1)} }$ and use binomial expansion. 

Polynomials ${ \sum _{0} ^{n} \binom{1/2}{j} x ^j  }$ converge uniformly to ${ \sqrt{1+x} }$ on ${ [-1,1] }.$   
As ${ {\color{green}{t}} }$ varies in ${ [-1, 1] },$  ${ {\color{purple}{ t ^2 -1  }} \in [-1,0] }.$ So ${ \vert \sqrt{1+{\color{purple}{(t ^2 -1 )}} } - \sum _{0} ^{n} \binom{1/2}{j} {\color{purple}{(t ^2 -1 )}} ^j \vert  }$ ${ \leq \max _{x \in [-1,0]} \vert \sqrt{1+x} - \sum _{0} ^{n} \binom{1/2}{j} x ^j \vert  }$ whenever ${ {\color{green}{t}} \in [-1,1]. }$ Hence ${ \max _{t \in [-1,1]} \left\vert \vert t \vert - \sum _{0} ^{n} \binom{1/2}{j} (t ^2 -1) ^j \right\vert  }$ ${ \leq \max _{x \in [-1,0]} \vert \sqrt{1+x} - \sum _{0} ^{n} \binom{1/2}{j} x ^j \vert  }$ ${ \to 0 }$ as ${ n \to \infty }.$ 



