---
layout: post
title: "Radius of convergence"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

$ \color{goldenrod}{\text{Th}} $: Consider series $ \sum_{n=1}^{\infty} v_n $, where $ (v_n) $ is a seq in a complete normed space.   
Let $ \ell $ $ := \limsup_{n\to\infty} \|v_n\|^{\frac{1}{n}} \in [0, \infty] $. Now :   
If $ 1 &lt; \ell \leq \infty $, the series diverges.   
If $ \ell &lt; 1 $, the series converges absolutely.   
$ \color{goldenrod}{\text{Pf}} $: If $ \ell = \infty $, seq $ \|v_n\|^{\frac{1}{n}} $ is unbounded. So $ \|v_n \| $ is unbounded, especially $ v_n \nrightarrow 0 $. So $ \sum v_n $ diverges.   
If $ 1 &lt; \ell &lt; \infty $, pick $ \epsilon &gt; 0 $ such that $ 1 &lt; \ell - \epsilon $. As $ \|v_n\|^{\frac{1}{n}} \geq \ell - \epsilon $ $ (&gt; 1 ) $ for infinitely many $ n $, we have $ \|v_n \| &gt; 1 $ for infinitely many $ n $. Especially $ v_n \nrightarrow 0 $, so $ \sum v_n $ diverges.   
If $ \ell &lt; 1 $, pick $ \epsilon &gt; 0 $ such that $ \ell + \epsilon &lt; 1 $. As $ \|v_n \|^{\frac{1}{n}} \leq \ell + \epsilon $ "for all large $ n $" (formally, "for all but finitely many $ n $"), we see $ \|v_n\| \leq (\ell + \epsilon)^n $ for all large $ n $. As $ \sum (\ell + \epsilon)^n $ converges, so does $ \sum \|v_n\| $ (and because of completeness, so does $ \sum v_n $). 

---

> Consider complex power series $ \sum_{n=0}^{\infty} a_n z^n $. Using above result and $ \limsup_{n \to \infty} \|a_n z^n\|^{\frac{1}{n}} = ( \limsup_{n \to \infty} \|a_n\|^{\frac{1}{n}} ) \|z\| $, we get the radius of convergence. 

**Edit** 

Consider ${ \sum _{0} ^{n} a _n z ^n }$ with ${ a _n \in \mathbb{C} }.$ From above "root test", it converges absolutely when ${ \varlimsup (\vert a _n \vert ^{\frac{1}{n}} \vert z \vert) \lt 1 }$ and diverges when ${ 1 \lt \varlimsup (\vert a _n \vert ^{\frac{1}{n}} \vert z \vert) \leq \infty .}$ 

${ \varlimsup (\vert a _n \vert ^{\frac{1}{n}} \vert z \vert) }$ is ${ \varlimsup (\vert a _n \vert ^{\frac{1}{n}} ) \vert z \vert }$ when ${ \varlimsup \vert a _n \vert ^{\frac{1}{n}} \lt \infty,  }$ is ${ 0 }$ when ${ \lbrace \varlimsup \vert a _n \vert ^{\frac{1}{n}} = \infty, z = 0 \rbrace, }$ and is ${ \infty }$ when ${ \lbrace \varlimsup \vert a _n \vert ^{\frac{1}{n}} = \infty, z \neq 0 \rbrace }.$ 

So if ${ \varlimsup \vert a _n \vert ^{\frac{1}{n}} = \infty }$ it converges only at ${ z = 0 }.$ If ${ 0 \lt \varlimsup \vert a _n \vert ^{\frac{1}{n}} \lt \infty },$ it converges absolutely at points ${ \vert z \vert \lt \frac{1}{\varlimsup \vert a _n \vert ^{\frac{1}{n}} } }$ and diverges at points ${ \vert z \vert \gt \frac{1}{\varlimsup \vert a _n \vert ^{\frac{1}{n}} } }.$ If ${ \varlimsup \vert a _n \vert ^{\frac{1}{n}} = 0 }$ it converges absolutely at all points. 

So taking ${ R := \frac{1}{\varlimsup \vert a _n \vert ^{\frac{1}{n}} } \in [0, \infty] }$ (with the conventions ${ \frac{1}{0} = \infty }$ and ${ \frac{1}{\infty} = 0 }$) we see ${ \sum _{0} ^{\infty} a _n z ^n }$ converges absolutely when ${ \vert z \vert \lt R }$ and diverges when ${ \vert z \vert \gt R }.$ 

---

[Limit points of ${ \vert a _n \vert ^{\frac{1}{n}} }$ and ${ \vert \frac{a _{n+1}}{a _n} \vert }$ are related. This helps study radius of convergence] 

Let sequence ${ (x _n) \subseteq \mathbb{R} _{\gt 0}. }$ Now ${ \varliminf \frac{x _{n+1}}{x _n} }$ ${ \leq \varliminf  x _n ^{\frac{1}{n}} }$ ${ \leq \varlimsup  x _n ^{\frac{1}{n}}  }$ ${ \leq \varlimsup \frac{x _{n+1}}{x _n} }.$   
**Pf**: We can first prove ${ \varlimsup  x _n ^{\frac{1}{n}}  }$ ${ \leq \varlimsup \frac{x _{n+1}}{x _n} }.$ If ${ L := \varlimsup \frac{x _{n+1}}{x _n} }$ is ${ \infty }$ its vacuous, so say ${ L \lt \infty }.$ Let ${ \epsilon \gt 0 }.$ There is an ${ N }$ such that ${ \frac{x _{n+1}}{x _n} \leq L + \epsilon }$ for ${ n \geq N }.$ Multiplying inequalities, ${ x _{n} \leq (L +\epsilon) ^{n-N} x _N }$ whenever ${ n \geq N}.$ That is, ${ x _n ^{\frac{1}{n}} \leq (L + \epsilon) ^{1-\frac{N}{n}} x _N ^{\frac{1}{n}} }$ whenever ${ n \geq N }.$ So ${ \varlimsup x _n ^{\frac{1}{n}} \leq \varlimsup (L + \epsilon) ^{1-\frac{N}{n}} x _N ^{\frac{1}{n}} }$ ${ = L + \epsilon }.$ As ${ \epsilon \gt 0 }$ was arbitrary, ${ \varlimsup x _n ^{\frac{1}{n}} \leq L }$ as needed.    
We can similarly prove ${ \varliminf \frac{x _{n+1}}{x _n} \leq \varliminf x _n ^{\frac{1}{n}} }.$ If ${ \ell := \varliminf \frac{x _{n+1}}{x _n}  }$ is ${ 0 }$ its vacuous, so say ${ 0 \lt \ell }.$ Let ${ \epsilon \in (0, \ell) }.$ There is an ${ N }$ such that ${ \ell - \epsilon \leq \frac{x _{n+1}}{x _n} }$ for ${ n \geq N }.$ So ${ (\ell - \epsilon) ^{n - N} x _N \leq x _n }$ whenever ${ n \geq N }.$ That is, ${ (\ell - \epsilon) ^{1- \frac{N}{n} } x _N ^{\frac{1}{n}} \leq x _n ^{\frac{1}{n}} }$ whenever ${ n \geq N }.$ So ${ \ell - \epsilon \leq \varliminf x _n ^{\frac{1}{n}} }.$ As ${ \epsilon \in (0, \ell) }$ was arbitrary, ${ \ell \leq \varliminf x _n ^{\frac{1}{n}} }$ as needed.  




