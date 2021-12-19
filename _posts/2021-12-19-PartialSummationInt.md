---
layout: post
title: "Partial Summation"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

**Th**: Consider a sequence ${ (a _n) \subseteq \mathbb{R} }.$   
Let ${ \mathbb{R} _{\gt 0} \overset{f}{\to} \mathbb{R} }$ be a ${ \mathcal{C} ^{1} }$ function, and ${ (x _n) \subseteq \mathbb{R} _{\gt 0} }$ a seq with ${ x _n \nearrow \infty }.$ Using ${ (x _n)}$ for indexing gives functions ${ S(X) := \sum _{x _n \leq X} a _n }$ and ${ S _f (X) := \sum _{x _n \leq X} f(x _n) a _n }$ (for every ${ X },$ there are only finitely many ${ n }$ with ${ x _n \leq X },$ making these sums finite).   
Now ${ S _f (X) = S(X) f(X) - \int _{x _1} ^{X} S(t) f'(t) dt . }$   
**Pf**: When ${ X \lt x _1 }$ both sides are ${ 0 },$ so let ${ x _1 \leq X }.$ We have ${ S _f (X) }$ ${ = \sum _{x _n \leq X} f(x _n) a _n }$ ${ = \sum _{x _n \leq X} \left( f(X) - \int _{x _n} ^{X} f'(t) dt  \right) a _n }$ ${ = S(X) f(X) - {\color{green}{\sum _{x _n \leq X} \int _{x _n} ^{X}  a _n f'(t) dt} } .}$   
Defining ${ g _n (t) }$ to be ${ a _n f '(t) }$ when ${ x _n \leq t }$ and ${ 0 }$ when ${ t \lt x _n },$ we see ${ {\color{green}{\sum _{x _n \leq X} \int _{x _n} ^{X}  a _n f'(t) dt} } }$ ${ = \sum _{x _n \leq X} \int _{x _1} ^{X} g _n (t) dt  }$ ${ = \int _{x _1} ^{X} \left( \sum _{x _n \leq X} g _n (t) \right)dt   }$ ${ = \int _{x _1} ^{X} \left( \sum _{x _n \leq X}  \mathbb{I}[ x _n \leq t ] a _n f'(t) \right) dt }$ ${ = \int _{x _1} ^{X} \left( \sum _{x _n \leq X} \mathbb{I}[x _n \leq t] a _n \right) f'(t) dt }$ ${ = \int _{x _1} ^{X} \left( \sum _{x _n \leq t} a _n \right) f '(t) dt }$ ${ = \int _{x _1} ^{X} S(t) f'(t) dt },$ as needed. 

> ${ x _n = n }$ case is commonly used. 

---

(?) Looks like one could replace ${ x _n \nerightarrow \infty }$ with ${ x _n \to \infty }$ and still make a statement like ${ ${ S _f (X) = S(X) f(X) - \int _{0} ^{X} S(t) f'(t) dt . }$
