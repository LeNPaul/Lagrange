---
layout: post
title: "Some Analysis Proofs"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

[*Existence of nth roots, without intermediate value theorem. From Shilov's "Real and Complex Analysis"*]

**Th**: Let $a \gt 0$ and $n \in \mathbb{Z} _{\gt 0}.$ There exists a unique real $x \gt 0$ with $x ^n = a.$   
**Pf**: [*Uniqueness*] For any reals $x, y \gt 0$ we have ${ x \lt y \iff x ^n \lt y ^n}.$   
> Because ${ (y ^n - x ^n) = (y - x) \underbrace{(y ^{n-1} + y ^{n-2} x + \ldots + y x ^{n-2} + x ^{n-1})} _{\gt 0} }$ 

So two reals ${0 \lt x _1 \lt x _2}$ with ${x _1 ^n = x _2 ^n = a}$ cant exist.
   
[*Existence*] Set ${ S := \lbrace x \in \mathbb{R} _{\geq 0} : x ^n \lt a \rbrace   }$ contains $0,$ and is bounded above.   
> If ${ a \in [1, \infty) }$: For any $x \in S,$  ${ x ^n \lt a \leq a ^n }$ giving ${ x \lt a }.$ So $a$ is an upper bound.   
> If ${ a \in (0,1) }$: For any $x \in S,$ ${ x ^n \leq a \lt 1 ^n }$ giving $x \lt 1.$ So $1$ is an upper bound. 

So take ${ s := \sup(S) }.$  
 
Is $s ^n \lt a$ ? Suppose it were. Now   
${ \begin{aligned} (s+t) ^n &=  s ^n + \binom{n}{1} s ^{n-1} t + \ldots + \binom{n}{n-1} s t ^{n-1} + t ^n \\\\\\ &{\color{green}{\leq}} \text{ } s ^n + t \left( \binom{n}{1} s ^{n-1} + \ldots + \binom{n}{n-1} s + 1  \right) \\\\\\ &= s ^n + t ( (1+s) ^n - s ^n ) \\\\\\ &{\color{purple}{\lt}} \text{ } a, \end{aligned} }$   
for ${ \color{green}{t \in (0,1)} }$ and ${ {\color{purple}{t \lt \frac{a - s ^n}{(1+s) ^n - s ^n} }}. }$   
So $(s+t) ^n \lt a$ for all ${ t \in (0, \min(1, \frac{a - s ^n}{(1+s) ^n - s ^n})  ) }.$ Especially there are points $\gt s$ in $S,$ absurd.

Is $s ^n \gt a$ ? Suppose it were. Now ${ \begin{aligned} (s - t) ^n &= s ^n + \binom{n}{1} s ^{n-1} (-t) + \ldots + \binom{n}{n-1} s (-t) ^{n-1} + (-t) ^n \\\\\\ &{ \color{green}{\geq}} \text{ } s ^n - t \left( \binom{n}{1} s ^{n-1} + \ldots + \binom{n}{n-1} s + 1  \right) \\\\\\ &= s ^n - t((1+s) ^n - s ^n ) \\\\\\ &{\color{purple}{\gt}} \text{ } a,    \end{aligned} }$   
for ${ \color{green}{t \in (0,1)} }$ and ${ {\color{purple}{t \lt \frac{s ^n - a}{(1+s) ^n - s ^n} }} .}$   
So $(s-t) ^n \gt a$ for ${ t \in (0, \min(1, \frac{s ^n - a}{(1+s) ^n - s ^n}) ) }.$ Especially there is a $\delta \gt 0$ such that there is no point of $S$ in $(s - \delta, s],$ absurd. 

So $s ^n = a.$ 

---

*[${ \lim g(f(x)) = g (\lim f(x)) }$]* 

**Th**: Consider ${ S \overset{f}{\to} T \overset{g}{\to} \mathbb{R} }$ with ${ S,T \subseteq \mathbb{R} }.$ Say $a$ is a limit point of $S$ and ${ \lim _{x \to a } f(x) }$ exists. Also suppose ${ \lim _{x \to a} f(x) \in T }$ and $g$ is continuous at $\lim _{x \to a} f(x).$   
Then ${ \lim _{x \to a} g(f(x)) = g( \lim _{x \to a} f(x) ) }.$   
**Pf**: Set ${ \lim _{x \to a} f(x) = \ell }.$ Let ${ \epsilon \gt 0 }.$   
There is a ${ \delta _1 \gt 0 }$ such that points ${ y \in T, \vert y - \ell \vert \lt \delta _1 }$ satisfy ${ \vert g(y) - g(\ell) \vert \lt \epsilon. }$   
There is a ${\delta _2 \gt 0 }$ such that points ${ x \in S \setminus \lbrace a \rbrace, \vert x - a \vert \lt \delta _2 }$ satisfy ${ \vert f(x) - \ell \vert \lt \delta _1.}$   
Now points ${ x \in S \setminus \lbrace a \rbrace, \vert x - a \vert \lt \delta _2 }$ satisfy ${ \vert g(f(x)) - g(\ell) \vert \lt \epsilon }.$   
So ${ \lim _{x \to a} g(f(x)) = g(\ell) },$ as needed. 

---



