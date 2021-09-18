---
layout: post
title: "[MSE] Why Normal Subgroups ?"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

[Reposted from Math.SE](https://math.stackexchange.com/a/4193143/303300)

Let $G$ be a group. We can ask ourselves :

$\color{goldenrod}{\mathbf{Q}}$) In how many ways can we put an equivalence relation $\sim$ on $G$ (ie partition set $G$), and put a binary operation $\ast$ on set $G/{\sim}$, such that "Equations in $G$ give corresponding equations between equivalence classes in $G/{\sim}$" that is "$ab=c$ in $G$ implies $[a]\ast [b] = [c]$ in $G/{\sim}$" ? 

[Under such $\sim, \ast$, notice $(G/{\sim}, \ast)$ automatically becomes a group] 

$\underline{\textbf{Part-1}}$ (Looking at the potential candidates for $ \sim, \ast$) 
 
Let $\sim, \ast$ be as needed.

Unwrapping the constraints one by one,
$$(1) \sim \text{ is an equivalence relation } $$ and $$ (2) \text{ } [a] = [a'],  [b] = [b'] \implies [a]\ast [b] = [a'] \ast [b']   $$ and $$(3) \text{ } [a]\ast [b] = [ab].$$

Using (3), we see (2) modifies as 
$\text{ } a \sim a', b \sim b' \implies ab \sim a' b' .$

Notice $a \sim b \iff b^{-1} a \in [e_G]$   
($\implies$: As $a \sim b$ and $b^{-1} \sim b ^{-1}$, we have $b^{-1} a \sim b ^{-1} b = e_G$.   
$\impliedby$: As $b^{-1} a \sim e_G$ and $b\sim b$, we have $b(b^{-1} a) \sim b$ ie $a \sim b$ )

Using this, constraint (1) can be rewritten as {$a^{-1} a \in [e_G]$; $b^{-1} a \in [e_G]$ implies $a^{-1}b \in [e_G]$; $b^{-1}a , c ^{-1}b \in [e_G]$ implies $c^{-1} a \in [e_G]$}, which just means "$[e_G]$ is a subgroup of $G$".   
Similarly constraint (2) becomes "$(a')^{-1} a, (b')^{-1} b \in [e_G]$ implies $(b')^{-1} (a')^{-1} a b \in [e_G]$", that is "$x, y \in [e_G]$ implies $(b')^{-1} (a')^{-1} (a' x) (b' y) \in [e_G]$", that is "$t \in [e_G]$ implies $g^{-1} tg \in [e_G]$".   
Constraint (3) remains the same. 

So to summarize, in any such $\sim, \ast$, we have : 
  
1) $ a \sim b \iff b^{-1} a \in H $, where $ H \subseteq G $ is a subgroup satisfying $ g^{-1} H g \subseteq H $ for all $ g \in G $

2) [Above condition gives $[a] = aH$] The operation $ * $ satisfies $(aH) \ast (bH) = (abH)$ 

$\underline{\textbf{Part-2}}$ (That all such $ \sim, \ast $ work) 

Let $ H \subseteq G $ be a subgroup with $ g^{-1} H g \subseteq H $ for all $ g \in G $. We can readily verify $ a \sim b \overset{\text{def}}{\iff} b^{-1} a \in H $ and $ (aH)\ast (bH) \overset{\text{def}}{=} (abH) $ satisfy the constraints in question. 

---

>To summarise the entire discussion, equivalence relations $ a \sim b \iff b^{-1} a \in H $, arising from subgroups $ H $ satisfying $ g^{-1} H g \subseteq H$ for all $ g \in G$, are precisely the ones we were looking for.  The subgroups here are traditionally called "Normal subgroups". 




