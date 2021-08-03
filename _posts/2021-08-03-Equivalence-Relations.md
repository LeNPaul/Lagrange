---
layout: post
title: "Equivalence Relations"
author: "Karthik"
categories: journal
tags: [documentation,sample]
---

Let $S$ be a set. Recall a partition of $S$ is a collection of non-empty pairwise disjoint subsets with union $S$. 

Any partition $P$ of $S$ gives a relation $a\sim_P b \overset{\text{def}}{\iff} (a,b \text{ are in same element of }P).$

One naturally wonders what $$\lbrace\sim_P \, : P\text{ a partition of }S\rbrace$$ is. Some thought reveals it is $$\lbrace\text{reflexive, symmetric, transitive relations on } S\rbrace$$.   
> $\subseteq \,$: Clear.   
> $\supseteq \,$: Let $\sim$ be refl symm trans relation on $S$. Sets $[a]:=\lbrace x \in S \, : \, x \sim a\rbrace$ form a partition $\mathscr{P}$ of $S$ and $\sim \, = \, \sim_{\mathscr{P}}.$ 

Reflexive symmetric transitive relations are traditionally called "Equivalence relations". 
