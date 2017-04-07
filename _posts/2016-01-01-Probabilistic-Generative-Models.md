---
layout: post
title: "Probabilistic Generative Models"
categories: bayesian
tags: [bayes]
image:
  feature: bayes.png
  teaser: mountains-teaser.jpg
  credit: Death to Stock Photo
  creditlink: ""
---

Bayesian inference is the coolest thing known to man. Together with generative modeling, it forms the backbone of how I think about applied statistics / machine learning problems. We can use probabilistic generative models for so many things, like modeling language, for example.

### Basics of Generative Models

A generative model consists of two parts: 

1. The data-generating process, also known as the *likelihood*. Parametrized by $\theta$.
2. Since we can never be sure what the parameter values are, we assign them distributions as well.

We can represent all this using **plate notation**:

![Plate notation](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Latent_Dirichlet_allocation.svg/593px-Latent_Dirichlet_allocation.svg.png)



The boxes represent repeated observations, and the circles represent variables/parameters. If a quantity is observed, it is shaded in grey. Otherwise, the quantity is *latent*. For this reason, this type of modeling also goes by the name *latent variable modeling*. Also, some statisticians have also called these types of problems *missing data problems* which confused the heck out of me when I was first learning about these types of models.

### Questions?

This blog is completely free and open source software. You may use it however you want, as it is distributed under the [MIT License](http://choosealicense.com/licenses/mit/). If you are having any problems, any questions or suggestions, feel free to google it.
