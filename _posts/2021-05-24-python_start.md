---
layout: post
title: "Reconstructing Climate data Using Python"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
---

Sea surface temperatures (SST) is studied more often than deep sea temperatures. More recently there has been more interest in deep ocean temperatures, but there has been a limitation due to the lack of measurement in those areas. The objective of this NERTO research is to make a high-resolution reconstruction of deep ocean temperatures of surfaces up to 5,500 meters depth at ¼ degree spatial resolution and 10-day time resolution with 33 layers from year 2000 to present.

The first step taken to model this data is to compute the emperical orthogonal functions for each month from 1950-2003 for all 33 depths of the ocean. On a one  degree by one degree grid this would mean:

$$
1 ^\circ \times 1^\circ \times 33 /text{layers} = 360 \times 180 \times 33 = 2138400$$ Bytes of data per month


If there are 54 years needed to be modeled and there are 4 bytes per datum then:

$$
2MB \times 4B \times 54  years \times 12 months = 5.5 GB
$$

the actual size of a one month file is actually 5.4 GB. This size of data is easier to work worth to start, but unfortunately if the grid is smaller than $$1^\circ$$ then there would be an issue reading in the data to compute EOFs. 

![Climatology]({{ site.url }}/assets/css/Clim_for_jan_1950.png)

