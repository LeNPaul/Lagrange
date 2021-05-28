---
layout: post
title: "Reconstructing Climate data Using Python"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
---

Sea surface temperatures (SST) is studied more often than deep sea temperatures. More recently there has been more interest in deep ocean temperatures, but there has been a limitation due to the lack of measurement in those areas. The objective of this NERTO research is to make a high-resolution reconstruction of deep ocean temperatures of surfaces up to 5,500 meters depth at ¼ degree spatial resolution and 5-day time resolution with 33 layers for 26 years.

The first step taken to model this data is to start with a larger resolution and therefore a smaller amount of data. Consider then computing emperical orthogonal functions (EOFs) for each month from 1950-2003 for 33 depths of the ocean. On a one degree by one degree grid this would mean:

$$
1 ^\circ \times 1^\circ \times 33 \ layers\ = 360 \times 180 \times 33 = 2,138,400/ Bytes/ of/ data/ per/ month$$


If there are 54 years needed to be modeled and there are 4 bytes per datum then:

$$
2MB \times 4B \times 54 \ years \times 12 \ months = 5.5 GB
$$

The actual size of a one month file is actually 5.4 GB. This size of data is easier to work worth to start as my personal laptop can read in this amount of data all at once to compute the EOFs. This is not the  case for a $$1/4^\circ$$ by $$1/4^\circ$$ grid. In that case the data is taken for every 5 days as opposed to every month. The total amount of bytes needed to be read is:


$$1/4^\circ \times 1/4^\circ \times 33 \ layers \  \times 5 \ days$$

$$ = 1442 \times 698 \times 33 = 33,215,028 \ entries$$

there are 8 bytes for each dataum therefore there is:
$$ 8 \times 33,215,028 = 265MB \ per\ entry$$

The actual amount of bytes per entry is 274MB.There are 73 files for each year as each file represents a 5 day block. The total amount of bytes is:
$$ 1901 \ files \times 274MB = 521GB \times \ temperature \ and \ salinity = 521 \times 2 = 3TB$$


![Climatology]({{ site.url }}/assets/css/Clim_for_jan_1950.png)

