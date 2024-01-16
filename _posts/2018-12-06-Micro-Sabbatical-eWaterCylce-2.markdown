---
layout: post
title:  "Using Jupyter to study Earth"
date:   2018-12-06 11:52:15 +0200
author: Susan Branchett
image: 2018-12-10-thelatestfrom-jupiter.jpg
---
**How TU Delft’s ICT-Innovation department is providing hands-on help to researchers in order to understand their IT requirements better.**

# How did it come about?
Earlier this year I was reading through the [‘TU Delft Strategic Framework 2018-2024’](https://d1rkab7tlqy5f1.cloudfront.net/TUDelft/Over_TU_Delft/Strategie/Towards%20a%20new%20strategy/TU%20Delft%20Strategic%20Framework%202018-2024%20%28EN%29.pdf) and buried deep within its pages I found this hidden gem:

> We strengthen the social cohesion and interaction within the organisation, by:
> * Supporting mobility across the campus. For example through interfaculty micro-sabbaticals.
> * Stimulating joint activities and knowledge exchange across the various faculties and service departments.
> * Strengthening relations between academic staff members and support staff.


![Hidden Gem]({{ "/assets/img/2018-12-10-secret-diamond-wedding-band.jpg" | absolute_url }})

This seemed especially relevant to our ICT-Innovation department. We are continually on the look-out for ways to support the primary processes of the university, research and education, by applying IT solutions. I decided to find myself a suitable micro-sabbatical.


Since October 2018 I’ve been spending one day a week in the group of [Prof.dr.ir. Nick van de Giesen](https://www.tudelft.nl/en/staff/n.c.vandegiesen/) and [Dr.ir. Rolf Hut](https://www.tudelft.nl/en/staff/r.w.hut/), working with their bright, new Ph.D. student, Jerom Aerts, on the eWaterCycle II project.

# What’s it about?
eWaterCylce II aims to understand water movement on a global scale in order to predict floods, droughts and the effect of land use on water. You can read more about it here <https://www.ewatercycle.org/> or here <https://www.esciencecenter.nl/project/ewatercycle-ii>.

![Sacramento River Delta]({{ "/assets/img/2018-12-10-Islands_Sacramento_River_Delta_California.jpg" | absolute_url }})

Hydrologists are encouraged to use their own local models within a global hydrological model.

In order to test whether their model is working properly, the project team is developing a Python [Jupyter notebook](https://jupyter.org/) that makes it easy for hydrologists to produce the graphs and statistics that they are familiar with.

During my micro-sabbatical, I am contributing to the development of this [Jupyter notebook](https://github.com/eWaterCycle/hydro-analyses/blob/master/eosc_pilot/forecast_ensemble_analyses.ipynb).

# What did I learn?
* Wi-Fi is an essential service for researchers and needs to be reliable
* Standard TU Delft laptops are not adequate for research
* Data for this project is hosted in Poland due to the collaboration with many partners and funding from [EOSC](https://ec.europa.eu/research/openscience/index.cfm?pg=open-science-cloud)
* The team initially hosted their forecasting site on AWS, because AWS is quick to set up and it works in all the countries involved. For the minimum viable product of the global hydrology model they moved to the [SURFsara HPC Cloud](https://userinfo.surfsara.nl/systems/hpc-cloud)
* If data is not open, then researchers are hesitant to use it. Their work can’t be reproduced easily, leading to fewer quality checks and less publicity
* In the face of bureaucracy, cramped conditions and an ever growing number of extra required activities, our researchers’ determination and passion for their field of expertise is truly magnificent

![TU Delft light bulb]({{ "/assets/img/2018-12-10-TU-Delft-light-bulb.jpg" | absolute_url }})

I shall be using these insights to guide my work within the ICT-Innovation department and to feed our conversations with the Shared Service Center.

# What next?
From 1st April 2019 I’ll be moving on to my next micro-sabbatical at the Chemical Engineering department of the Applied Sciences faculty. There I shall be installing molecular simulation software on a computer cluster and getting it up and running.

My ambition is to cover all 8 faculties of the TU Delft within 4 years. In October 2019 I shall be available for the next micro-sabbatical. If you have any suggestions, please do not hesitate to get in touch.

# About the Author
Susan Branchett is Expert Research Data Innovation in the ICT-Innovation department of the TU Delft. She has a Ph.D. in physics and many years’ experience in software development and IT.
Find her at 
[TU Delft](https://www.tudelft.nl/staff/s.e.branchett/) or
[LinkedIn](https://linkedin.com/in/sebranchett) or
[Twitter](https://twitter.com/sebranchett) or
[github](https://github.com/sebranchett).

This blog expresses the views of the author.

# Acknowledgements
The image of Jupiter is from [here](https://www.jpl.nasa.gov/spaceimages/details.php?id=pia21974). Image credit: NASA/JPL-Caltech/SwRI/MSSS/Kevin M. Gill. [License](https://www.jpl.nasa.gov/imagepolicy/).

The hidden gem image is from [here](https://www.macintyres.co.uk/diamond-fancy-wedding-rings-/8075-18ct-yellow-gold-secret-diamond-wedding-band.html) and is reproduced by kind permission of Macintyres.

The Sacramento River Delta image is from [here](https://commons.wikimedia.org/wiki/File:Islands,_Sacramento_River_Delta,_California.jpg) and is reproduced under a [CC-BY-2.0](https://creativecommons.org/licenses/by/2.0/) license.

Except where otherwise noted this blog is available under a [CC-BY-4.0 international license](https://creativecommons.org/licenses/by/4.0/).
