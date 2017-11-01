# Lagrange

Lagrange is a minimalist Jekyll theme for running a personal blog or site for free through [Github Pages](https://pages.github.com/), or on your own server. Everything that you will ever need to know about this Jekyll theme is included in the README below, which you can also find in [the demo site](https://lenpaul.github.io/Lagrange/).

![alt text](https://cloud.githubusercontent.com/assets/8409329/21747617/7ef0e18e-d53a-11e6-8f90-8bb14b62ba20.jpg "Lagrange Demo Image")

## Table of Contents

1. [Installation](#installation)
  1. [GitHub Pages Installation](#github-pages-installation)
  2. [Local Installation](#local-installation)
  3. [Directory Structure](#directory-structure)
  4. [Starting From Scratch](#starting-from-scratch)
2. [Configuration](#configuration)
  1. [Site Variables](#site-variables)
  2. [Adding Menu Pages](#adding-menu-pages)
  3. [Posts](#posts)
  4. [Layouts](#layouts)
  5. [YAML Front Block Matter](#yaml-front-block-matter)
3. [Features](#features)
  1. [Design Considerations](#design-considerations)
  2. [Disqus](#disqus)
  3. [Google Analytics](#google-analytics)
  4. [RSS Feeds](#rss-feeds)
  5. [Social Media Icons](#social-media-icons)
4. [Everything Else](#everything-else)

## Installation

### GitHub Pages Installation

To start using Jekyll right away using GitHub Pages, [fork the Lagrange repository on GitHub](https://github.com/LeNPaul/Lagrange/fork). From there, you can rename your repository to 'USERNAME.github.io', where 'USERNAME' is your GitHub username, and edit the `settings.yml` file in the `_data` folder to your liking. Ensure that you have a branch named `gh-pages`. Your website should be ready immediately at 'http://USERNAME.github.io'.

Head over to the `_posts` directory to view all the posts that are currently on the website, and to see examples of what post files generally look like. You can simply just duplicate the template post and start adding your own content.

### Local Installation

For a full local installation of Lagrange, [download your own copy of Lagrange](https://github.com/LeNPaul/Lagrange/archive/gh-pages.zip) and unzip it into it's own directory. From there, open up your favorite command line tool, and enter `jekyll serve`. Your site should be up and running locally at [http://localhost:4000](http://localhost:4000).

### Directory Structure

If you are familiar with Jekyll, then the Lagrange directory structure shouldn't be too difficult to navigate. The following some highlights of the differences you might notice between the default directory structure. More information on what these folders and files do can be found in the [Jekyll documentation site](https://jekyllrb.com/docs/structure/).

```bash
Lagrange

├── _data                      # Data files
|  └── authors.yml             # For managing multiple authors
|  └── settings.yml            # Theme settings and custom text
├── _includes                  # Theme includes
├── _layouts                   # Theme layouts (see below for details)
├── _posts                     # Where all your posts will go
├── assets                     # Style sheets and images are found here
|  ├── css
|  |  └── main.css
|  |  └── syntax.css
|  └── img
├── menu                       # Menu pages
├── _config.yml                # Site build settings
└── index.md                   # Home page
```

### Starting From Scratch



## Configuration

### Site Variables

To change site build settings, edit the `_config.yml` file found in the root of your repository, which you can tweak however you like. More information on configuration settings can be found on [the Jekyll documentation site](https://jekyllrb.com/docs/configuration/).

If you are hosting your site on GitHub Pages, then committing a change to the `_config.yml` file will force a rebuild of your site with Jekyll. Any changes made should be viewable soon after. If you are hosting your site locally, then you must run `jekyll serve` again for the changes to take place.

In the `settings.yml` and `authors.yml` files found in the `_data` folder, you will be able to customize your site settings, such as the title of your site, what shows up in your menu, and social media information. To make author organization easier, especially if you have multiple authors, all author information is stored in the `authors.yml` file.

### Adding Menu Pages

The menu pages are found in the `menu` folder in the root directory, and can be added to your menu in the `settings.yml` file.

### Posts

You will find example posts in your `_posts` directory. Go ahead and edit any post and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention of `YYYY-MM-DD-name-of-post.md` and includes the necessary front matter. Take a look at any sample post to get an idea about how it works. If you already have a website built with Jekyll, simply copy over your posts to migrate to Lagrange. Note: Images were designed to be 1024x600 pixels, with teaser images being 1024x380 pixels.

### Layouts

There are two main layout options that are included with Lagrange: post and page. Layouts are specified through the [YAML front block matter](https://jekyllrb.com/docs/frontmatter/). Any file that contains a YAML front block matter will be processed by Jekyll. For example:

```
---
layout: post
title: "Example Post"
---
```

Examples of what posts looks like can be found in the `_posts` directory, which includes this post you are reading right now. Posts are the basic blog post layout, which includes a header image, post content, author name, date published, social media sharing links, and related posts.

Pages are essentially the post layout without and of the extra features of the posts layout. An example of what pages look like can be found at the [About]({{ site.github.url }}/about.html) and [Contacts]({{ site.github.url }}/contacts.html).

In addition to the two main layout options above, there are also custom layouts that have been created for the [home page]({{ site.github.url }}) and the [archives page]({{ site.github.url }}/writing.html). These are simply just page layouts with some [Liquid template code](https://shopify.github.io/liquid/). Check out the `index.html` and `writing.md` files in the root directory for what the code looks like.

### YAML Front Block Matter

The recommended YAML front block is:

```
---
layout:
title:
categories:
tags: []
image:
  feature:
  teaser:
  credit:
  creditlink:

---
```

`layout` specifies which layout to use, `title` is the page or post title, `categories` can be used to better organize your posts, `tags` are used to show related posts, as well as indicate what topics are related in a given post, and `image` specifies which images to use. There are two main types of images that can be used in a given post, the `feature` and the `teaser`, which are typically the same image, except the teaser image is cropped for the home page. You can give credit to images under `credit`, and provide a link if possible under `creditlink`.

## Features

### Design Considerations

Lagrange was designed to be a minimalist theme in order for the focus to remain on your content. For example, links are signified mainly through an underline text-decoration, in order to maximize the perceived affordance of clickability (I originally just wanted to make the links a darker shade of grey).

### Disqus

Lagrange supports comments at the end of posts through [Disqus](https://disqus.com/). In order to activate Disqus commenting, set `disqus.comments` to true in the `settings.yml` file under `_data`. If you do not have a Disqus account already, you will have to set one up, and create a profile for your website. You will be given a `disqus_shortname` that will be used to generate the appropriate comments sections for your site. More information on [how to set up Disqus](http://www.perfectlyrandom.org/2014/06/29/adding-disqus-to-your-jekyll-powered-github-pages/).

### Google Analytics

It is possible to track your site statistics through [Google Analytics](https://www.google.com/analytics/). Similar to Disqus, you will have to create an account for Google Analytics, and enter the correct Google ID for your site under `google-ID` in the `settings.yml` file. More information on [how to set up Google Analytics](https://michaelsoolee.com/google-analytics-jekyll/).

### RSS Feeds

Atom is supported through [Jekyll-Feed](https://github.com/jekyll/jekyll-feed) and RSS 2.0 is supported through [RSS autodiscovery](http://www.rssboard.org/rss-autodiscovery).


### Social Media icons

All social media icons are courtesy of [Font Awesome](http://fontawesome.io/). You can change which icons appear, as well as the account that they link to, in the `settings.yml` file in the `_data` folder.

## Everything Else

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
