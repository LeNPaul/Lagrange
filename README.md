# Lagrange

Lagrange is a minimalist Jekyll theme for running a personal blog or site for free through [Github Pages](https://pages.github.com/), or on your own server. Everything that you will ever need to know about this Jekyll theme is included in the README below, which you can also find in [the demo site](https://lenpaul.github.io/Lagrange/).

![alt text](https://cloud.githubusercontent.com/assets/8409329/21747617/7ef0e18e-d53a-11e6-8f90-8bb14b62ba20.jpg "Lagrange Demo Image")

## Table of Contents

1. [Installation](#installation)
  1. [GitHub Pages Installation](#github-pages-installation)
  2. [Local Installation](#local-installation)
2. [Directory Structure](#directory-structure)
3. [Configuration](#configuration)
4. [Features](#features)
5. [Everything Else](#everything-else)

## Installation

### GitHub Pages Installation

To start using Jekyll right away using GitHub Pages, [fork the Lagrange repository on GitHub](https://github.com/LeNPaul/Lagrange/fork). From there, you can rename your repository to 'USERNAME.github.io', where 'USERNAME' is your GitHub username, and edit the `settings.yml` file in the `_data` folder to your liking. Ensure that you have a branch named `gh-pages`. Your website should be ready immediately at 'http://USERNAME.github.io'.

Head over to the `_posts` directory to view all the posts that are currently on the website, and to see examples of what post files generally look like. You can simply just duplicate the template post and start adding your own content.

### Local Installation

For a full local installation of Lagrange, [download your own copy of Lagrange](https://github.com/LeNPaul/Lagrange/archive/gh-pages.zip) and unzip it into it's own directory. From there, open up your favorite command line tool, and enter `jekyll serve`. Your site should be up and running locally at [http://localhost:4000](http://localhost:4000).

## Directory Structure

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

## Configuration

### Site Variables

To change site build settings, edit the `_config.yml` file found in the root of your repository, which you can tweak however you like. More information on configuration settings can be found on [the Jekyll documentation site](https://jekyllrb.com/docs/configuration/).

If you are hosting your site on GitHub Pages, then committing a change to the `_config.yml` file will force a rebuild of your site with Jekyll. Any changes made should be viewable soon after. If you are hosting your site locally, then you must run `jekyll serve` again for the changes to take place.

In the `settings.yml` and `authors.yml` files found in the `_data` folder, you will be able to customize your site settings, such as the title of your site, what shows up in your menu, and social media information. To make author organization easier, especially if you have multiple authors, all author information is stored in the `authors.yml` file.

### Adding Menu Pages

The menu pages are found in the `menu` folder in the root directory, and can be added to your menu in the `settings.yml` file.

### Posts

You will find example posts in your `_posts` directory. Go ahead and edit any post and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention of `YYYY-MM-DD-name-of-post.md` and includes the necessary front matter. Take a look at any sample post to get an idea about how it works. If you already have a website built with Jekyll, simply copy over your posts to migrate to Lagrange.

## Features

## Everything Else

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
