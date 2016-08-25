---
layout: post
title: "Working With Lagrange"
categories: journal
tags: [documentation,sample]
image:
  feature: room.jpg
  teaser: room-teaser.jpg
  credit:
  creditlink:
---

Lagrange was designed to be a minimalist theme in order for the focus to remain on your content.

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

`layout` specifies which layout to use, `title` is the page or post title, `categories` can be used to better organize your posts, `tags` are used to show related posts, as well as indicate what topics are related in a given post, and `image` specifies which images to use. There are two main types of images that can be used in a given post, the `feature` and the `teaser`, which are typically the same image, except the teaser image is cropped for the home page. You can give credit to images under `credit`, and provide a link if possible underst `creditlink`.

### Links

Links are signified mainly through an underline text-decoration, in order to maximize the perceived affordance of clickability (I originally just wanted to make the links a darker shade of grey).

### Images

Images were designed to be 1024x600 pixels, with teaser images being 1024x380 pixels.
