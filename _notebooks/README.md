To turn a Jupyter notebook into a blog post, follow https://jaketae.github.io/blog/jupyter-automation/.

The steps are:

1) Run the cells of the notebook and save a checkpoint that includes cell output via
   `python3 -m jupyter lab notebook.ipynb`
2) Convert from .ipynb to .md using this command:
   `jupyter nbconvert --to markdown notebook.ipynb`
3) Manually add [YAML front matter](https://jekyllrb.com/docs/front-matter/).
4) Change image links to follow this format (with spaces next to brackets removed): < img src="/assets/img/some_file_name.png" >
5) Move generated `md` file to `_posts`
6) Move generated images to `assets/img/`
