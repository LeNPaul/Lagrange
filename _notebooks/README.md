To turn a Jupyter notebook into a blog post, follow https://jaketae.github.io/blog/jupyter-automation/.

The steps are:

1) Run the cells of the notebook and save a checkpoint that includes cell output.
2) Convert from .ipynb to .md using this command:
   `jupyter nbconvert --to markdown notebook.ipynb`
3) Manually add [YAML front matter](https://jekyllrb.com/docs/front-matter/).
4) Move generated `md` file to `_posts`
5) Move generated images to `assets/img/`
