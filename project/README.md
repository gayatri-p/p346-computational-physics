# Computation Physics Project

To build the project onto github-pages,

```
poetry run jupyter-book build project
poetry run ghp-import -n -p -f project/_build/html
```