# Design: Migrate Blog from fastpages to Hugo

**Date**: 2026-02-28
**Status**: Approved

## Overview

Migrate the blog at `arcosdiaz.com` from the deprecated fastpages (Jekyll-based) framework to Hugo with the PaperMod theme. The blog is hosted on GitHub Pages and contains 5 Markdown posts and 7 Jupyter notebooks. Future posts will be written in Markdown.

## Architecture

- **Framework**: Hugo
- **Theme**: PaperMod (via git submodule)
- **Hosting**: GitHub Pages (same as today)
- **Custom domain**: `arcosdiaz.com` (preserved via `CNAME`)
- **Branch**: Deploy from `master`, GitHub Pages branch managed by Actions

## Content Migration

### Jupyter Notebooks (7 files)
Convert each `.ipynb` in `_notebooks/` to Markdown using `jupyter nbconvert --to markdown`. Place output in `content/posts/`. Images go to `static/images/`.

Notebooks to convert:
- `2016-10-15-product-revenue-forecast.ipynb`
- `2017-02-06-event-tracker.ipynb`
- `2017-02-06-medicare-drug-cost.ipynb`
- `2017-10-07-personalized-medicine.ipynb`
- `2018-04-01-fitbit_prophet.ipynb`
- `2019-12-15-btc-fraud-detection.ipynb`
- `2021-01-01-covid19-germany-dashboard.ipynb`

### Markdown Posts (5 files)
Copy posts from `_posts/` to `content/posts/`, updating front matter:
- Remove Jekyll-specific fields (`layout`, `toc`)
- Keep: `title`, `date`, `tags`, `description`

### Pages
- `_pages/about.md` → `content/about.md`
- Remove Jekyll-specific `{{site.baseurl}}` image reference

### Images
- Move `images/` → `static/images/`
- Update image paths in migrated posts

## Hugo Configuration (`hugo.yaml`)

Key settings to carry over:
- `title`: Dario Arcos-Díaz, PhD
- `baseURL`: https://www.arcosdiaz.com
- Google Analytics: G-2Q04SCXNNC
- Math rendering: KaTeX (via PaperMod config)
- Social links: Twitter, GitHub, LinkedIn
- Tags taxonomy

## GitHub Actions

Replace all existing workflows with a single `deploy.yaml`:
1. Checkout repo (with submodules for theme)
2. Install Hugo
3. Build: `hugo --minify`
4. Deploy to GitHub Pages

**Remove**: All fastpages workflows (`gh-page.yaml`, `check_cdns.yaml`, `check_config.yaml`, `docker-nbdev.yaml`, `issue_reminder.yaml`, `setup.yaml`, `upgrade.yaml`, `ci.yaml`)

## Files to Remove

All fastpages-specific files:
- `_action_files/`, `_fastpages_docs/`, `_layouts/`, `_plugins/`, `_sass/`, `_includes/`
- `Gemfile`, `Gemfile.lock`, `_config.yml`, `docker-compose.yml`, `Makefile`, `settings.ini`
- `index.html` (replaced by Hugo's index)
- `_word/` directory

## Files to Keep

- `_posts/` (source reference during migration, then remove)
- `_notebooks/` (source reference during migration, then remove)
- `images/` (migrated to `static/images/`)
- `CNAME` (if present, or create new one)
- `docs/` directory
