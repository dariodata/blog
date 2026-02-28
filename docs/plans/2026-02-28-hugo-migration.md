# Hugo Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the blog from deprecated fastpages (Jekyll) to Hugo with the PaperMod theme, preserving all content and features.

**Architecture:** Initialize Hugo in the existing repo, add PaperMod as a git submodule, convert all Jupyter notebooks to Markdown with nbconvert, update front matter, replace GitHub Actions workflows, then remove all fastpages files.

**Tech Stack:** Hugo (static site generator), PaperMod theme, jupyter nbconvert, GitHub Pages, GitHub Actions

---

## Important Context

- All 5 `_posts/*.md` files are **auto-generated HTML** from fastpages — do NOT use them as source
- Source of truth for all posts is `_notebooks/*.ipynb`
- 7 total notebooks to convert: 5 with matching `_posts/` entries + 2 notebook-only (`btc-fraud-detection`, `covid19-germany-dashboard`)
- Front matter fields to keep per post: `title`, `description`, `categories` (→ `tags`), `image` (→ `cover.image`)
- Date comes from the filename prefix (e.g. `2018-04-01-fitbit_prophet.ipynb` → `date: 2018-04-01`)
- Custom domain: `arcosdiaz.com`
- Google Analytics ID: `G-2Q04SCXNNC`

---

## Task 1: Initialize Hugo Site Structure

**Files:**
- Create: `hugo.yaml`
- Create: `content/` (directory)
- Create: `static/` (directory)
- Create: `themes/` (directory)

**Step 1: Check Hugo is installed**

```bash
hugo version
```
Expected: `hugo v0.xxx` — if not installed: `brew install hugo`

**Step 2: Initialize Hugo in the existing repo**

```bash
hugo new site . --force --format yaml
```
Expected: Creates `archetypes/`, `content/`, `static/`, `themes/`, `hugo.yaml`. The `--force` flag allows running in a non-empty directory.

**Step 3: Delete the generated hugo.toml if it exists**

```bash
rm -f hugo.toml
```

**Step 4: Add PaperMod theme as git submodule**

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive
```
Expected: `themes/PaperMod/` directory created with theme files.

**Step 5: Commit**

```bash
git add hugo.yaml archetypes/ content/ static/ themes/ .gitmodules
git commit -m "feat: initialize Hugo site with PaperMod theme"
```

---

## Task 2: Configure Hugo (`hugo.yaml`)

**Files:**
- Modify: `hugo.yaml`

**Step 1: Replace the generated `hugo.yaml` with full config**

```yaml
baseURL: "https://www.arcosdiaz.com/"
title: "Dario Arcos-Díaz, PhD"
theme: PaperMod
paginate: 15
enableRobotsTXT: true
buildDrafts: false
buildFuture: false

minify:
  disableXML: true
  minifyOutput: true

services:
  googleAnalytics:
    id: G-2Q04SCXNNC

params:
  description: "Data Scientist and Neuroscientist with strong statistical modeling and programming skills at the intersection between machine learning and the natural sciences."
  author: Dario Arcos-Díaz
  defaultTheme: auto
  ShowReadingTime: true
  ShowShareButtons: false
  ShowPostNavLinks: true
  ShowBreadCrumbs: true
  ShowCodeCopyButtons: true
  ShowToc: true
  math: true
  socialIcons:
    - name: twitter
      url: "https://twitter.com/darioarcosdiaz"
    - name: github
      url: "https://github.com/dariodata"
    - name: linkedin
      url: "https://www.linkedin.com/in/arcosdiaz/"

menu:
  main:
    - identifier: about
      name: About
      url: /about/
      weight: 10
    - identifier: tags
      name: Tags
      url: /tags/
      weight: 20

outputs:
  home:
    - HTML
    - RSS
    - JSON

taxonomies:
  tag: tags
```

**Step 2: Verify Hugo builds without errors**

```bash
hugo server --bind 0.0.0.0 -D
```
Expected: Server starts at `http://localhost:1313` — site renders (empty but no errors). Stop with Ctrl+C.

**Step 3: Commit**

```bash
git add hugo.yaml
git commit -m "feat: configure Hugo with PaperMod settings and Google Analytics"
```

---

## Task 3: Convert Jupyter Notebooks to Markdown

**Files:**
- Create: `content/posts/YYYY-MM-DD-<name>.md` (7 files)
- Create: `static/images/<notebook-name>_files/` (image output dirs)

**Step 1: Verify nbconvert is available**

```bash
jupyter nbconvert --version
```
Expected: version string. If not installed: `pip install nbconvert`

**Step 2: Convert all notebooks**

Run each command:
```bash
jupyter nbconvert --to markdown _notebooks/2016-10-15-product-revenue-forecast.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2017-02-06-event-tracker.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2017-02-06-medicare-drug-cost.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2017-10-07-personalized-medicine.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2018-04-01-fitbit_prophet.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2019-12-15-btc-fraud-detection.ipynb --output-dir content/posts/
jupyter nbconvert --to markdown _notebooks/2021-01-01-covid19-germany-dashboard.ipynb --output-dir content/posts/
```
Expected: 7 `.md` files in `content/posts/` plus `*_files/` directories for any notebooks with images.

**Step 3: Move generated image directories to static/**

```bash
mv content/posts/*_files static/images/ 2>/dev/null; echo "done"
```
Expected: Any `*_files/` dirs moved to `static/images/`. If there are no image dirs, that's fine.

**Step 4: Update image paths in converted markdown files**

For each `.md` file in `content/posts/`, replace image references from relative `<name>_files/` to `/images/<name>_files/`. Example — in `2018-04-01-fitbit_prophet.md`, find lines like:
```
![png](2018-04-01-fitbit_prophet_files/output_12_0.png)
```
and change to:
```
![png](/images/2018-04-01-fitbit_prophet_files/output_12_0.png)
```

Run this for all posts:
```bash
for f in content/posts/*.md; do
  base=$(basename "$f" .md)
  sed -i '' "s|${base}_files/|/images/${base}_files/|g" "$f"
done
```

**Step 5: Verify no broken image references remain**

```bash
grep -r "_files/" content/posts/ | grep -v "^Binary"
```
Expected: No output (all paths updated).

**Step 6: Commit**

```bash
git add content/posts/ static/images/
git commit -m "feat: convert Jupyter notebooks to markdown posts"
```

---

## Task 4: Add Front Matter to Converted Posts

**Files:**
- Modify: each `content/posts/*.md`

Nbconvert strips front matter from notebooks. Each file needs Hugo front matter added at the top. Use the metadata from the original notebooks.

**Step 1: Open each notebook in `_notebooks/` to find its metadata**

Look in the notebook's first cell or the notebook metadata for `title`, `description`, `categories`, `image`. These were in the fastpages cells at the top of each notebook.

Check a notebook's metadata:
```bash
python3 -c "import json; nb=json.load(open('_notebooks/2018-04-01-fitbit_prophet.ipynb')); print(nb.get('metadata', {}))"
```

**Step 2: Add front matter to each post**

For each `content/posts/YYYY-MM-DD-<name>.md`, prepend the appropriate front matter block. Use this template (replace values per post):

```yaml
---
title: "Post Title Here"
date: YYYY-MM-DD
description: "Post description here"
tags: [tag1, tag2]
cover:
  image: /images/thumbnail.png
  relative: false
math: true
---
```

**Front matter values per post** (from `_posts/` files and notebook metadata):

| File | Title | Date | Tags | Image |
|------|-------|------|------|-------|
| `2016-10-15-product-revenue-forecast.md` | Simulating the revenue of a product with Monte-Carlo random walks | 2016-10-15 | [forecasting, simulation] | /images/product-revenue_thumb.png |
| `2017-02-06-event-tracker.md` | (check notebook) | 2017-02-06 | (check notebook) | (check notebook) |
| `2017-02-06-medicare-drug-cost.md` | (check notebook) | 2017-02-06 | (check notebook) | (check notebook) |
| `2017-10-07-personalized-medicine.md` | (check notebook) | 2017-10-07 | (check notebook) | (check notebook) |
| `2018-04-01-fitbit_prophet.md` | Fitbit activity and sleep data: a time-series analysis with Generalized Additive Models | 2018-04-01 | [time series, wearables, forecasting] | /images/fitbit_prophet_thumb.png |
| `2019-12-15-btc-fraud-detection.md` | (check notebook) | 2019-12-15 | (check notebook) | (check notebook) |
| `2021-01-01-covid19-germany-dashboard.md` | (check notebook) | 2021-01-01 | (check notebook) | (check notebook) |

To check a notebook's metadata quickly:
```bash
python3 -c "
import json
nb = json.load(open('_notebooks/2019-12-15-btc-fraud-detection.ipynb'))
# Check first cell for front matter / metadata
for cell in nb['cells'][:3]:
    print(cell['source'][:300])
    print('---')
"
```

**Step 3: Verify Hugo lists all posts**

```bash
hugo list all
```
Expected: Table showing all 7 posts with their dates and titles.

**Step 4: Commit**

```bash
git add content/posts/
git commit -m "feat: add Hugo front matter to all posts"
```

---

## Task 5: Migrate About Page and Thumbnail Images

**Files:**
- Create: `content/about.md`
- Modify: `static/images/` (add portfolio photo)

**Step 1: Create the About page**

Create `content/about.md`:
```markdown
---
title: About Me
layout: page
---

Hi! I'm a data scientist / neuroscientist / cat lover interested in the complex
dynamics of the world we live in and how we can use data _and_ science to solve
humanity's big and small challenges. One idea at a time.

Fun fact: I started university when I was 15 years old and got my PhD at 25,
and, while I wouldn't recommend it, this has allowed me to explore many of my
interests throughout my life and to develop a diverse and unique set of skills
You can find out more about my professional background on
[LinkedIn](https://www.linkedin.com/in/arcosdiaz/)

![](/images/portfolio-small.png)
```

**Step 2: Copy thumbnail images to static/images/**

```bash
cp images/*.png static/images/ 2>/dev/null; cp images/*.jpg static/images/ 2>/dev/null; echo "done"
```

**Step 3: Verify images directory**

```bash
ls static/images/
```
Expected: Portfolio photo, post thumbnails listed.

**Step 4: Commit**

```bash
git add content/about.md static/images/
git commit -m "feat: add About page and migrate images to static/"
```

---

## Task 6: Create GitHub Actions Deployment Workflow

**Files:**
- Create: `.github/workflows/deploy.yaml`

**Step 1: Create the deployment workflow**

Create `.github/workflows/deploy.yaml`:
```yaml
name: Deploy Hugo to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: true
          fetch-depth: 0

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true

      - name: Build
        run: hugo --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public
          cname: arcosdiaz.com
```

**Step 2: Commit**

```bash
git add .github/workflows/deploy.yaml
git commit -m "feat: add Hugo GitHub Actions deployment workflow"
```

---

## Task 7: Remove fastpages Files

**Files:**
- Delete: all fastpages-specific directories and files

**Step 1: Remove fastpages directories**

```bash
git rm -r _action_files/ _fastpages_docs/ _includes/ _layouts/ _plugins/ _sass/ _word/
```

**Step 2: Remove fastpages config and tooling files**

```bash
git rm Gemfile Gemfile.lock _config.yml docker-compose.yml Makefile settings.ini index.html
```

**Step 3: Remove old fastpages GitHub Actions workflows**

```bash
git rm .github/workflows/gh-page.yaml \
       .github/workflows/check_cdns.yaml \
       .github/workflows/check_config.yaml \
       .github/workflows/docker-nbdev.yaml \
       .github/workflows/issue_reminder.yaml \
       .github/workflows/setup.yaml \
       .github/workflows/upgrade.yaml \
       .github/workflows/ci.yaml 2>/dev/null; echo "done"
```
Note: some of these files may not exist — that's fine.

**Step 4: Remove source directories (now migrated)**

```bash
git rm -r _posts/ _notebooks/ _pages/
```

**Step 5: Remove old images directory (content moved to static/images/)**

```bash
git rm -r images/
```

**Step 6: Verify Hugo still builds**

```bash
hugo
```
Expected: Builds successfully, output in `public/`.

**Step 7: Commit**

```bash
git commit -m "chore: remove all fastpages files"
```

---

## Task 8: Local Verification

**Step 1: Start Hugo dev server**

```bash
hugo server -D
```
Expected: Server at `http://localhost:1313`

**Step 2: Verify each page manually**

Check in browser:
- [ ] Home page shows all 7 posts listed
- [ ] Each post page renders correctly (text, code blocks, images)
- [ ] Math equations render (KaTeX)
- [ ] About page renders with photo
- [ ] Tags page lists all tags
- [ ] Navigation menu shows About and Tags links
- [ ] Social icons in footer/header

**Step 3: Check for broken images**

```bash
hugo --gc --minify 2>&1 | grep -i "error\|warn"
```
Expected: No errors.

**Step 4: Commit any fixes found during review**

```bash
git add -A
git commit -m "fix: address issues found during local review"
```

---

## Task 9: Push and Verify GitHub Pages Deployment

**Step 1: Push to remote**

```bash
git push origin master
```

**Step 2: Monitor GitHub Actions**

Go to `https://github.com/dariodata/blog/actions` — watch the `Deploy Hugo to GitHub Pages` workflow complete.

**Step 3: Verify live site**

Visit `https://www.arcosdiaz.com` and confirm:
- [ ] Site loads correctly
- [ ] Posts are visible
- [ ] Custom domain works
- [ ] No 404s on navigation

**Step 4: If GitHub Pages source needs updating**

In repo Settings → Pages → Source: set to `gh-pages` branch, root `/`. The `peaceiris/actions-gh-pages` action creates this branch automatically on first deploy.
