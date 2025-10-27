# Quantum Machine Learning Atlas

A fast, modern React + Vite + Tailwind experience for exploring Quantum Machine Learning (QML) content. The Atlas curates
blogs, news, research, and algorithm primers tailored to graduate students, quantitative researchers, and engineers who demand
citation-ready material.

## Features

- ⚡️ React 18 + Vite with TypeScript and Tailwind CSS (dark mode ready).
- 🧭 React Router v6 layout with sticky navigation, skip-to-content, and keyboard-friendly interactions.
- 🔎 Client-side fuzzy search (Fuse.js) across blogs, news, research papers, and algorithms.
- 📚 MDX-powered content pipeline for blogs and algorithm deep dives; JSON-driven research and news feeds.
- 🧱 Reusable UI primitives: cards, badges, filters, callouts, pagination, code blocks, and more.
- 🧠 SEO essentials via `react-helmet-async` including OpenGraph metadata and JSON-LD for articles.
- 🚀 Deploy-ready for Netlify, Vercel, or GitHub Pages.

## Getting started

### Prerequisites

- Node.js 18+
- pnpm, npm, or yarn (examples below use `npm`).

### Install dependencies

```bash
npm install
```

### Run the development server

```bash
npm run dev
```

The site will be available at [http://localhost:5173](http://localhost:5173). Hot module replacement is enabled by Vite.

### Lint the project

```bash
npm run lint
```

### Build for production

```bash
npm run build
```

This command runs TypeScript type-checking followed by `vite build`. The output lives in the `dist/` directory and can be
uploaded directly to Netlify, Vercel, GitHub Pages, or any static host.

### Preview the production build

```bash
npm run preview
```

## Project structure

```
├── content/                # MDX + JSON content primitives
│   ├── blog/               # MDX blog posts
│   ├── algorithms/         # MDX algorithm deep dives
│   ├── news/               # JSON news feed entries
│   └── research/           # JSON research library entries
├── public/                 # Static assets
├── src/
│   ├── components/         # Reusable UI components
│   ├── layouts/            # Layout shells
│   ├── providers/          # Theme + search context providers
│   ├── routes/             # Route-level views
│   ├── lib/                # Content + search utilities
│   └── types/              # Shared TypeScript definitions
└── vite.config.ts          # Vite configuration with MDX + alias support
```

## Content authoring

- **Blogs**: add MDX files to `content/blog`. Export a `metadata` object with fields like `title`, `slug`, `date`, `tags`, and
  `readingTime`. Use regular Markdown, React components, or the bundled `CodeBlock` component for syntax highlighting.
- **Algorithms**: MDX files in `content/algorithms` define catalog entries with metadata such as `category`, `difficulty`, and
  `references`.
- **News** and **Research**: append JSON files to `content/news` and `content/research` respectively. Arrays are merged
  automatically at build time.

## Deployment

The project is fully static and can be deployed to any CDN-backed host. Popular options:

- **Netlify**: set build command to `npm run build` and publish directory to `dist`.
- **Vercel**: import the repository, choose the Vite framework preset, and deploy.
- **GitHub Pages**: run `npm run build` and push the `dist` directory to the `gh-pages` branch.

## License

MIT
