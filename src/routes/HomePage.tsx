import { Helmet } from 'react-helmet-async';
import { Link } from 'react-router-dom';
import Button from '@/components/Button';
import Card from '@/components/Card';
import Tag from '@/components/Tag';
import { useSearch } from '@/providers/SearchProvider';

function HomePage() {
  const { datasets } = useSearch();
  const featuredPosts = datasets.blogs.slice(0, 3);
  const latestNews = datasets.news.slice(0, 3);
  const highlightedAlgorithms = datasets.algorithms.slice(0, 4);

  return (
    <div className="space-y-16">
      <Helmet>
        <title>Quantum Machine Learning Atlas</title>
        <link rel="canonical" href="https://qml-atlas.example.com/" />
        <meta
          name="description"
          content="Explore the Quantum Machine Learning Atlas—curated blogs, research, news, and algorithms for advanced practitioners."
        />
        <meta property="og:title" content="Quantum Machine Learning Atlas" />
        <meta
          property="og:description"
          content="Curated quantum machine learning resources with citation-friendly summaries and algorithm intuition."
        />
        <meta property="og:type" content="website" />
      </Helmet>

      <section className="grid gap-10 lg:grid-cols-2 lg:items-center">
        <div className="space-y-6">
          <Tag>Quantum Machine Learning Hub</Tag>
          <h1 className="text-4xl font-semibold tracking-tight text-slate-900 dark:text-white">
            Fast-track quantum-native models with rigorously curated intelligence.
          </h1>
          <p className="max-w-2xl text-lg text-slate-600 dark:text-slate-300">
            Dive into vetted research briefs, reproducible code snippets, and algorithm primers spanning QSVMs, VQEs, quantum
            kernels, and beyond. Built for researchers who need clarity and citations.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button as="a" href="#start">
              Start with QML
            </Button>
            <Button variant="secondary" as="a" href="/research">
              Explore research library
            </Button>
          </div>
        </div>
        <div className="rounded-3xl border border-slate-200 bg-gradient-to-br from-primary-50 via-white to-slate-50 p-8 shadow-xl dark:border-slate-800 dark:from-slate-900 dark:via-slate-950 dark:to-slate-900">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-300">
            Why QML Atlas
          </h2>
          <ul className="mt-6 space-y-4 text-sm text-slate-600 dark:text-slate-300">
            <li>• Fuzzy search across articles, research, and algorithms.</li>
            <li>• JSON-LD ready article metadata and canonical URLs.</li>
            <li>• Dark mode optimized for night-long literature reviews.</li>
            <li>• Modular content system for MDX and JSON sources.</li>
          </ul>
        </div>
      </section>

      <section id="start" className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Featured briefings</h2>
            <p className="text-sm text-slate-500 dark:text-slate-300">Handpicked reads to anchor your next QML project.</p>
          </div>
          <Link to="/blogs" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
            Browse all blogs →
          </Link>
        </div>
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {featuredPosts.map((post) => (
            <Card
              key={post.slug}
              to={`/blogs/${post.slug}`}
              title={post.title}
              eyebrow={new Date(post.date).toLocaleDateString()}
              description={post.excerpt}
              footer={`${post.author.name} • ${post.readingTime}`}
            />
          ))}
        </div>
      </section>

      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Latest intel</h2>
            <p className="text-sm text-slate-500 dark:text-slate-300">Quantum news and funding updates from the last week.</p>
          </div>
          <Link to="/news" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
            View news feed →
          </Link>
        </div>
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {latestNews.map((item) => (
            <Card
              key={item.title}
              as="article"
              title={item.title}
              description={item.summary}
              eyebrow={`${item.source} • ${new Date(item.publishedAt).toLocaleDateString()}`}
              footer={
                <a className="font-semibold text-primary-600 hover:text-primary-500" href={item.url} target="_blank" rel="noreferrer">
                  Read source ↗
                </a>
              }
            />
          ))}
        </div>
      </section>

      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Algorithm deep dives</h2>
            <p className="text-sm text-slate-500 dark:text-slate-300">Understand intuition, math, and use-cases for core QML techniques.</p>
          </div>
          <Link to="/algorithms" className="text-sm font-semibold text-primary-600 hover:text-primary-500">
            Explore catalog →
          </Link>
        </div>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {highlightedAlgorithms.map((algo) => (
            <Card
              key={algo.slug}
              title={algo.name}
              description={algo.intuition}
              eyebrow={algo.category}
              footer={`Complexity: ${algo.complexity}`}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

export default HomePage;
