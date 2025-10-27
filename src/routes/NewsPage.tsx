import { useMemo, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import Card from '@/components/Card';
import ChipFilter from '@/components/ChipFilter';
import { useSearch } from '@/providers/SearchProvider';

function NewsPage() {
  const { datasets } = useSearch();
  const [source, setSource] = useState('All');

  const sources = useMemo(() => Array.from(new Set(datasets.news.map((item) => item.source))).sort(), [datasets.news]);

  const filtered = useMemo(() => {
    return datasets.news.filter((item) => (source === 'All' ? true : item.source === source));
  }, [datasets.news, source]);

  return (
    <div className="space-y-10">
      <Helmet>
        <title>News | Quantum Machine Learning Atlas</title>
        <link rel="canonical" href="https://qml-atlas.example.com/news" />
        <meta
          name="description"
          content="Stay current with curated Quantum Machine Learning news, funding updates, and product releases."
        />
      </Helmet>

      <header className="space-y-4">
        <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">Quantum ML Newswire</h1>
        <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-300">
          Aggregated feed from arXiv, industry labs, and standards bodies. Filters let you focus on the signals that matter.
        </p>
      </header>

      <section className="space-y-6">
        <ChipFilter options={sources} active={source} onChange={setSource} />
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((item) => (
            <Card
              key={`${item.source}-${item.title}`}
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
    </div>
  );
}

export default NewsPage;
