import { useMemo, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import Card from '@/components/Card';
import ChipFilter from '@/components/ChipFilter';
import { useSearch } from '@/providers/SearchProvider';

function AlgorithmsPage() {
  const { datasets } = useSearch();
  const [category, setCategory] = useState('All');
  const [difficulty, setDifficulty] = useState('All');

  const categories = useMemo(
    () => Array.from(new Set(datasets.algorithms.map((item) => item.category))).sort(),
    [datasets.algorithms]
  );
  const difficulties = useMemo(
    () => Array.from(new Set(datasets.algorithms.map((item) => item.difficulty))).sort(),
    [datasets.algorithms]
  );

  const filtered = useMemo(() => {
    return datasets.algorithms.filter((algorithm) => {
      const matchesCategory = category === 'All' || algorithm.category === category;
      const matchesDifficulty = difficulty === 'All' || algorithm.difficulty === difficulty;
      return matchesCategory && matchesDifficulty;
    });
  }, [category, difficulty, datasets.algorithms]);

  return (
    <div className="space-y-10">
      <Helmet>
        <title>Algorithms | Quantum Machine Learning Atlas</title>
        <link rel="canonical" href="https://qml-atlas.example.com/algorithms" />
        <meta
          name="description"
          content="Catalog of Quantum Machine Learning algorithms with intuition, math snippets, complexity, and references."
        />
      </Helmet>

      <header className="space-y-4">
        <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">Algorithm Catalog</h1>
        <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-300">
          Structured primer for QML algorithm design—compare intuition, complexity, and canonical references. Ideal for quick
          onboarding or cross-team reviews.
        </p>
      </header>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Category</p>
            <ChipFilter options={categories} active={category} onChange={setCategory} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Difficulty</p>
            <ChipFilter options={difficulties} active={difficulty} onChange={setDifficulty} />
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((algorithm) => {
            const Content = algorithm.Content;
            return (
              <Card
                key={algorithm.slug}
                title={<span id={algorithm.slug}>{algorithm.name}</span>}
                description={algorithm.intuition}
                eyebrow={`${algorithm.category} • ${algorithm.difficulty}`}
                footer={
                  <div className="space-y-1 text-xs">
                    <p>Complexity: {algorithm.complexity}</p>
                    <p>Use cases: {algorithm.useCases.join(', ')}</p>
                    <ul className="mt-2 space-y-1">
                      {algorithm.references.map((ref) => (
                        <li key={ref.url}>
                          <a className="text-primary-600 hover:text-primary-500" href={ref.url} target="_blank" rel="noreferrer">
                            {ref.label}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                }
              >
                <div className="prose prose-sm max-w-none text-slate-600 dark:prose-invert dark:text-slate-300">
                  <Content />
                </div>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}

export default AlgorithmsPage;
