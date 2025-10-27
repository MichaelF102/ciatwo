import { useMemo, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import Badge from '@/components/Badge';
import ChipFilter from '@/components/ChipFilter';
import { useSearch } from '@/providers/SearchProvider';

function ResearchPage() {
  const { datasets } = useSearch();
  const [year, setYear] = useState('All');
  const [venue, setVenue] = useState('All');
  const [topic, setTopic] = useState('All');

  const years = useMemo(
    () => Array.from(new Set(datasets.research.map((item) => item.year.toString()))).sort().reverse(),
    [datasets.research]
  );
  const venues = useMemo(() => Array.from(new Set(datasets.research.map((item) => item.venue))).sort(), [datasets.research]);
  const topics = useMemo(() => Array.from(new Set(datasets.research.flatMap((item) => item.tags))).sort(), [datasets.research]);

  const filtered = useMemo(() => {
    return datasets.research.filter((paper) => {
      const matchesYear = year === 'All' || paper.year.toString() === year;
      const matchesVenue = venue === 'All' || paper.venue === venue;
      const matchesTopic = topic === 'All' || paper.tags.includes(topic);
      return matchesYear && matchesVenue && matchesTopic;
    });
  }, [datasets.research, topic, venue, year]);

  return (
    <div className="space-y-10">
      <Helmet>
        <title>Research | Quantum Machine Learning Atlas</title>
        <link rel="canonical" href="https://qml-atlas.example.com/research" />
        <meta name="description" content="Curated Quantum Machine Learning research papers with badges and quick filters." />
      </Helmet>

      <header className="space-y-4">
        <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">Research Library</h1>
        <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-300">
          Annotated bibliography across conference proceedings, arXiv preprints, and benchmark reports. Use the filters to surface
          the most relevant work.
        </p>
      </header>

      <section className="space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Year</p>
            <ChipFilter options={years} active={year} onChange={setYear} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Venue</p>
            <ChipFilter options={venues} active={venue} onChange={setVenue} />
          </div>
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Topic</p>
            <ChipFilter options={topics} active={topic} onChange={setTopic} />
          </div>
        </div>

        <ul className="space-y-4">
          {filtered.map((paper) => (
            <li key={`${paper.title}-${paper.year}`} className="rounded-2xl border border-slate-200 bg-white/80 p-6 shadow-md dark:border-slate-800 dark:bg-slate-900/80">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{paper.title}</h2>
                  <p className="text-xs text-slate-500 dark:text-slate-300">{paper.authors.join(', ')}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Badge>{paper.venue}</Badge>
                  <Badge>{paper.year}</Badge>
                  {paper.badge ? <Badge>{paper.badge}</Badge> : null}
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-300">
                {paper.tags.map((tag) => (
                  <span key={tag} className="rounded-full bg-slate-100 px-2 py-1 dark:bg-slate-800">
                    {tag}
                  </span>
                ))}
              </div>
              <div className="mt-4 text-sm">
                <a className="font-semibold text-primary-600 hover:text-primary-500" href={paper.url} target="_blank" rel="noreferrer">
                  View paper ↗
                </a>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

export default ResearchPage;
