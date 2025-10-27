import { useMemo, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import ArticleList from '@/components/ArticleList';
import ChipFilter from '@/components/ChipFilter';
import Pagination from '@/components/Pagination';
import { useSearch } from '@/providers/SearchProvider';

const PAGE_SIZE = 6;

function BlogsPage() {
  const { datasets } = useSearch();
  const [tag, setTag] = useState('All');
  const [year, setYear] = useState('All');
  const [page, setPage] = useState(1);

  const tags = useMemo(() => Array.from(new Set(datasets.blogs.flatMap((item) => item.tags))).sort(), [datasets.blogs]);
  const years = useMemo(
    () => Array.from(new Set(datasets.blogs.map((item) => new Date(item.date).getFullYear().toString()))).sort().reverse(),
    [datasets.blogs]
  );

  const filtered = useMemo(() => {
    return datasets.blogs.filter((post) => {
      const matchesTag = tag === 'All' || post.tags.includes(tag);
      const matchesYear = year === 'All' || new Date(post.date).getFullYear().toString() === year;
      return matchesTag && matchesYear;
    });
  }, [datasets.blogs, tag, year]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div className="space-y-10">
      <Helmet>
        <title>Blogs | Quantum Machine Learning Atlas</title>
        <link rel="canonical" href="https://qml-atlas.example.com/blogs" />
        <meta
          name="description"
          content="Explore curated Quantum Machine Learning blog posts with filters by tag and year, including MDX-ready content."
        />
      </Helmet>

      <header className="space-y-4">
        <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">Blogs & Technical Briefings</h1>
        <p className="max-w-2xl text-sm text-slate-600 dark:text-slate-300">
          Long-form analyses, derivations, and implementation notes designed to be citation-friendly. Filter by topic and year to
          zero in on what matters for your research agenda.
        </p>
      </header>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Filter by tag</p>
            <ChipFilter options={tags} active={tag} onChange={(value) => { setPage(1); setTag(value); }} />
          </div>
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Filter by year</p>
            <ChipFilter options={years} active={year} onChange={(value) => { setPage(1); setYear(value); }} />
          </div>
        </div>
        <ArticleList articles={paginated} />
        <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
      </section>
    </div>
  );
}

export default BlogsPage;
