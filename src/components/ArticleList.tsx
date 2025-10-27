import { Link } from 'react-router-dom';
import type { BlogPost } from '@/types/content';
import Tag from '@/components/Tag';

function ArticleList({ articles }: { articles: BlogPost[] }) {
  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {articles.map((article) => (
        <article key={article.slug} className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white/80 shadow-md transition hover:-translate-y-1 hover:shadow-lg dark:border-slate-800 dark:bg-slate-900/80">
          <div className="flex flex-col gap-3 p-6">
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-300">
              <span>{new Date(article.date).toLocaleDateString()}</span>
              <span aria-hidden="true">•</span>
              <span>{article.readingTime}</span>
            </div>
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              <Link to={`/blogs/${article.slug}`} className="hover:text-primary-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500">
                {article.title}
              </Link>
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-300">{article.excerpt}</p>
          </div>
          <div className="flex flex-wrap gap-2 px-6 pb-6">
            {article.tags.map((tag) => (
              <Tag key={tag}>{tag}</Tag>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}

export default ArticleList;
