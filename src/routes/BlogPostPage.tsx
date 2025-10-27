import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';
import Tag from '@/components/Tag';
import Callout from '@/components/Callout';
import { useSearch } from '@/providers/SearchProvider';

function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>();
  const { datasets } = useSearch();
  const post = datasets.blogs.find((item) => item.slug === slug);

  const related = useMemo(() => {
    if (!post) return [];
    return datasets.blogs.filter((item) => item.slug !== post.slug && item.tags.some((tag) => post.tags.includes(tag))).slice(0, 3);
  }, [datasets.blogs, post]);

  if (!post) {
    return (
      <div className="space-y-4">
        <Helmet>
          <title>Blog not found | Quantum Machine Learning Atlas</title>
        </Helmet>
        <p className="text-lg font-semibold text-slate-900 dark:text-slate-100">Post not found.</p>
      </div>
    );
  }

  const Content = post.Content;
  const canonical = `https://qml-atlas.example.com/blogs/${post.slug}`;
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: post.title,
    datePublished: post.date,
    author: {
      '@type': 'Person',
      name: post.author.name
    },
    url: canonical,
    keywords: post.tags.join(', ')
  };

  return (
    <article className="space-y-10">
      <Helmet>
        <title>{post.title} | Quantum Machine Learning Atlas</title>
        <link rel="canonical" href={post.canonicalUrl ?? canonical} />
        <meta name="description" content={post.excerpt} />
        <meta property="og:title" content={post.title} />
        <meta property="og:description" content={post.excerpt} />
        <meta property="og:type" content="article" />
        <script type="application/ld+json">{JSON.stringify(jsonLd)}</script>
      </Helmet>

      <header className="space-y-4">
        <Tag>{new Date(post.date).toLocaleDateString()}</Tag>
        <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">{post.title}</h1>
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600 dark:text-slate-300">
          <span>{post.author.name}</span>
          <span aria-hidden="true">•</span>
          <span>{post.author.title}</span>
          <span aria-hidden="true">•</span>
          <span>{post.readingTime}</span>
        </div>
        <div className="flex flex-wrap gap-2">
          {post.tags.map((tag) => (
            <Tag key={tag}>{tag}</Tag>
          ))}
        </div>
      </header>

      <section className="prose prose-slate max-w-none dark:prose-invert">
        <Content />
      </section>

      <Callout title="Key takeaways">
        <ul>
          {post.tags.map((tag) => (
            <li key={tag}>{tag}</li>
          ))}
        </ul>
      </Callout>

      {related.length > 0 ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Related posts</h2>
          <ul className="grid gap-4 sm:grid-cols-2">
            {related.map((item) => (
              <li key={item.slug} className="rounded-2xl border border-slate-200 bg-white/60 p-4 dark:border-slate-800 dark:bg-slate-900/60">
                <Link className="font-semibold text-primary-600 hover:text-primary-500" to={`/blogs/${item.slug}`}>
                  {item.title}
                </Link>
                <p className="text-xs text-slate-500 dark:text-slate-300">{item.excerpt}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </article>
  );
}

export default BlogPostPage;
