import Fuse from 'fuse.js';
import type { Algorithm, BlogPost, NewsItem, ResearchPaper } from '@/types/content';

type Searchable =
  | (BlogPost & { type: 'blog' })
  | (NewsItem & { type: 'news' })
  | (ResearchPaper & { type: 'research' })
  | (Algorithm & { type: 'algorithm' });

export function createSearchIndex(
  blogs: BlogPost[],
  news: NewsItem[],
  research: ResearchPaper[],
  algorithms: Algorithm[]
) {
  const records: Searchable[] = [
    ...blogs.map((item) => ({ ...item, type: 'blog' as const })),
    ...news.map((item) => ({ ...item, type: 'news' as const })),
    ...research.map((item) => ({ ...item, type: 'research' as const })),
    ...algorithms.map((item) => ({ ...item, type: 'algorithm' as const }))
  ];

  return new Fuse<Searchable>(records, {
    includeScore: true,
    keys: [
      'title',
      'name',
      'tags',
      'excerpt',
      'summary',
      'authors',
      'venue',
      'category'
    ],
    threshold: 0.35
  });
}

export type SearchResult = Fuse.FuseResult<Searchable>;
export type { Searchable };
