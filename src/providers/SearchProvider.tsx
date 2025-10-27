import type { PropsWithChildren } from 'react';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import debounce from 'lodash.debounce';
import { createSearchIndex } from '@/lib/search';
import { getAllContent } from '@/lib/content';
import type { Algorithm, BlogPost, NewsItem, ResearchPaper } from '@/types/content';
import type { SearchResult } from '@/lib/search';

const { blogs, news, research, algorithms } = getAllContent();
const fuse = createSearchIndex(blogs, news, research, algorithms);

const SearchContext = createContext<{
  query: string;
  setQuery: (value: string) => void;
  results: SearchResult[];
  datasets: {
    blogs: BlogPost[];
    news: NewsItem[];
    research: ResearchPaper[];
    algorithms: Algorithm[];
  };
} | null>(null);

export function SearchProvider({ children }: PropsWithChildren) {
  const [query, setQueryState] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  const debouncedSearch = useMemo(
    () =>
      debounce((value: string) => {
        const trimmed = value.trim();
        setResults(trimmed ? fuse.search(trimmed).slice(0, 10) : []);
      }, 200),
    []
  );

  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
    };
  }, [debouncedSearch]);

  const setQuery = (value: string) => {
    setQueryState(value);
    debouncedSearch(value);
  };

  const value = useMemo(
    () => ({
      query,
      setQuery,
      results,
      datasets: { blogs, news, research, algorithms }
    }),
    [results, query]
  );

  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
}

export function useSearch() {
  const context = useContext(SearchContext);
  if (!context) {
    throw new Error('useSearch must be used within SearchProvider');
  }
  return context;
}
