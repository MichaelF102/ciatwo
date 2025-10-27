import { useState } from 'react';
import { Link } from 'react-router-dom';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useSearch } from '@/providers/SearchProvider';
import type { SearchResult } from '@/lib/search';

function resolveHref(item: SearchResult['item']) {
  if (item.type === 'blog') {
    return `/blogs/${item.slug}`;
  }
  if (item.type === 'algorithm') {
    return `/algorithms#${item.slug}`;
  }
  if (item.type === 'news' || item.type === 'research') {
    return item.url;
  }
  return '#';
}

function SearchBar() {
  const { query, setQuery, results } = useSearch();
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div className="relative w-full max-w-sm">
      <label htmlFor="global-search" className="sr-only">
        Search QML resources
      </label>
      <div className="flex items-center rounded-full border border-slate-200 bg-white/80 shadow-sm ring-1 ring-transparent transition focus-within:border-primary-500 focus-within:ring-primary-500 dark:border-slate-700 dark:bg-slate-900/80">
        <MagnifyingGlassIcon className="ml-3 h-5 w-5 text-slate-400" aria-hidden="true" />
        <input
          id="global-search"
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setTimeout(() => setIsFocused(false), 150)}
          placeholder="Search articles, news, algorithms..."
          className="w-full rounded-full border-0 bg-transparent px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-0 dark:text-slate-100"
        />
      </div>
      {isFocused && results.length > 0 && (
        <div className="absolute z-30 mt-2 w-full overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl dark:border-slate-700 dark:bg-slate-900">
          <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
            {results.map((result) => {
              const item = result.item;
              const label = 'title' in item ? (item.title as string) : item.name;
              const href = resolveHref(item);
              const isExternal = href.startsWith('http');

              const Wrapper = isExternal ? 'a' : Link;
              const props = isExternal
                ? { href, target: '_blank', rel: 'noreferrer' }
                : { to: href };

              return (
                <li key={`${item.type}-${label}`} className="group">
                  <Wrapper
                    {...(props as never)}
                    className="flex items-center justify-between gap-4 px-4 py-3 transition hover:bg-primary-50 dark:hover:bg-primary-950/40"
                  >
                    <span className="font-medium text-slate-700 group-hover:text-primary-600 dark:text-slate-200">
                      {label}
                    </span>
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-xs uppercase tracking-wide text-slate-500 group-hover:bg-primary-100 group-hover:text-primary-700 dark:bg-slate-800 dark:text-slate-300 dark:group-hover:bg-primary-900/40">
                      {item.type}
                    </span>
                  </Wrapper>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default SearchBar;
