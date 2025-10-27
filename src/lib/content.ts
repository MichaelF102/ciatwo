import type { Algorithm, BlogPost, NewsItem, ResearchPaper } from '@/types/content';

const blogModules = import.meta.glob('../../content/blog/*.mdx', { eager: true });
const algorithmModules = import.meta.glob('../../content/algorithms/*.mdx', { eager: true });
const newsModules = import.meta.glob('../../content/news/*.json', { eager: true });
const researchModules = import.meta.glob('../../content/research/*.json', { eager: true });

type BlogModule = {
  default: BlogPost['Content'];
  metadata: Omit<BlogPost, 'Content'>;
};

type AlgorithmModule = {
  default: Algorithm['Content'];
  metadata: Omit<Algorithm, 'Content'>;
};

function unwrapJson<T>(module: unknown): T {
  if (module && typeof module === 'object' && 'default' in (module as Record<string, unknown>)) {
    return (module as { default: T }).default;
  }
  return module as T;
}

export function getBlogs(): BlogPost[] {
  return Object.values(blogModules)
    .map((module) => module as unknown as BlogModule)
    .map((module) => ({
      ...module.metadata,
      Content: module.default
    }))
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getAlgorithms(): Algorithm[] {
  return Object.values(algorithmModules)
    .map((module) => module as unknown as AlgorithmModule)
    .map((module) => ({
      ...module.metadata,
      Content: module.default
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function getNews(): NewsItem[] {
  return Object.values(newsModules)
    .flatMap((module) => unwrapJson<NewsItem[] | NewsItem>(module))
    .flat()
    .sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime());
}

export function getResearch(): ResearchPaper[] {
  return Object.values(researchModules)
    .flatMap((module) => unwrapJson<ResearchPaper[] | ResearchPaper>(module))
    .flat()
    .sort((a, b) => b.year - a.year || a.title.localeCompare(b.title));
}

export function getAllContent() {
  return {
    blogs: getBlogs(),
    news: getNews(),
    research: getResearch(),
    algorithms: getAlgorithms()
  };
}
