import type { ComponentType } from 'react';

export type BlogFrontmatter = {
  title: string;
  slug: string;
  date: string;
  tags: string[];
  excerpt: string;
  coverImage?: string;
  author: {
    name: string;
    title: string;
    avatar?: string;
  };
  readingTime: string;
  canonicalUrl?: string;
};

export type BlogPost = BlogFrontmatter & {
  Content: ComponentType;
};

export type NewsItem = {
  source: string;
  title: string;
  url: string;
  publishedAt: string;
  summary: string;
  tags: string[];
};

export type ResearchPaper = {
  title: string;
  authors: string[];
  venue: string;
  year: number;
  url: string;
  tags: string[];
  badge?: 'SOTA' | 'Survey' | 'Benchmark';
};

export type AlgorithmFrontmatter = {
  name: string;
  slug: string;
  aliases?: string[];
  category: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  references: { label: string; url: string }[];
  complexity: string;
  useCases: string[];
  tags: string[];
  intuition: string;
};

export type Algorithm = AlgorithmFrontmatter & {
  Content: ComponentType;
};

export type SiteContent = {
  blogs: BlogPost[];
  news: NewsItem[];
  research: ResearchPaper[];
  algorithms: Algorithm[];
};
