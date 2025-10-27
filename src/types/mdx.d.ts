declare module '*.mdx' {
  import type { ComponentType } from 'react';
  const MDXComponent: ComponentType<Record<string, unknown>>;
  export const metadata: Record<string, unknown>;
  export default MDXComponent;
}
