import type { PropsWithChildren } from 'react';

function Tag({ children }: PropsWithChildren) {
  return (
    <span className="inline-flex items-center rounded-full bg-primary-50 px-3 py-1 text-xs font-medium text-primary-700 dark:bg-primary-900/40 dark:text-primary-300">
      {children}
    </span>
  );
}

export default Tag;
