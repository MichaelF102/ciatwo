import type { PropsWithChildren } from 'react';

function Badge({ children }: PropsWithChildren) {
  return (
    <span className="inline-flex items-center rounded-md border border-primary-200 bg-primary-50 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary-700 dark:border-primary-900/60 dark:bg-primary-900/40 dark:text-primary-300">
      {children}
    </span>
  );
}

export default Badge;
