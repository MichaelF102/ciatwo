import type { PropsWithChildren, ReactNode } from 'react';
import { InformationCircleIcon } from '@heroicons/react/24/outline';

function Callout({ title, children }: PropsWithChildren<{ title: ReactNode }>) {
  return (
    <aside className="flex gap-3 rounded-2xl border border-primary-200/70 bg-primary-50/80 p-4 text-sm text-primary-800 dark:border-primary-900/50 dark:bg-primary-900/40 dark:text-primary-100">
      <InformationCircleIcon className="mt-1 h-5 w-5 flex-shrink-0" aria-hidden="true" />
      <div className="flex flex-col gap-1">
        <p className="text-sm font-semibold uppercase tracking-wide">{title}</p>
        <div className="prose prose-sm text-primary-900 dark:prose-invert">{children}</div>
      </div>
    </aside>
  );
}

export default Callout;
