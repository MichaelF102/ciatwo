import type { PropsWithChildren, ReactNode } from 'react';
import { Link } from 'react-router-dom';

type CardProps = PropsWithChildren<{
  to?: string;
  title: ReactNode;
  description?: ReactNode;
  eyebrow?: ReactNode;
  footer?: ReactNode;
  as?: 'article' | 'div';
}>;

function Card({ to, title, description, eyebrow, footer, children, as: Component = 'article' }: CardProps) {
  return (
    <Component
      className={`group relative flex h-full flex-col gap-4 rounded-2xl border border-slate-200/80 bg-white/80 p-6 shadow-md transition hover:-translate-y-1 hover:shadow-lg dark:border-slate-800 dark:bg-slate-900/80 ${
        to ? 'cursor-pointer' : ''
      }`}
    >
      {to ? <Link to={to} className="absolute inset-0 rounded-2xl" aria-hidden="true" tabIndex={-1} /> : null}
      <div className="flex flex-col gap-2">
        {eyebrow ? <div className="text-xs font-semibold uppercase tracking-wide text-primary-600">{eyebrow}</div> : null}
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
        {description ? <p className="text-sm text-slate-600 dark:text-slate-300">{description}</p> : null}
      </div>
      {children}
      {footer ? <div className="mt-auto pt-4 text-xs text-slate-500 dark:text-slate-400">{footer}</div> : null}
    </Component>
  );
}

export default Card;
