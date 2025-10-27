import type { AnchorHTMLAttributes, ButtonHTMLAttributes, PropsWithChildren } from 'react';
import classNames from 'classnames';

type ButtonVariants = 'primary' | 'secondary' | 'ghost';

type ButtonProps = PropsWithChildren<
  (
    | ({ as?: 'button' } & ButtonHTMLAttributes<HTMLButtonElement>)
    | ({ as: 'a' } & AnchorHTMLAttributes<HTMLAnchorElement>)
  ) & { variant?: ButtonVariants }
>;

function Button({ children, className, variant = 'primary', as = 'button', ...props }: ButtonProps) {
  const Component = as;
  return (
    <Component
      {...(props as never)}
      className={classNames(
        'inline-flex items-center justify-center rounded-full px-5 py-2 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500 disabled:cursor-not-allowed disabled:opacity-60',
        {
          primary: 'bg-primary-600 text-white shadow-md hover:bg-primary-500',
          secondary:
            'border border-slate-200 bg-white text-slate-700 shadow-sm hover:border-primary-500 hover:text-primary-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200',
          ghost: 'text-primary-600 hover:text-primary-500 dark:text-primary-300'
        }[variant],
        className
      )}
    >
      {children}
    </Component>
  );
}

export default Button;
