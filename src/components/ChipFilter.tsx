import classNames from 'classnames';

type ChipFilterProps = {
  options: string[];
  active: string;
  onChange: (value: string) => void;
  allowAll?: boolean;
};

function ChipFilter({ options, active, onChange, allowAll = true }: ChipFilterProps) {
  const items = allowAll ? ['All', ...options] : options;
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onChange(option)}
          className={classNames(
            'rounded-full border px-3 py-1 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500',
            active === option
              ? 'border-primary-500 bg-primary-50 text-primary-700 dark:border-primary-400 dark:bg-primary-900/40 dark:text-primary-200'
              : 'border-slate-200 bg-white text-slate-600 hover:border-primary-500 hover:text-primary-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300'
          )}
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export default ChipFilter;
