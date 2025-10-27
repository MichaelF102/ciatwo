import { MoonIcon, SunIcon } from '@heroicons/react/24/solid';
import { useTheme } from '@/providers/ThemeProvider';

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center justify-center rounded-full border border-slate-300 bg-white p-2 text-slate-600 shadow-sm transition hover:border-primary-500 hover:text-primary-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
      aria-label="Toggle dark mode"
    >
      {theme === 'dark' ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
    </button>
  );
}

export default ThemeToggle;
