import { Link, NavLink } from 'react-router-dom';
import ThemeToggle from '@/components/ThemeToggle';
import SearchBar from '@/components/SearchBar';

const navItems = [
  { to: '/', label: 'Home' },
  { to: '/blogs', label: 'Blogs' },
  { to: '/news', label: 'News' },
  { to: '/research', label: 'Research' },
  { to: '/algorithms', label: 'Algorithms' }
];

function Navbar() {
  return (
    <header className="fixed inset-x-0 top-0 z-40 border-b border-slate-200/60 bg-white/70 backdrop-blur dark:border-slate-800/60 dark:bg-slate-950/70">
      <div className="mx-auto flex h-20 w-full max-w-7xl items-center justify-between gap-4 px-6">
        <Link to="/" className="flex items-center gap-2 font-mono text-lg font-semibold text-primary-600 dark:text-primary-400">
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-primary-600 text-white shadow-lg dark:bg-primary-500">
            Qµ
          </span>
          <span>QML Atlas</span>
        </Link>
        <nav aria-label="Primary" className="hidden items-center gap-6 text-sm font-medium text-slate-600 lg:flex dark:text-slate-300">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-full px-3 py-2 transition hover:text-primary-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500 dark:hover:text-primary-400 ${isActive ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/40 dark:text-primary-300' : ''}`
              }
              end={item.to === '/'}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="flex items-center gap-4">
          <div className="hidden md:block">
            <SearchBar />
          </div>
          <ThemeToggle />
        </div>
      </div>
      <div className="px-6 pb-4 pt-2 md:hidden">
        <SearchBar />
      </div>
    </header>
  );
}

export default Navbar;
