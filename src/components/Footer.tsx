function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white/80 py-10 dark:border-slate-800 dark:bg-slate-950/80">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 text-sm text-slate-600 dark:text-slate-400 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-semibold text-slate-900 dark:text-slate-100">Quantum Machine Learning Atlas</p>
          <p className="mt-1 max-w-lg text-xs">
            Curated knowledge base for quantum-native machine learning research. Built for graduate students, quants, and
            practitioners who need citation-ready resources.
          </p>
        </div>
        <div className="flex flex-col gap-4 md:items-end">
          <div className="flex gap-4 text-xs">
            <a className="hover:text-primary-600" href="https://github.com" target="_blank" rel="noreferrer">
              GitHub
            </a>
            <a className="hover:text-primary-600" href="https://twitter.com" target="_blank" rel="noreferrer">
              X/Twitter
            </a>
            <a className="hover:text-primary-600" href="https://www.linkedin.com" target="_blank" rel="noreferrer">
              LinkedIn
            </a>
          </div>
          <form className="flex w-full max-w-xs gap-2">
            <label htmlFor="newsletter" className="sr-only">
              Join the newsletter
            </label>
            <input
              id="newsletter"
              type="email"
              className="flex-1 rounded-full border border-slate-300 bg-white px-3 py-2 text-xs text-slate-900 placeholder:text-slate-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
              placeholder="your.email@lab.org"
            />
            <button
              type="submit"
              className="rounded-full bg-primary-600 px-4 py-2 text-xs font-semibold text-white shadow-md transition hover:bg-primary-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-500"
            >
              Subscribe
            </button>
          </form>
          <p className="text-xs text-slate-400">
            © {new Date().getFullYear()} QML Atlas. Designed for reproducible research.
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
