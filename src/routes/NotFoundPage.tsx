import { Helmet } from 'react-helmet-async';
import Button from '@/components/Button';

function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-24 text-center">
      <Helmet>
        <title>404 | Quantum Machine Learning Atlas</title>
      </Helmet>
      <p className="text-sm font-semibold uppercase tracking-wide text-primary-600">404</p>
      <h1 className="text-3xl font-semibold text-slate-900 dark:text-slate-100">We misplaced that quantum state.</h1>
      <p className="max-w-md text-sm text-slate-600 dark:text-slate-300">
        The page you are looking for might have decohered. Head back to the homepage to continue exploring QML resources.
      </p>
      <Button as="a" href="/">
        Return home
      </Button>
    </div>
  );
}

export default NotFoundPage;
