import { Outlet, useLocation } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { Fragment, useEffect } from 'react';

function RootLayout() {
  const location = useLocation();

  useEffect(() => {
    const main = document.getElementById('main');
    main?.focus();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [location.pathname]);

  return (
    <Fragment>
      <Navbar />
      <main id="main" tabIndex={-1} className="mx-auto flex min-h-screen w-full max-w-7xl flex-col px-6 pb-16 pt-24">
        <Outlet />
      </main>
      <Footer />
    </Fragment>
  );
}

export default RootLayout;
