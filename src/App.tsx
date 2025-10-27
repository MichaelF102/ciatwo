import { Route, Routes } from 'react-router-dom';
import RootLayout from '@/layouts/RootLayout';
import HomePage from '@/routes/HomePage';
import BlogsPage from '@/routes/BlogsPage';
import BlogPostPage from '@/routes/BlogPostPage';
import NewsPage from '@/routes/NewsPage';
import ResearchPage from '@/routes/ResearchPage';
import AlgorithmsPage from '@/routes/AlgorithmsPage';
import NotFoundPage from '@/routes/NotFoundPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<RootLayout />}>
        <Route index element={<HomePage />} />
        <Route path="blogs" element={<BlogsPage />} />
        <Route path="blogs/:slug" element={<BlogPostPage />} />
        <Route path="news" element={<NewsPage />} />
        <Route path="research" element={<ResearchPage />} />
        <Route path="algorithms" element={<AlgorithmsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

export default App;
