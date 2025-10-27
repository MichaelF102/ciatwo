import Button from '@/components/Button';

function Pagination({ page, totalPages, onPageChange }: { page: number; totalPages: number; onPageChange: (page: number) => void }) {
  return (
    <nav className="mt-8 flex items-center justify-between" aria-label="Pagination">
      <Button variant="secondary" disabled={page === 1} onClick={() => onPageChange(Math.max(1, page - 1))}>
        Previous
      </Button>
      <p className="text-sm text-slate-500 dark:text-slate-300">
        Page {page} of {totalPages}
      </p>
      <Button variant="secondary" disabled={page === totalPages} onClick={() => onPageChange(Math.min(totalPages, page + 1))}>
        Next
      </Button>
    </nav>
  );
}

export default Pagination;
