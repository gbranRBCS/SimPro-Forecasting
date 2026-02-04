import { ChevronLeft, ChevronRight } from '../../components/icons';

/**
 PAGINATION CONTROL
 ------------------
 A simple pagination bar
 Takes the current state (current page, num of pages)
 and provides callbacks to change the page or the page size.
 */

export type PaginationState = {
  page: number;
  pageSize: number;
  totalPages: number;
};

interface PaginationProps {
  pagination: PaginationState;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  /** Disable controls while data is loading */
  disabled?: boolean;
}

export function Pagination({
  pagination,
  onPageChange,
  onPageSizeChange,
  disabled = false,
}: PaginationProps) {
  const { page, pageSize, totalPages } = pagination;

  // -- Handlers --

  const handlePrevious = () => {
    // Don't go below page 1
    if (page > 1) {
      onPageChange(page - 1);
    }
  };

  const handleNext = () => {
    // Don't go past the last page
    if (page < totalPages) {
      onPageChange(page + 1);
    }
  };

  const handlePageSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    // Convert string value from select to number
    const newSize = Number(e.target.value);
    onPageSizeChange(newSize);
  };

  return (
    <div className="flex items-center justify-between gap-4 px-6 py-4 bg-slate-900 border-t border-slate-800 rounded-b-xl">
      
      {/* Page Navigation Buttons */}
      <div className="flex items-center gap-4 whitespace-nowrap">
        <button
          onClick={handlePrevious}
          disabled={disabled || page <= 1}
          className="inline-flex items-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          aria-label="Previous page"
        >
          <ChevronLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Previous</span>
        </button>

        <span className="text-sm text-slate-300 font-medium">
          Page {page} of {totalPages || 1}
        </span>

        <button
          onClick={handleNext}
          disabled={disabled || page >= totalPages}
          className="inline-flex items-center gap-1 px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          aria-label="Next page"
        >
          <span className="hidden sm:inline">Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* Page Size Selector */}
      <div className="flex items-center gap-2 whitespace-nowrap">
        <label htmlFor="pageSize" className="text-sm text-slate-400">
          Per page:
        </label>
        <select
          id="pageSize"
          value={pageSize}
          onChange={handlePageSizeChange}
          disabled={disabled}
          className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <option value={10}>10</option>
          <option value={20}>20</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </div>
    </div>
  );
}
