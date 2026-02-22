import { ChevronLeft, ChevronRight } from '../../components/icons';

// defines the shape of the pagination state object
// used by the parent component to track current position
export type PaginationState = {
  page: number;
  pageSize: number;
  totalPages: number;
};

interface PaginationProps {
  pagination: PaginationState;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  disabled?: boolean;
}

export function Pagination({
  pagination,
  onPageChange,
  onPageSizeChange,
  disabled,
}: PaginationProps) {
  // destructure for easier access in the template
  const { page, pageSize, totalPages } = pagination;

  // decrements the current page number
  // checks to ensure we do not go below page 1
  function handlePrevious() {
    if (page > 1) {
      const previousPage = page - 1;
      onPageChange(previousPage);
    }
  }

  // increments the current page number
  // checks to ensure total number of pages is not exceeded
  function handleNext() {
    if (page < totalPages) {
      const nextPage = page + 1;
      onPageChange(nextPage);
    }
  }

  // updates the number of items shown per page
  // triggered when the user selects a new value from the dropdown
  function handlePageSizeChange(event: React.ChangeEvent<HTMLSelectElement>) {
    // value from event is string, so str -> num
    const newSize = Number(event.target.value);
    onPageSizeChange(newSize);
  }

  return (
    <div className="flex items-center justify-between gap-4 px-6 py-4 bg-slate-900 border-t border-slate-800 rounded-b-xl">
      
      {/* navigation buttons for moving between pages */}
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
          Page {page} of {totalPages}
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

      {/* dropdown to select how many items are shown per page */}
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
