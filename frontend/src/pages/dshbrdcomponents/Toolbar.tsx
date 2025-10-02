import { Filter, RefreshCw, Sparkles } from 'lucide-react';

interface FilterState {
  fromDate: string;
  toDate: string;
  minRevenue: string;
  maxRevenue: string;
  order: 'asc' | 'desc';
  limit: string;
}

interface ToolbarProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
  onSync: () => void;
  onLoadJobs: () => void;
  onPredict: () => void;
  isSyncing: boolean;
  isLoading: boolean;
}

export function Toolbar({
  filters,
  onFilterChange,
  onSync,
  onLoadJobs,
  onPredict,
  isSyncing,
  isLoading,
}: ToolbarProps) {
  const isDisabled = isSyncing || isLoading;

  const handleChange = (field: keyof FilterState) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    onFilterChange({ ...filters, [field]: e.target.value });
  };

  return (
    <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 px-6 py-4">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-slate-100">
          <Filter className="w-5 h-5 text-blue-500" />
          <h2 className="text-sm font-medium">Filters & Actions</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="fromDate" className="text-xs text-slate-400 font-medium">
              From Date
            </label>
            <input
              id="fromDate"
              type="date"
              value={filters.fromDate}
              onChange={handleChange('fromDate')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="toDate" className="text-xs text-slate-400 font-medium">
              To Date
            </label>
            <input
              id="toDate"
              type="date"
              value={filters.toDate}
              onChange={handleChange('toDate')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="minRevenue" className="text-xs text-slate-400 font-medium">
              Min Revenue
            </label>
            <input
              id="minRevenue"
              type="number"
              placeholder="0"
              value={filters.minRevenue}
              onChange={handleChange('minRevenue')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="maxRevenue" className="text-xs text-slate-400 font-medium">
              Max Revenue
            </label>
            <input
              id="maxRevenue"
              type="number"
              placeholder="∞"
              value={filters.maxRevenue}
              onChange={handleChange('maxRevenue')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="order" className="text-xs text-slate-400 font-medium">
              Order
            </label>
            <select
              id="order"
              value={filters.order}
              onChange={handleChange('order')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <option value="desc">Desc</option>
              <option value="asc">Asc</option>
            </select>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="limit" className="text-xs text-slate-400 font-medium">
              Limit (debug)
            </label>
            <input
              id="limit"
              type="number"
              placeholder="All"
              value={filters.limit}
              onChange={handleChange('limit')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-600/60 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={onSync}
            disabled={isDisabled}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSyncing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Syncing...
              </>
            ) : (
              <>
                <RefreshCw className="w-4 h-4" />
                Sync
              </>
            )}
          </button>

          <button
            onClick={onLoadJobs}
            disabled={isDisabled}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Filter className="w-4 h-4" />
            Load Jobs
          </button>

          <button
            onClick={onPredict}
            disabled={isDisabled}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}
