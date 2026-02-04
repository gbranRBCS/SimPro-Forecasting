import React, { useEffect, useRef, useState } from 'react';
import { Filter, RefreshCw, Sparkles, ChevronDown } from '../../components/icons';

/**
TOOLBAR COMPONENT
-----------------
The main center for the Dashboard.
Responsibilities:
 - 1. Filtering: Date ranges, revenue thresholds.
 - 2. Actions: Triggering Sync (Update vs Full) and data loading.
 */

interface FilterState {
  fromDate: string; // ISO Date string yyyy-mm-dd
  toDate: string;   // ISO Date string yyyy-mm-dd
  minRevenue: string; // Raw input string (e.g. "1000")
  maxRevenue: string; // Raw input string
  order: 'asc' | 'desc';
  limit: string;    // "All" or a number as string
}

interface ToolbarProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
  /** 'update' = fetch new/modified. 'full' = re-fetch everything. */
  onSync: (mode: 'update' | 'full') => void;
  onLoadJobs: () => void;
  isSyncing: boolean;
  isLoading: boolean;
}

export function Toolbar({
  filters,
  onFilterChange,
  onSync,
  onLoadJobs,
  isSyncing,
  isLoading,
}: ToolbarProps) {
  const isDisabled = isSyncing || isLoading;
  
  // -- Sync Dropdown State --
  const [syncMenuOpen, setSyncMenuOpen] = useState(false);
  const syncControlRef = useRef<HTMLDivElement | null>(null);

  // Handle clicking outside the sync menu to close it
  useEffect(() => {
    if (!syncMenuOpen) return;

    function handleClickOutside(event: MouseEvent) {
      if (!syncControlRef.current) return;
      if (!syncControlRef.current.contains(event.target as Node)) {
        setSyncMenuOpen(false);
      }
    }

    function handleKey(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setSyncMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [syncMenuOpen]);

  const triggerSync = (mode: 'update' | 'full') => {
    setSyncMenuOpen(false);
    onSync(mode);
  };

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
          <div
            ref={syncControlRef}
            className="relative inline-flex rounded-lg shadow-sm"
          >
            <button
              onClick={() => triggerSync('update')}
              disabled={isDisabled}
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
              type="button"
              onClick={() => setSyncMenuOpen((open) => !open)}
              disabled={isDisabled}
              aria-label="Open sync menu"
              className="inline-flex items-center justify-center px-2 bg-blue-600 hover:bg-blue-500 text-white rounded-r-lg border-l border-blue-500/60 focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronDown className="w-4 h-4" />
            </button>
            {syncMenuOpen && !isDisabled && (
              <div className="absolute right-0 top-full mt-2 w-44 rounded-md bg-slate-800 border border-slate-700 shadow-lg ring-1 ring-black/5 focus:outline-none overflow-hidden z-20">
                <button
                  type="button"
                  onClick={() => triggerSync('update')}
                  className="w-full px-4 py-3 text-left hover:bg-slate-700 transition-colors border-b border-slate-700/50"
                  title="Fetches only modified or new jobs since the last sync date."
                >
                  <div className="text-sm font-medium text-slate-100">Update Sync</div>
                  <div className="text-xs text-slate-400 mt-0.5">Faster. Gets recent changes only.</div>
                </button>
                <button
                  type="button"
                  onClick={() => triggerSync('full')}
                  className="w-full px-4 py-3 text-left hover:bg-slate-700 transition-colors"
                  title="Deletes local data and re-downloads everything from SimPRO."
                >
                  <div className="text-sm font-medium text-slate-100">Full Sync</div>
                  <div className="text-xs text-slate-400 mt-0.5">Slower. Refreshes all data.</div>
                </button>
              </div>
            )}
          </div>

          <button
            onClick={onLoadJobs}
            disabled={isDisabled}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Filter className="w-4 h-4" />
            Load Jobs
          </button>
        </div>
      </div>
    </div>
  );
}
