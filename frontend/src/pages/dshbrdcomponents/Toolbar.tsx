import React, { useEffect, useRef, useState } from 'react';
import { Filter, RefreshCw, ChevronDown } from '../../components/icons';

// the main toolbar at the top of the dashboard
// handles filtering, syncing and loading data

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
  // 'update' fetches only new changes, 'full' deletes and refetches everything
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
  const [syncMenuOpen, setSyncMenuOpen] = useState(false);
  const syncControlRef = useRef<HTMLDivElement | null>(null);

  const isDisabled = isSyncing || isLoading;

  // this effect handles closing the sync menu when we click outside of it
  // it listens for clicks on the whole document
  useEffect(() => {
    if (!syncMenuOpen) return;

    function handleClickOutside(event: MouseEvent) {
      // if ref and the clicked element is not inside it
      if (syncControlRef.current && !syncControlRef.current.contains(event.target as Node)) {
        setSyncMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    
    // cleanup function to remove the listener when the menu closes
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [syncMenuOpen]);

  // helper to trigger a sync and close the menu
  function handleSyncClick(mode: 'update' | 'full') {
    setSyncMenuOpen(false);
    onSync(mode);
  }

  // helper to update a single filter field
  function updateFilter(field: keyof FilterState, value: string) {
    onFilterChange({
      ...filters,
      [field]: value
    });
  }

  return (
    <div className="sticky top-0 z-10 bg-slate-900/95 border-b border-slate-800 px-6 py-4">
      <div className="flex flex-col gap-4">
        
        {/* header section */}
        <div className="flex items-center gap-2 text-slate-100">
          <Filter className="w-5 h-5 text-blue-500" />
          <h2 className="text-sm font-medium">Filters & Actions</h2>
        </div>

        {/* filter inputs grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          
          <div className="flex flex-col gap-1.5">
            <label htmlFor="fromDate" className="text-xs text-slate-400 font-medium">
              From Date
            </label>
            <input
              id="fromDate"
              type="date"
              value={filters.fromDate}
              onChange={(e) => updateFilter('fromDate', e.target.value)}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 disabled:opacity-50"
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
              onChange={(e) => updateFilter('toDate', e.target.value)}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 disabled:opacity-50"
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
              onChange={(e) => updateFilter('minRevenue', e.target.value)}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 disabled:opacity-50"
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
              onChange={(e) => updateFilter('maxRevenue', e.target.value)}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 disabled:opacity-50"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="order" className="text-xs text-slate-400 font-medium">
              Order
            </label>
            <select
              id="order"
              value={filters.order}
              onChange={(e) => updateFilter('order', e.target.value as 'asc' | 'desc')}
              disabled={isDisabled}
              className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-100 disabled:opacity-50"
            >
              <option value="desc">Desc</option>
              <option value="asc">Asc</option>
            </select>
          </div>
        </div>

        {/* action buttons */}
        <div className="flex flex-wrap gap-3">
          
          {/* split button for sync (main button + dropdown toggle) */}
          <div ref={syncControlRef} className="relative inline-flex rounded-lg shadow-sm">
            <button
              onClick={() => handleSyncClick('update')}
              disabled={isDisabled}
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-l-lg disabled:opacity-50 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isSyncing ? 'animate-spin' : ''}`} />
              {isSyncing ? 'Syncing...' : 'Sync'}
            </button>
            
            <button
              type="button"
              onClick={() => setSyncMenuOpen(!syncMenuOpen)}
              disabled={isDisabled}
              className="inline-flex items-center justify-center px-2 bg-blue-600 hover:bg-blue-500 text-white rounded-r-lg border-l border-blue-500/60 disabled:opacity-50 transition-colors"
            >
              <ChevronDown className="w-4 h-4" />
            </button>

            {/* dropdown menu */}
            {syncMenuOpen && !isDisabled && (
              <div className="absolute right-0 top-full mt-2 w-56 rounded-md bg-slate-800 border border-slate-700 shadow-lg z-20 overflow-hidden">
                <button
                  type="button"
                  onClick={() => handleSyncClick('update')}
                  className="w-full px-4 py-3 text-left hover:bg-slate-700 border-b border-slate-700/50"
                >
                  <div className="text-sm font-medium text-slate-100">Update Sync</div>
                  <div className="text-xs text-slate-400 mt-0.5">Faster. Gets recent changes only.</div>
                </button>
                
                <button
                  type="button"
                  onClick={() => handleSyncClick('full')}
                  className="w-full px-4 py-3 text-left hover:bg-slate-700"
                >
                  <div className="text-sm font-medium text-slate-100">Full Sync</div>
                  <div className="text-xs text-slate-400 mt-0.5">Slower. Redownloads everything.</div>
                </button>
              </div>
            )}
          </div>

          <button
            onClick={onLoadJobs}
            disabled={isDisabled}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
          >
            <Filter className="w-4 h-4" />
            Load Jobs
          </button>
        </div>
      </div>
    </div>
  );
}
