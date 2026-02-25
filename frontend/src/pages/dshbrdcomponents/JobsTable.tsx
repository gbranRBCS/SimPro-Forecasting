import { useMemo, useState } from 'react';
import { Loader2, Search } from '../../components/icons';
import { formatCurrency, formatDate } from '../../utils/jobs';
import type { Job, Prediction } from '../../features/jobs/api';

// Import Algorithms
import { searchJobs } from '../../utils/searching';
import { sortJobs, type SortKey, type SortDirection } from '../../utils/sorting';

// this component is the main data table for the dashboard
// it handles listing jobs, searching, sorting and selection
// it also shows prediction results inline when available

interface JobsTableProps {
  jobs: Job[];
  isLoading: boolean;
  selectedJobIds?: string[];
  onSelectionChange?: (ids: string[]) => void;
  predictions?: Record<string, Prediction>;
  predictionType?: 'profitability' | 'duration' | 'none';
}

interface SortState {
  key: SortKey | null;
  direction: SortDirection | null;
}

export function JobsTable({ 
  jobs, 
  isLoading, 
  selectedJobIds = [], 
  onSelectionChange, 
  predictions = {}, 
  predictionType = 'none' 
}: JobsTableProps) {
  
  const [sortState, setSortState] = useState<SortState>({
    key: null,
    direction: null,
  });

  const [searchTerm, setSearchTerm] = useState('');

  // Derived Data:

  // filter and sort the jobs based on user input
  const processedJobs = useMemo(() => {
    // first apply search filter
    const searched = searchJobs(jobs, searchTerm);

    // then apply sorting
    return sortJobs(
      searched, 
      sortState.key, 
      sortState.direction, 
      predictionType === 'profitability' ? predictions : undefined
    );
  }, [jobs, sortState, searchTerm, predictions, predictionType]);

  // -- Handlers --

  function handleSort(key: SortKey) {
    setSortState((prev) => {
      // if clicking the same column, flip direction
      if (prev.key === key) {
        if (prev.direction === 'asc') return { key, direction: 'desc' };
        // if desc, clear sort
        if (prev.direction === 'desc') return { key: null, direction: null };
      }
      // default to asc for new column
      return { key, direction: 'asc' };
    });
  }

  function handleSelectAll(e: React.ChangeEvent<HTMLInputElement>) {
    if (!onSelectionChange) return;
    
    if (e.target.checked) {
      // select all currently visible jobs
      const allIds = processedJobs.map(j => String(j.id));
      onSelectionChange(allIds);
    } else {
      onSelectionChange([]);
    }
  }

  function handleSelectRow(id: string, checked: boolean) {
    if (!onSelectionChange) return;

    if (checked) {
      onSelectionChange([...selectedJobIds, id]);
    } else {
      onSelectionChange(selectedJobIds.filter(selectedId => selectedId !== id));
    }
  }

  // helper to render the sort arrow
  function SortIndicator({ active, direction }: { active: boolean, direction: SortDirection | null }) {
    if (!active || !direction) return <span className="text-slate-600 ml-1">⇅</span>;
    return <span className="text-slate-200 ml-1">{direction === 'asc' ? '▲' : '▼'}</span>;
  }

  // helper to column headers to reduce repetition
  function ColumnHeader({ label, sortKey, align = 'left' }: { label: string, sortKey?: SortKey, align?: 'left'|'right' }) {
    const isSortable = !!sortKey;
    const alignClass = align === 'right' ? 'text-right' : 'text-left';
    
    if (!isSortable) {
      return (
        <th className={`px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap ${alignClass}`}>
          {label}
        </th>
      );
    }

    return (
      <th className={`px-4 py-3 text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap ${alignClass}`}>
        <button
          type="button"
          onClick={() => handleSort(sortKey)}
          className={`flex items-center gap-1 uppercase tracking-wider hover:text-slate-100 transition-colors ${align === 'right' ? 'justify-end w-full' : ''}`}
        >
          <span>{label}</span>
          <SortIndicator active={sortState.key === sortKey} direction={sortState.direction} />
        </button>
      </th>
    );
  }

  // check if all currently visible jobs are selected
  const allSelected = processedJobs.length > 0 && processedJobs.every(j => selectedJobIds.includes(String(j.id)));

  // -- Render --

  if (jobs.length === 0 && !isLoading) {
    return (
      <div className="relative min-h-[400px] bg-slate-900 rounded-xl border border-slate-800 flex items-center justify-center">
        <div className="text-center text-slate-400">
          <p className="text-sm">No jobs found</p>
          <p className="text-xs mt-1 text-slate-500">Try adjusting your filters or load jobs</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      
      {/* update loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm z-20 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <p className="text-sm text-slate-300">Loading...</p>
          </div>
        </div>
      )}

      {/* search bar */}
      <div className="p-4 border-b border-slate-800 bg-slate-900/50">
        <div className="relative max-w-md">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-slate-500" />
          </div>
          <input
            type="text"
            className="block w-full pl-10 pr-3 py-2 border border-slate-700 rounded-lg bg-slate-950/50 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 text-sm transition-all"
            placeholder="Search by customer, site, or job description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/50">
              <th className="px-4 py-3 w-[40px]">
                <input
                  type="checkbox"
                  className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-blue-500/20 cursor-pointer"
                  checked={allSelected}
                  onChange={handleSelectAll}
                />
              </th>
              
              <ColumnHeader label="ID" />
              <ColumnHeader label="Description / Name" />
              <ColumnHeader label="Customer" sortKey="customer" />
              <ColumnHeader label="Site" sortKey="site" />
              <ColumnHeader label="Status" sortKey="status" />
              <ColumnHeader label="Issued" sortKey="issued" />
              <ColumnHeader label="Due" />
              <ColumnHeader label="Revenue (IncTax)" sortKey="revenue" align="right" />
              <ColumnHeader label="Profitability" sortKey="profitability" />

              {predictionType === 'profitability' && (
                <ColumnHeader label="Confidence" align="right" />
              )}
            </tr>
          </thead>
          
          <tbody className="divide-y divide-slate-800">
            {processedJobs.map((job) => {
              const idString = String(job.id);
              const isSelected = selectedJobIds.includes(idString);
              
              // get prediction if available
              const prediction = predictions[idString];
              const profitClass = prediction?.class;
              
              // dynamic row styling based on prediction
              let rowClass = "hover:bg-slate-800/50 transition-colors";
              let badgeColor = "bg-slate-800 text-slate-300 border-slate-700";

              if (predictionType === 'profitability' && profitClass) {
                const p = profitClass.toLowerCase();
                if (p === 'high') {
                  rowClass = "bg-green-950/10 hover:bg-green-900/20";
                  badgeColor = "bg-green-900/30 text-green-300 border-green-800/50";
                } else if (p === 'medium') {
                  rowClass = "bg-yellow-950/10 hover:bg-yellow-900/20";
                  badgeColor = "bg-yellow-900/30 text-yellow-300 border-yellow-800/50";
                } else if (p === 'low') {
                  rowClass = "bg-red-950/10 hover:bg-red-900/20";
                  badgeColor = "bg-red-900/30 text-red-300 border-red-800/50";
                }
              }

              return (
                <tr key={job.id} className={rowClass}>
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-blue-500/20 cursor-pointer"
                      checked={isSelected}
                      onChange={(e) => handleSelectRow(idString, e.target.checked)}
                    />
                  </td>
                  
                  <td className="px-4 py-3 text-sm text-slate-400 font-mono">{job.id}</td>
                  <td className="px-4 py-3 text-sm text-slate-100 font-medium max-w-xs truncate" title={job.descriptionText}>
                    {job.descriptionText}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 max-w-[150px] truncate" title={job.customerName}>
                    {job.customerName}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 max-w-[150px] truncate" title={job.siteName}>
                    {job.siteName}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">{job.status_name}</td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">{formatDate(job.dateIssued)}</td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">{formatDate(job.dateDue)}</td>
                  <td className="px-4 py-3 text-sm text-slate-100 text-right font-medium whitespace-nowrap">
                    {job.revenue !== null ? formatCurrency(job.revenue) : '-'}
                  </td>
                  
                  <td className="px-4 py-3 whitespace-nowrap">
                    {profitClass ? (
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${badgeColor}`}>
                        {profitClass}
                      </span>
                    ) : (
                      <span className="text-slate-600 text-xs italic">Unknown</span>
                    )}
                  </td>

                  {predictionType === 'profitability' && (
                    <td className="px-4 py-3 text-sm text-right whitespace-nowrap font-mono text-slate-400">
                      {prediction?.confidence ? `${(prediction.confidence * 100).toFixed(0)}%` : '-'}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
