import { useMemo, useState, useCallback } from 'react';
import { Loader2 } from '../../components/icons';
import { formatCurrency, formatDate, classBadgeProps } from '../../utils/jobs';
import type { Job, Prediction } from '../../features/jobs/api';

/**
JOBS TABLE COMPONENT
-------------------
This component is the main component of the Dashboard. It displays a list of SimPRO jobs,
normalized into a common format, and allows the user to sort and select them.

Why this matters:
SimPRO data can be messy. Fields such as "Name" might be "CompanyName" on a customer object,
or "SiteName" on a site object. This table tries to normalise this by using "getter" functions
that look for the best available data.
 */

interface JobsTableProps {
  // The list of jobs to display. Currently strictly typed to our normalized Job interface.
  jobs: Job[];
  // Loading state for the async fetch or sync operation.
  isLoading: boolean;
  // Array of Job IDs that are currently selected (checked).
  selectedJobIds?: string[];
  // Callback when the selection changes.
  onSelectionChange?: (ids: string[]) => void;
  // A dictionary of local predictions, keyed by Job ID.
  predictions?: Record<string, Prediction>;
  // Which type of prediction colums to show.
  predictionType?: 'profitability' | 'duration' | 'none';
}

type SortDirection = 'asc' | 'desc' | null;
type SortKey = 'customer' | 'site' | 'status' | 'issued' | 'revenue' | 'profitability';

// A helper for local sorting state.

interface SortState {
  key: SortKey | null;
  direction: SortDirection;
}

export function JobsTable({ 
  jobs, 
  isLoading, 
  selectedJobIds = [], 
  onSelectionChange, 
  predictions = {}, 
  predictionType = 'none' 
}: JobsTableProps) {
  // -- Sorting State --
  const [sortState, setSortState] = useState<SortState>({
    key: null,
    direction: null,
  });

  /**
  Safe String Converter
  ---------------------
  SimPRO APIs can return objects where we expect strings, or numbers 
  where we expect strings. This helper tries to extract a meaningful label 
  from the input.
  */
  const toDisplayString = (value: unknown): string => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'string') return value;
    
    // If it's an object, try to find a "Name" or "Title" property
    if (typeof value === 'object') {
      const record = value as Record<string, unknown>;
      const candidate =
        record.name ??
        record.Name ??
        record.title ??
        record.CompanyName ??
        record.id ??
        record.ID ??
        null;

      if (typeof candidate === 'string') return candidate;
      if (typeof candidate === 'number') return String(candidate);

      // Last resort: JSON stringify
      try {
        return JSON.stringify(value);
      } catch (error) {
        console.error('Unable to stringify value:', error, value);
        return '-';
      }
    }
    
    // Primitives
    try {
      return String(value);
    } catch (error) {
      console.error('Unable to convert value to string:', error, value);
      return '-';
    }
  };

  /**
  Safe Number Parser
  ------------------
  Handles "$1,200.50", "1200", or 1200.
  */
  const toNumber = (value: unknown): number | null => {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : null;
    }

    if (typeof value === 'string') {
      // Strip currency symbols and commas
      const parsed = Number(value.replace(/[^0-9.\-]/g, ''));
      return Number.isFinite(parsed) ? parsed : null;
    }

    return null;
  };

  /**
  Generates a string suitable for sorting.
  */
  const toSortableString = (value: unknown): string | null => {
    const display = toDisplayString(value);
    if (!display || display === '-' || !display.trim()) return null;
    return display.trim().toLowerCase();
  };

  // -- Data Accessors --
  // These functions locate data within the Job object.

  const getCustomerValue = (job: Job) =>
    job.customerName ??
    // Fallbacks for raw data properties if they exist
    (job as any).Customer?.CompanyName ??
    (job as any).Customer?.Name ??
    (job as any).Customer ??
    null;

  const getSiteValue = (job: Job) => 
    job.siteName ?? 
    (job as any).Site?.Name ?? 
    (job as any).Site ?? 
    null;

  const getStatusValue = (job: Job) =>
    job.status_name ??
    job.status?.Name ??
    (job as any).Status?.Name ?? // Handle case where 'status' / 'Status' casing is different
    (job as any).Status ?? 
    job.stage ?? 
    null;

  const getIssuedValue = (job: Job) =>
    job.dateIssued ??
    (job as any).Issued ?? 
    (job as any).DateIssued ?? 
    null;

  const getDueValue = (job: Job) =>
    job.dateDue ??
    (job as any).Due ?? 
    (job as any).DueDate ?? 
    null;

  const getNameValue = (job: Job) => 
    job.descriptionText ?? 
    job.Description ?? 
    (job as any).Name ?? // Some SimPRO endpoints use Name, others Description
    (job as any).RequestNo ?? 
    null;

  const getIdValue = (job: Job) => 
    job.id ?? 
    job.ID ?? 
    null;

  const getRevenueValue = (job: Job) =>
    toNumber(job.revenue) ?? 
    toNumber((job as any).Total?.IncTax) ?? 
    null;

  const getProfitClassValue = (job: Job) =>
    job.profitability_class ?? 
    (job as any).profitability?.class ?? 
    null;

  // Map High/Medium/Low to numeric values for sorting
  const getProfitRank = (job: Job): number | null => {
    const cls = getProfitClassValue(job);
    if (!cls || typeof cls !== 'string') return null;
    const key = cls.trim().toLowerCase();
    
    // Higher number = Better profitability
    if (key === 'high') return 3;
    if (key === 'medium') return 2;
    if (key === 'low') return 1;
    return 0; 
  };

  /**
  Badge Renderer
  Renders the colored pill for Profitability (High/Medium/Low).
  */
  const getProfitabilityBadge = (job: Job) => {
    const profitClass = getProfitClassValue(job);

    if (!profitClass) return null;

    const badge = classBadgeProps(profitClass);

    // Map the tone to Tailwind classes
    const styles = {
      success: 'bg-green-900/30 text-green-300 border-green-800/50',
      warning: 'bg-yellow-900/30 text-yellow-300 border-yellow-800/50',
      destructive: 'bg-red-900/30 text-red-300 border-red-800/50',
      default: 'bg-slate-800 border-slate-700 text-slate-300',
    };

    const className = styles[badge.tone as keyof typeof styles] || styles.default;

    // Is there a calculated score attached?
    const score = (job as any).profitability?.score;
    const scoreType = (job as any).profitability?.scoreType;
    const tooltip =
      score != null
        ? `${scoreType || 'score'}: ${(score * 100).toFixed(1)}%`
        : undefined;

    return (
      <span
        className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${className}`}
        title={tooltip}
      >
        {badge.label}
      </span>
    );
  };

  const truncate = (value: unknown, maxLength = 40) => {
    const text = toDisplayString(value);
    if (text === '-' || text.length <= maxLength) return text;
    return `${text.substring(0, maxLength)}...`;
  };

  const parseDateValue = (value: unknown): Date | null => {
    if (value === null || value === undefined) return null;
    if (value instanceof Date) {
      return Number.isFinite(value.getTime()) ? value : null;
    }
    if (typeof value === 'string' || typeof value === 'number') {
      const candidate = new Date(value);
      return Number.isFinite(candidate.getTime()) ? candidate : null;
    }
    return null;
  };

  // -- Sorting Logic --
  // This configuration object tells us how to extract values for each sort key,
  // and what type of comparison to perform (string vs number).
  
  function getSortConfig(key: SortKey): {
    type: 'string' | 'number';
    extractor: (job: Job) => string | number | null;
  } {
    switch (key) {
      case 'customer':
        return {
          type: 'string',
          extractor: (job) => toSortableString(getCustomerValue(job)),
        };
      case 'site':
        return {
          type: 'string',
          extractor: (job) => toSortableString(getSiteValue(job)),
        };
      case 'status':
        return {
          type: 'string',
          extractor: (job) => toSortableString(getStatusValue(job)),
        };
      case 'issued':
        return {
          type: 'number',
          extractor: (job) => {
            const issued = getIssuedValue(job) ?? getDueValue(job);
            // Assuming ISO strings for dates
            const parsed = parseDateValue(issued);
            return parsed ? parsed.getTime() : null;
          },
        };
      case 'revenue':
        return {
          type: 'number',
          extractor: (job) => getRevenueValue(job),
        };
      case 'profitability':
      default:
        return {
          type: 'number',
          extractor: (job) => getProfitRank(job),
        };
    }
  }

  // Safe comparisons that handle nulls
  const compareValues = (
    aValue: string | number | null,
    bValue: string | number | null,
    type: 'string' | 'number',
  ) => {
    if (aValue === null || aValue === undefined) {
      // If both are null, they are equal. If only A is null, it goes after B.
      return bValue === null || bValue === undefined ? 0 : 1;
    }
    if (bValue === null || bValue === undefined) {
      return -1;
    }

    if (type === 'number') {
      return (aValue as number) - (bValue as number);
    }

    // String comparison for proper handling of accents, layout, etc.
    return String(aValue).localeCompare(String(bValue), undefined, {
      sensitivity: 'base',
      numeric: true,
    });
  };

  // Memoized sorted list of jobs.
  // We map to {job, index} to guarantee correct sorting when values are equal.
  const sortedJobs = useMemo(() => {
    if (!sortState.key || !sortState.direction) {
      return jobs;
    }

    const { extractor, type } = getSortConfig(sortState.key);
    const directionFactor = sortState.direction === 'asc' ? 1 : -1;

    return [...jobs]
      .map((job, index) => ({ job, index }))
      .sort((a, b) => {
        const aValue = extractor(a.job);
        const bValue = extractor(b.job);
        
        const primary = compareValues(aValue, bValue, type);
        if (primary !== 0) {
          return primary * directionFactor;
        }
        return a.index - b.index;
      })
      .map(({ job }) => job);
  }, [jobs, sortState]);

  const handleSort = useCallback(
    (key: SortKey) => {
      setSortState((prev) => {
        // If clicking the same key, toggle direction
        if (prev.key !== key) {
          return { key, direction: 'asc' };
        }
        if (prev.direction === 'asc') {
          return { key, direction: 'desc' };
        }
        if (prev.direction === 'desc') {
          // Third click removes sort
          return { key: null, direction: null }; 
        }
        return { key, direction: 'asc' };
      });
    },
    [],
  );

  const renderSortIndicator = (key: SortKey) => {
    if (sortState.key !== key || !sortState.direction) {
      return <span className="text-slate-600 ml-1">⇅</span>;
    }
    if (sortState.direction === 'asc') {
      return <span className="text-slate-200 ml-1">▲</span>;
    }
    return <span className="text-slate-200 ml-1">▼</span>;
  };

  if (jobs.length === 0 && !isLoading) {
    return (
      <div className="relative min-h-[400px] bg-slate-900 rounded-xl border border-slate-800">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <p className="text-slate-400 text-sm">No jobs found</p>
            <p className="text-slate-500 text-xs mt-1">
              Try adjusting your filters or load jobs
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      {isLoading && (
        <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm z-10 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
            <p className="text-sm text-slate-300">Loading...</p>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-800">
              {/* Select All Checkbox */}
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap w-[40px]">
                <input
                  type="checkbox"
                  className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-blue-500/20"
                  checked={jobs.length > 0 && jobs.every((j) => selectedJobIds.includes(String(getIdValue(j) ?? '')))}
                  onChange={(e) => {
                    if (!onSelectionChange) return;
                    if (e.target.checked) {
                      const ids = jobs
                        .map((j) => getIdValue(j))
                        .filter((x) => x !== null)
                        .map((x) => String(x));
                      onSelectionChange(ids);
                    } else {
                      onSelectionChange([]);
                    }
                  }}
                />
              </th>
              
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Description / Name
              </th>
              
              {/* Sortable Columns */}
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('customer')}
                  className="flex items-center gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors"
                >
                  <span>Customer</span>
                  {renderSortIndicator('customer')}
                </button>
              </th>
              
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('site')}
                  className="flex items-center gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors"
                >
                  <span>Site</span>
                  {renderSortIndicator('site')}
                </button>
              </th>
              
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('status')}
                  className="flex items-center gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors"
                >
                  <span>Status</span>
                  {renderSortIndicator('status')}
                </button>
              </th>
              
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('issued')}
                  className="flex items-center gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors"
                >
                  <span>Issued</span>
                  {renderSortIndicator('issued')}
                </button>
              </th>

              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Due
              </th>
              
              <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('revenue')}
                  className="flex items-center justify-end gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors w-full"
                >
                  <span>Revenue (IncTax)</span>
                  {renderSortIndicator('revenue')}
                </button>
              </th>
              
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <button
                  type="button"
                  onClick={() => handleSort('profitability')}
                  className="flex items-center gap-1 uppercase tracking-wider text-xs font-medium text-slate-400 hover:text-slate-100 transition-colors"
                >
                  <span>Profitability</span>
                  {renderSortIndicator('profitability')}
                </button>
              </th>

              {/* Dynamic Prediction Columns */}
              {predictionType === 'profitability' && (
                <>
                  <th className="px-4 py-3 text-left text-xs font-medium text-purple-400 uppercase tracking-wider whitespace-nowrap">Predicted Class</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-purple-400 uppercase tracking-wider whitespace-nowrap">Confidence</th>
                </>
              )}
            </tr>
          </thead>
          
          <tbody className="divide-y divide-slate-800">
            {sortedJobs.map((job, index) => {
              // Safety check for empty slots can happen with bad API data)
              if (!job || typeof job !== 'object') {
                return (
                  <tr key={`invalid-${index}`} className="bg-red-900/20">
                    <td colSpan={11} className="px-4 py-3 text-sm text-red-300">
                      Row {index + 1}: Invalid Data
                    </td>
                  </tr>
                );
              }

              const idValue = getIdValue(job);
              const jobIdKey = String(idValue ?? `row-${index}`);
              
              // Extract values for display
              const nameValue = getNameValue(job);
              // Use descriptionText if available as it's cleaner
              const displayName = job.descriptionText || toDisplayString(nameValue);
              const nameText = truncate(displayName, 60);
              
              const customerValue = getCustomerValue(job);
              const customerText = truncate(customerValue);
              
              const siteValue = getSiteValue(job);
              const siteText = truncate(siteValue);
              
              const statusValue = getStatusValue(job);
              
              const issuedValue = getIssuedValue(job);
              const dueValue = getDueValue(job);
              const revenueValue = getRevenueValue(job);

              return (
                <tr
                  key={jobIdKey}
                  className="hover:bg-slate-800/50 transition-colors group"
                >
                  {/* Select Row */}
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    <input
                      type="checkbox"
                      className="rounded border-slate-700 bg-slate-800 text-blue-600 focus:ring-blue-500/20"
                      checked={selectedJobIds.includes(jobIdKey)}
                      onChange={(e) => {
                        if (!onSelectionChange) return;
                        const next = new Set(selectedJobIds);
                        if (e.target.checked) next.add(jobIdKey);
                        else next.delete(jobIdKey);
                        onSelectionChange(Array.from(next));
                      }}
                    />
                  </td>
                  
                  {/* ID */}
                  <td className="px-4 py-3 text-sm text-slate-400 font-mono whitespace-nowrap">
                    {idValue}
                  </td>
                  
                  {/* Name/Desc */}
                  <td
                    className="px-4 py-3 text-sm text-slate-100 font-medium"
                    title={toDisplayString(nameValue)}
                  >
                    {nameText}
                  </td>
                  
                  {/* Customer */}
                  <td
                    className="px-4 py-3 text-sm text-slate-300"
                    title={toDisplayString(customerValue)}
                  >
                    {customerText}
                  </td>
                  
                  {/* Site */}
                  <td
                    className="px-4 py-3 text-sm text-slate-300"
                    title={toDisplayString(siteValue)}
                  >
                    {siteText}
                  </td>
                  
                  {/* Status */}
                  <td
                    className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap"
                    title={toDisplayString(statusValue)}
                  >
                    {statusValue}
                  </td>
                  
                  {/* Issued */}
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {formatDate(issuedValue)}
                  </td>
                  
                  {/* Due */}
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {formatDate(dueValue)}
                  </td>
                  
                  {/* Revenue */}
                  <td className="px-4 py-3 text-sm text-slate-100 text-right whitespace-nowrap font-medium">
                    {revenueValue !== null ? formatCurrency(revenueValue) : '-'}
                  </td>
                  
                  {/* Profitability Badge (Historical or Estimated) */}
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    {getProfitabilityBadge(job) || (
                      <span className="text-slate-600 text-xs italic">Unknown</span>
                    )}
                  </td>

                  {/* ML Predictions */}
                  {predictionType === 'profitability' && (
                    <>
                      <td className="px-4 py-3 text-sm text-slate-100 whitespace-nowrap">
                        {(() => {
                          const p = predictions[jobIdKey];
                          // Use prediction if available, else fall back to known class
                          const klass = p?.class ?? job.profitability_class ?? (job as any).profitability?.class ?? null;
                          
                          if (p?.class) {
                            return <span className="text-purple-300 font-semibold">{p.class} ✨</span>;
                          }
                          return klass ? (
                            <span className="text-slate-400">{klass}</span>
                          ) : (
                            <span className="text-slate-600">-</span>
                          );
                        })()}
                      </td>
                      <td className="px-4 py-3 text-sm text-right whitespace-nowrap">
                        {(() => {
                          const p = predictions[jobIdKey];
                          const score = p?.confidence ?? p?.probability ?? null;
                          return typeof score === 'number' ? (
                            <span className="text-purple-300 font-mono">{(score * 100).toFixed(0)}%</span>
                          ) : (
                            <span className="text-slate-600">-</span>
                          );
                        })()}
                      </td>
                    </>
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
