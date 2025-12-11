import { useMemo, useState, useCallback } from 'react';
import { Loader2 } from '../../components/icons';
import { formatCurrency, formatDate, classBadgeProps } from '../../utils/jobs';
import type { Prediction } from '../../features/jobs/api';

type ApiJob = Record<string, any>;

interface JobsTableProps {
  jobs: ApiJob[];
  isLoading: boolean;
  selectedJobIds?: string[];
  onSelectionChange?: (ids: string[]) => void;
  predictions?: Record<string, Prediction>;
  predictionType?: 'profitability' | 'duration' |'none';
}

type SortDirection = 'asc' | 'desc' | null;
type SortKey = 'customer' | 'site' | 'status' | 'issued' | 'revenue' | 'profitability';

export function JobsTable({ jobs, isLoading, selectedJobIds = [], onSelectionChange, predictions = {}, predictionType = 'none' }: JobsTableProps) {
  const [sortState, setSortState] = useState<{ key: SortKey | null; direction: SortDirection }>({
    key: null,
    direction: null,
  });

  const toDisplayString = (value: unknown): string => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'string') return value;
    if (typeof value === 'object') {
      const candidate =
        (value as Record<string, unknown>).name ??
        (value as Record<string, unknown>).Name ??
        (value as Record<string, unknown>).title ??
        (value as Record<string, unknown>).CompanyName ??
        (value as Record<string, unknown>).id ??
        (value as Record<string, unknown>).ID ??
        null;

      if (typeof candidate === 'string') return candidate;
      if (typeof candidate === 'number') return String(candidate);

      // fall back to safe string conversion without exposing [object Object]
      try {
        return JSON.stringify(value);
      } catch (error) {
        console.error('Unable to stringify value:', error, value);
        return '-';
      }
    }
    try {
      return String(value);
    } catch (error) {
      console.error('Unable to convert value to string:', error, value);
      return '-';
    }
  };

  const toNumber = (value: unknown): number | null => {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : null;
    }

    if (typeof value === 'string') {
      const parsed = Number(value.replace(/[^0-9.\-]/g, ''));
      return Number.isFinite(parsed) ? parsed : null;
    }

    return null;
  };

  const toSortableString = (value: unknown): string | null => {
    const display = toDisplayString(value);
    if (!display || display === '-' || !display.trim()) return null;
    return display.trim().toLowerCase();
  };

  const getCustomerValue = (job: ApiJob) =>
    job.customerName ??
    job.Customer?.CompanyName ??
    job.Customer?.Name ??
    job.Customer ??
    null;

  const getSiteValue = (job: ApiJob) => job.siteName ?? job.Site?.Name ?? job.Site ?? null;

  const getStatusValue = (job: ApiJob) =>
    job.statusName ?? job.Status?.Name ?? job.Status ?? job.Stage ?? null;

  const getIssuedValue = (job: ApiJob) =>
    job.Issued ?? job.DateIssued ?? job.dateIssued ?? job.issued ?? null;

  const getDueValue = (job: ApiJob) =>
    job.Due ?? job.DueDate ?? job.dateDue ?? job.DateDue ?? job.due ?? null;

  const getNameValue = (job: ApiJob) => job.Name ?? job.RequestNo ?? null;

  const getIdValue = (job: ApiJob) => job.ID ?? job.id ?? null;

  const getRevenueValue = (job: ApiJob) =>
    toNumber(job.revenue) ?? toNumber(job.Total?.IncTax) ?? null;

  const getProfitClassValue = (job: ApiJob) =>
    job.profitability?.class ?? job.profitability_class ?? null;

  const getProfitRank = (job: ApiJob): number | null => {
    const cls = getProfitClassValue(job);
    if (!cls || typeof cls !== 'string') return null;
    const key = cls.trim().toLowerCase();
    if (key === 'low') return 0;
    if (key === 'medium') return 1;
    if (key === 'high') return 2;
    return null;
  };

  const getProfitabilityBadge = (job: ApiJob) => {
    // check multiple possible locations for profitability class
    const profitClass =
      job.profitability?.class || job.profitability_class || null;

    if (!profitClass) return null;

    const badge = classBadgeProps(profitClass);

    // map tone to Tailwind classes
    const styles = {
      success: 'bg-green-900/30 text-green-300 border-green-800/50',
      warning: 'bg-yellow-900/30 text-yellow-300 border-yellow-800/50',
      destructive: 'bg-red-900/30 text-red-300 border-red-800/50',
      default: 'bg-slate-800 border-slate-700 text-slate-300',
    };

    const className = styles[badge.tone as keyof typeof styles] || styles.default;

    // show score in tooltip if available
    const score = job.profitability?.score;
    const scoreType = job.profitability?.scoreType;
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

  function getSortConfig(key: SortKey): {
    type: 'string' | 'number';
    extractor: (job: ApiJob) => string | number | null;
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

  const compareValues = (
    aValue: string | number | null,
    bValue: string | number | null,
    type: 'string' | 'number',
  ) => {
    if (aValue === null || aValue === undefined) {
      return bValue === null || bValue === undefined ? 0 : 1;
    }
    if (bValue === null || bValue === undefined) {
      return -1;
    }

    if (type === 'number') {
      return (aValue as number) - (bValue as number);
    }

    return String(aValue).localeCompare(String(bValue), undefined, {
      sensitivity: 'base',
      numeric: true,
    });
  };

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
        if (prev.key !== key) {
          return { key, direction: 'asc' };
        }
        if (prev.direction === 'asc') {
          return { key, direction: 'desc' };
        }
        if (prev.direction === 'desc') {
          return { key: null, direction: null };
        }
        return { key, direction: 'asc' };
      });
    },
    [],
  );

  const renderSortIndicator = (key: SortKey) => {
    if (sortState.key !== key || !sortState.direction) {
      return <span className="text-slate-600">⇅</span>;
    }
    if (sortState.direction === 'asc') {
      return <span className="text-slate-200">▲</span>;
    }
    return <span className="text-slate-200">▼</span>;
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
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                <input
                  type="checkbox"
                  checked={jobs.length > 0 && jobs.every((j) => selectedJobIds.includes(String(j.ID ?? j.id ?? '')))}
                  onChange={(e) => {
                    if (!onSelectionChange) return;
                    if (e.target.checked) {
                      const ids = jobs
                        .map((j) => j.ID ?? j.id ?? null)
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
                Name
              </th>
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
              {predictionType === 'profitability' && (
                <>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">Predicted class</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">Confidence</th>
                </>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {sortedJobs.map((job, index) => {
              if (!job || typeof job !== 'object') {
                return (
                  <tr key={`invalid-${index}`} className="bg-red-900/20">
                    <td colSpan={9} className="px-4 py-3 text-sm text-red-300">
                      Unable to display job row {index + 1} due to invalid data.
                    </td>
                  </tr>
                );
              }

              const jobRecord = job as ApiJob;
              const jobKey = getIdValue(jobRecord) ?? `row-${index}`;

              const customerValue = getCustomerValue(jobRecord);
              const siteValue = getSiteValue(jobRecord);
              const nameValue = getNameValue(jobRecord);
              const statusValue = getStatusValue(jobRecord);
              const idValue = getIdValue(jobRecord);
              const issuedValue = getIssuedValue(jobRecord);
              const dueValue = getDueValue(jobRecord);
              const revenueValue = getRevenueValue(jobRecord);

              const customerText = truncate(customerValue);
              const siteText = truncate(siteValue);
              const nameText = truncate(nameValue);
              const statusText = truncate(statusValue);
              const idText = truncate(idValue);
              const customerTitle = toDisplayString(customerValue);
              const siteTitle = toDisplayString(siteValue);

              const jobIdKey = String(getIdValue(jobRecord) ?? `row-${index}`);

              return (
                <tr
                  key={jobKey}
                  className="hover:bg-slate-800/50 transition-colors"
                >
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    <input
                      type="checkbox"
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
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {idText}
                  </td>
                  <td
                    className="px-4 py-3 text-sm text-slate-100"
                    title={toDisplayString(nameValue)}
                  >
                    {nameText}
                  </td>
                  <td
                    className="px-4 py-3 text-sm text-slate-300"
                    title={customerTitle}
                  >
                    {customerText}
                  </td>
                  <td
                    className="px-4 py-3 text-sm text-slate-300"
                    title={siteTitle}
                  >
                    {siteText}
                  </td>
                  <td
                    className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap"
                    title={toDisplayString(statusValue)}
                  >
                    {statusText}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {formatDate(issuedValue)}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {formatDate(dueValue)}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-100 text-right whitespace-nowrap font-medium">
                    {formatCurrency(revenueValue)}
                  </td>
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    {getProfitabilityBadge(jobRecord) || (
                      <span className="text-slate-500 text-xs">-</span>
                    )}
                    {predictionType === 'profitability' && (
                      <></>
                    )}
                  </td>
                  {predictionType === 'profitability' && (
                    <>
                      <td className="px-4 py-3 text-sm text-slate-100 whitespace-nowrap">
                        {(() => {
                          const p = predictions[jobIdKey];
                          const klass = p?.class ?? jobRecord.profitability?.class ?? jobRecord.profitability_class ?? null;
                          return klass ? (
                            <span className="text-sm text-slate-100">{klass}</span>
                          ) : (
                            <span className="text-slate-500 text-xs">-</span>
                          );
                        })()}
                      </td>
                      <td className="px-4 py-3 text-sm text-right whitespace-nowrap">
                        {(() => {
                          const p = predictions[jobIdKey];
                          const score = p?.confidence ?? p?.probability ?? null;
                          return typeof score === 'number' ? (
                            <span className="text-slate-300">{(score * 100).toFixed(0)}%</span>
                          ) : (
                            <span className="text-slate-500 text-xs">-</span>
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
