import { Loader2 } from '../../components/icons';
import { formatCurrency, formatDate, classBadgeProps } from '../../utils/jobs';

type ApiJob = Record<string, any>;

interface JobsTableProps {
  jobs: ApiJob[];
  isLoading: boolean;
}

export function JobsTable({ jobs, isLoading }: JobsTableProps) {
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
                ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Name
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Customer
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Site
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Issued
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Due
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Revenue (IncTax)
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-slate-400 uppercase tracking-wider whitespace-nowrap">
                Profitability
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {jobs.map((job, index) => {
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
              const jobKey = jobRecord.ID ?? jobRecord.id ?? `row-${index}`;

              const customerValue =
                jobRecord.customerName ??
                jobRecord.Customer?.CompanyName ??
                jobRecord.Customer?.Name ??
                jobRecord.Customer ??
                null;
              const siteValue =
                jobRecord.siteName ??
                jobRecord.Site?.Name ??
                jobRecord.Site ??
                null;
              const nameValue = jobRecord.Name ?? jobRecord.RequestNo;
              const statusValue =
                jobRecord.statusName ??
                jobRecord.Status?.Name ??
                jobRecord.Status ??
                jobRecord.Stage;
              const idValue = jobRecord.ID ?? jobRecord.id;

              const customerText = truncate(customerValue);
              const siteText = truncate(siteValue);
              const nameText = truncate(nameValue);
              const statusText = truncate(statusValue);
              const idText = truncate(idValue);
              const customerTitle = toDisplayString(customerValue);
              const siteTitle = toDisplayString(siteValue);
              const revenueValue =
                toNumber(jobRecord.revenue) ?? toNumber(jobRecord?.Total?.IncTax);

              return (
                <tr
                  key={jobKey}
                  className="hover:bg-slate-800/50 transition-colors"
                >
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
                    {formatDate(jobRecord.Issued ?? jobRecord.DateIssued)}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                    {formatDate(jobRecord.Due ?? jobRecord.DueDate)}
                  </td>
                  <td className="px-4 py-3 text-sm text-slate-100 text-right whitespace-nowrap font-medium">
                    {formatCurrency(revenueValue)}
                  </td>
                  <td className="px-4 py-3 text-sm whitespace-nowrap">
                    {getProfitabilityBadge(jobRecord) || (
                      <span className="text-slate-500 text-xs">-</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
