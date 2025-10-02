import { Loader2 } from 'lucide-react';
import { formatCurrency, formatDate, classBadgeProps } from '../../utils/jobs';

type ApiJob = Record<string, any>;

interface JobsTableProps {
  jobs: ApiJob[];
  isLoading: boolean;
}

export function JobsTable({ jobs, isLoading }: JobsTableProps) {
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

  const truncate = (text: string | undefined | null, maxLength = 40) => {
    if (!text) return '-';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
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
              try {
                const customerText = typeof job.Customer === 'string' 
                  ? job.Customer 
                  : (job.Customer?.CompanyName || job.Customer?.Name || '-');
                
                const siteText = typeof job.Site === 'string'
                  ? job.Site
                  : (job.Site?.Name || '-');

                return (
                  <tr
                    key={job.ID || index}
                    className="hover:bg-slate-800/50 transition-colors"
                  >
                    <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                      {job.ID || '-'}
                    </td>
                    <td
                      className="px-4 py-3 text-sm text-slate-100"
                      title={String(job.Name || '')}
                    >
                      {truncate(job.Name)}
                    </td>
                    <td
                      className="px-4 py-3 text-sm text-slate-300"
                      title={customerText}
                    >
                      {truncate(customerText)}
                    </td>
                    <td
                      className="px-4 py-3 text-sm text-slate-300"
                      title={siteText}
                    >
                      {truncate(siteText)}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                      {job.Status || '-'}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                      {formatDate(job.Issued)}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-300 whitespace-nowrap">
                      {formatDate(job.Due)}
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-100 text-right whitespace-nowrap font-medium">
                      {formatCurrency(job.revenue ?? 0)}
                    </td>
                    <td className="px-4 py-3 text-sm whitespace-nowrap">
                      {getProfitabilityBadge(job) || (
                        <span className="text-slate-500 text-xs">-</span>
                      )}
                    </td>
                  </tr>
                );
              } catch (error) {
                console.error('Error rendering job row:', error, job);
                return (
                  <tr key={`error-${index}`}>
                    <td colSpan={9} className="px-4 py-3 text-sm text-red-400">
                      Error rendering job {job.ID}: {error instanceof Error ? error.message : 'Unknown error'}
                    </td>
                  </tr>
                );
              }
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
