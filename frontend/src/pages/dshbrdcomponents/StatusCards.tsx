import { AlertCircle, CheckCircle, Loader2, TrendingUp } from '../../components/icons';

/**
STATUS CARDS & ALERTS
---------------------
Small, reusable UI components for displaying status, errors, and progress.
Includes:
 - StatusAlert: Generic success/error/info banner.
 - SyncProgressCard: Specific loading state for the SimPRO sync.
 - PredictionSummary: A mini-dashboard for profitability metrics (Legacy/Optional).
*/

interface StatusAlertProps {
  message: string;
  type: 'success' | 'error' | 'info';
}

export function StatusAlert({ message, type }: StatusAlertProps) {
  const styles = {
    success: {
      bg: 'bg-emerald-900/20 border-emerald-800/50',
      text: 'text-emerald-300',
      icon: CheckCircle,
    },
    error: {
      bg: 'bg-red-900/20 border-red-800/50',
      text: 'text-red-300',
      icon: AlertCircle,
    },
    info: {
      bg: 'bg-blue-900/20 border-blue-800/50',
      text: 'text-blue-300',
      icon: CheckCircle,
    },
  };

  const style = styles[type];
  const Icon = style.icon;

  return (
    <div
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${style.bg} ${style.text}`}
    >
      <Icon className="w-5 h-5 flex-shrink-0" />
      <p className="text-sm font-medium">{message}</p>
    </div>
  );
}

/**
Shows a loading spinner specifically for the SimPRO sync process.

SimPRO's API defaults to 30 jobs per page and has strict rate limits.
Syncing thousands of jobs can take minutes, so we need to set expectations.
 */
export function SyncProgressCard() {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl px-6 py-4">
      <div className="flex items-center gap-4">
        <Loader2 className="w-6 h-6 text-blue-500 animate-spin flex-shrink-0" />
        <div>
          <p className="text-sm font-medium text-slate-100">
            Sync in progress...
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            This can take a few minutes due to SimPRO API rate limits.
          </p>
        </div>
      </div>
    </div>
  );
}

/**
Helper type for the props of the PredictionSummary component.
 */
export type PredictionSummaryData = {
  highCount: number;
  mediumCount: number;
  lowCount: number;
  count: number;
  avgConfidence?: number;
};

interface PredictionSummaryProps {
  summary: PredictionSummaryData;
}

/**
Displays a quick textual summary of profitability predictions.
 */
export function PredictionSummary({ summary }: PredictionSummaryProps) {
  const { highCount, mediumCount, lowCount, count, avgConfidence } = summary;

  const classData = [
    { label: 'High', count: highCount, color: 'text-green-300' },
    { label: 'Medium', count: mediumCount, color: 'text-yellow-300' },
    { label: 'Low', count: lowCount, color: 'text-red-300' },
  ];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl px-6 py-4">
      <div className="flex items-start gap-3">
        <TrendingUp className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1 space-y-2">
          <h3 className="text-sm font-medium text-slate-100">
            Prediction Summary
          </h3>
          <div className="flex flex-wrap gap-4 text-sm">
            {classData.map(({ label, count: classCount, color }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-slate-400">{label}:</span>
                <span className={`font-medium ${color}`}>
                  {classCount}
                </span>
              </div>
            ))}
            {typeof avgConfidence === 'number' && (
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Avg Confidence:</span>
                <span className="font-medium text-blue-300">
                  {(avgConfidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
          <p className="text-xs text-slate-500">
            Total predictions: {count}
          </p>
        </div>
      </div>
    </div>
  );
}