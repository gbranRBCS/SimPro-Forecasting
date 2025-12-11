import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

type ApiJob = Record<string, any>;

type Prediction = {
  jobId: string | number | null;
  class?: 'Low' | 'Medium' | 'High';
  confidence?: number; // 0..1
};

interface PredictionPanelProps {
  predictionType: 'profitability' | 'duration' |'none';
  selectedJobs: ApiJob[];
  predictions: Record<string, Prediction>;
  loading: boolean;
  error?: string | null;
  onPredict: () => void;
  onPredictionTypeChange: (t: 'profitability' | 'duration' |'none') => void;
}

export function PredictionPanel({ predictionType, selectedJobs, predictions, loading, error, onPredict, onPredictionTypeChange }: PredictionPanelProps) {
  if (predictionType === 'none') {
    return (
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Predictions</h2>
        <p className="text-sm text-slate-400">No prediction model selected.</p>
      </div>
    );
  }

  const totalSelected = selectedJobs.length;

  // standard header
  return ( 
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-slate-100">Predictions</h2>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400">Model</label>
          <select
            value={predictionType}
            onChange={(e) => onPredictionTypeChange(e.target.value as 'profitability' | 'duration' |'none')}
            className="px-2 py-1 bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-100"
          >
            <option value="profitability">Profitability</option>
            <option value="duration">Duration</option>
          </select>
          <button
            onClick={onPredict}
            disabled={loading || totalSelected === 0}
            className="inline-flex items-center gap-2 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-600/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Predict
          </button>
        </div>
      </div>

      {/* Switch based on prediction choice */}
      {predictionType === 'profitability' && (
        <ProfitabilityView
        predictions={predictions}
        selectedJobs={selectedJobs}
        loading={loading}
        error={error}
        />
      )}

      {predictionType === 'duration' && (
        <DurationView
          predictions={predictions}
          selectedJobs={selectedJobs}
          loading={loading}
          error={error}
        />
      )}
    </div>
  );
}

function DurationView({ predictions, selectedJobs, loading, error }: {
  predictions: Record<string, any>;
  selectedJobs: any[];
  loading: boolean;
  error?: string | null;
}) {
  const totalSelected = selectedJobs.length;
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.ID ?? j.id ?? '');
    return !!predictions[id];
  });
  const predictedCount = predictedJobs.length;

  const durations = predictedJobs
    .map((j) => {
      const id = String(j.ID ?? j.id ?? '');
      const pred = predictions[id];
      return pred?.predicted_completion_days;
    })
    .filter((d): d is number => typeof d === 'number');

  const avgDuration = durations.length > 0 
    ? durations.reduce((a, b) => a + b, 0) / durations.length 
    : undefined;

  const minDuration = durations.length > 0 ? Math.min(...durations) : undefined;
  const maxDuration = durations.length > 0 ? Math.max(...durations) : undefined;

  return (
    <>
      <h3 className="text-sm font-medium text-slate-300 mb-2">Completion Time</h3>

      {loading ? (
        <p className="text-sm text-slate-400">Running predictions…</p>
      ) : error ? (
        <p className="text-sm text-red-300">Error: {error}</p>
      ) : totalSelected === 0 ? (
        <p className="text-sm text-slate-400">Select one or more jobs to run duration predictions.</p>
      ) : predictedCount === 0 ? (
        <p className="text-sm text-slate-400">Click "Predict" to run completion time predictions.</p>
      ) : (
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm text-slate-300">Selected: <strong>{totalSelected}</strong></div>
            <div className="text-sm text-slate-300">Predicted: <strong>{predictedCount}</strong></div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Average</div>
              <div className="text-lg font-semibold text-slate-100">
                {avgDuration != null ? `${avgDuration.toFixed(1)} days` : '—'}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Shortest</div>
              <div className="text-lg font-semibold text-green-400">
                {minDuration != null ? `${minDuration.toFixed(1)} days` : '—'}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Longest</div>
              <div className="text-lg font-semibold text-red-400">
                {maxDuration != null ? `${maxDuration.toFixed(1)} days` : '—'}
              </div>
            </div>
          </div>

          <div className="text-sm text-slate-400 mt-4">
            Duration predictions help estimate job completion timelines based on historical data.
          </div>
        </div>
      )}
    </>
  );
}

function ProfitabilityView({ predictions, selectedJobs, loading, error }: {
  predictions: Record<string, Prediction>;
  selectedJobs: any[];
  loading: boolean;
  error?: string | null;
}) {
  const totalSelected = selectedJobs.length;
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.ID ?? j.id ?? '');
    return !!predictions[id];
  });
  const predictedCount = predictedJobs.length;

  const counts = { High: 0, Medium: 0, Low: 0 } as Record<string, number>;
  let sumConfidence = 0;
  let confCount = 0;

  Object.values(predictions).forEach((p) => {
    if (!p) return;
    const k = p.class ?? null;
    if (k && (k === 'High' || k === 'Medium' || k === 'Low')) counts[k]++;
    if (typeof p.confidence === 'number') {
      sumConfidence += p.confidence;
      confCount += 1;
    }
  });

  const avgConfidence = confCount > 0 ? sumConfidence / confCount : undefined;

  return (
    <>
      <h3 className="text-sm font-medium text-slate-300 mb-2">Profitability</h3>

      {loading ? (
        <p className="text-sm text-slate-400">Running predictions…</p>
      ) : error ? (
        <p className="text-sm text-red-300">Error: {error}</p>
      ) : totalSelected === 0 ? (
        <p className="text-sm text-slate-400">Select one or more jobs to run profitability predictions.</p>
      ) : predictedCount === 0 ? (
        <p className="text-sm text-slate-400">Click "Predict" to run profitability on the selected jobs.</p>
      ) : (
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm text-slate-300">Selected: <strong>{totalSelected}</strong></div>
            <div className="text-sm text-slate-300">Predicted: <strong>{predictedCount}</strong></div>
          </div>

          <div className="space-y-3">
            {/* Chart.js Bar Chart */}
            <div className="bg-slate-800/50 rounded-lg p-4">
              <Bar
                data={{
                  labels: ['Low', 'Medium', 'High'],
                  datasets: [
                    {
                      label: 'Profitability Distribution',
                      data: [counts.Low, counts.Medium, counts.High],
                      backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',   // red for Low
                        'rgba(234, 179, 8, 0.8)',   // yellow for Medium
                        'rgba(34, 197, 94, 0.8)',   // green for High
                      ],
                      borderColor: [
                        'rgb(239, 68, 68)',
                        'rgb(234, 179, 8)',
                        'rgb(34, 197, 94)',
                      ],
                      borderWidth: 1,
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false,
                    },
                    title: {
                      display: false,
                    },
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      ticks: {
                        color: 'rgb(148, 163, 184)', // slate-400
                        stepSize: 1,
                      },
                      grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                      },
                    },
                    x: {
                      ticks: {
                        color: 'rgb(148, 163, 184)',
                      },
                      grid: {
                        display: false,
                      },
                    },
                  },
                }}
                height={200}
              />
            </div>

            <div className="pt-4 border-t border-slate-800 mt-4">
              <div className="text-sm text-slate-300">Counts</div>
              <div className="flex gap-4 mt-2 text-sm text-slate-200">
                <div>High: <strong>{counts.High}</strong></div>
                <div>Medium: <strong>{counts.Medium}</strong></div>
                <div>Low: <strong>{counts.Low}</strong></div>
              </div>

              <div className="text-sm text-slate-300 mt-3">Average confidence: <strong>{avgConfidence != null ? `${(avgConfidence * 100).toFixed(1)}%` : '—'}</strong></div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
