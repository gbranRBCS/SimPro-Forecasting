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
import type { Job, Prediction } from '../../features/jobs/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

/**
PREDICTION PANEL COMPONENT
--------------------------
Displays the results of our ML models.
It handles two modes:
 1. Profitability: Classification (Low/Medium/High) - visualized as a distribution chart.
 2. Duration: Regression (Days) - visualized as summary stats (Avg, Min, Max).

Reacts to the user's selection in the main table. If 5 jobs are selected, it shows predictions for those 5 jobs.
 */

interface PredictionPanelProps {
  predictionType: 'profitability' | 'duration' | 'none';
  selectedJobs: Job[];
  predictions: Record<string, Prediction>;
  loading: boolean;
  error?: string | null;
  onPredict: () => void;
  onPredictionTypeChange: (t: 'profitability' | 'duration' | 'none') => void;
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

  // Why this matters: We only enable the "Predict" button if items are actually selected.
  // This prevents api waste.
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
            onChange={(e) => onPredictionTypeChange(e.target.value as 'profitability' | 'duration' | 'none')}
            className="px-2 py-1 bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-100 focus:ring-2 focus:ring-blue-500/50 focus:outline-none"
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
  predictions: Record<string, Prediction>;
  selectedJobs: Job[];
  loading: boolean;
  error?: string | null;
}) {
  const totalSelected = selectedJobs.length;
  
  // Filter for jobs that we actually have a result for.
  // We match based on normalized IDs
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.id ?? '');
    return !!predictions[id];
  });
  const predictedCount = predictedJobs.length;

  // Extract pure numbers for stats
  const durations = predictedJobs
    .map((j) => {
      const id = String(j.id ?? '');
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
          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-slate-300">Selected: <strong>{totalSelected}</strong></div>
            <div className="text-sm text-slate-300">Predicted: <strong>{predictedCount}</strong></div>
          </div>

          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-slate-800/50 rounded-lg p-6 flex flex-col items-center justify-center text-center border border-slate-700/50">
              <div className="text-sm font-medium text-slate-400 mb-2 uppercase tracking-wide">Average Duration</div>
              <div className="text-3xl font-bold text-slate-100">
                {avgDuration != null ? `${avgDuration.toFixed(1)}` : '—'}
                <span className="text-lg font-normal text-slate-400 ml-1">days</span>
              </div>
            </div>
            
            <div className="bg-slate-800/50 rounded-lg p-6 flex flex-col items-center justify-center text-center border border-green-900/20">
              <div className="text-sm font-medium text-slate-400 mb-2 uppercase tracking-wide">Shortest Estimate</div>
              <div className="text-3xl font-bold text-green-400">
                {minDuration != null ? `${minDuration.toFixed(1)}` : '—'}
                <span className="text-lg font-normal text-slate-500/80 ml-1">days</span>
              </div>
            </div>
            
            <div className="bg-slate-800/50 rounded-lg p-6 flex flex-col items-center justify-center text-center border border-red-900/20">
              <div className="text-sm font-medium text-slate-400 mb-2 uppercase tracking-wide">Longest Estimate</div>
              <div className="text-3xl font-bold text-red-400">
                {maxDuration != null ? `${maxDuration.toFixed(1)}` : '—'}
                <span className="text-lg font-normal text-slate-500/80 ml-1">days</span>
              </div>
            </div>
          </div>

          <div className="bg-blue-900/10 border border-blue-900/20 rounded-lg p-4 text-sm text-blue-200/80">
            <p><strong>Note:</strong> Completion time predictions utilize historical job data to estimate job duration. These are estimates only.</p>
          </div>
        </div>
      )}
    </>
  );
}

function ProfitabilityView({ predictions, selectedJobs, loading, error }: {
  predictions: Record<string, Prediction>;
  selectedJobs: Job[];
  loading: boolean;
  error?: string | null;
}) {
  const totalSelected = selectedJobs.length;
  
  // Match results to selected jobs
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.id ?? '');
    return !!predictions[id];
  });
  const predictedCount = predictedJobs.length;

  const counts = { High: 0, Medium: 0, Low: 0 } as Record<string, number>;
  let sumConfidence = 0;
  let confCount = 0;

  // Collect results for the chart
  predictedJobs.forEach((job) => {
    const id = String(job.id ?? '');
    const p = predictions[id];

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

          <div className="flex flex-col lg:flex-row gap-8 items-start">
            {/* Chart Section */}
            <div className="flex-1 w-full bg-slate-800/50 rounded-lg p-4 min-h-[300px]">
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
              />
            </div>

            {/* Stats Section */}
            <div className="w-full lg:w-72 pt-4 lg:pt-0 border-t lg:border-t-0 lg:border-l border-slate-800 lg:pl-8 flex flex-col gap-6 justify-center h-full min-h-[300px]">
              <div>
                <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-3">Distribution</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 rounded bg-slate-800/30 border border-green-900/30">
                    <span className="text-green-400 font-medium">High</span>
                    <span className="text-xl font-bold text-slate-100">{counts.High}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 rounded bg-slate-800/30 border border-yellow-900/30">
                    <span className="text-yellow-400 font-medium">Medium</span>
                    <span className="text-xl font-bold text-slate-100">{counts.Medium}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 rounded bg-slate-800/30 border border-red-900/30">
                    <span className="text-red-400 font-medium">Low</span>
                    <span className="text-xl font-bold text-slate-100">{counts.Low}</span>
                  </div>
                </div>
              </div>

              <div>
                 <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Confidence</h4>
                 <div className="text-3xl font-bold text-blue-400">
                    {avgConfidence != null ? `${(avgConfidence * 100).toFixed(1)}%` : '—'}
                 </div>
                 <p className="text-xs text-slate-500 mt-1">Average model confidence score</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
