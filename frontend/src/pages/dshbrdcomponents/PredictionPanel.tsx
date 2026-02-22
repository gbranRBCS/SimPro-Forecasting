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

// register the chart components we need
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// this component displays the results of our machine learning models
// it switches between two modes: profitability (classification) and duration (regression)

interface PredictionPanelProps {
  predictionType: 'profitability' | 'duration' | 'none';
  selectedJobs: Job[];
  predictions: Record<string, Prediction>;
  loading: boolean;
  error?: string | null;
  onPredict: () => void;
  onPredictionTypeChange: (t: 'profitability' | 'duration' | 'none') => void;
}

export function PredictionPanel({ 
  predictionType, 
  selectedJobs, 
  predictions, 
  loading, 
  error, 
  onPredict, 
  onPredictionTypeChange 
}: PredictionPanelProps) {
  
  // if no model is selected, we show a simple placeholder
  if (predictionType === 'none') {
    return (
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-slate-100 mb-2">Predictions</h2>
        <p className="text-sm text-slate-400">No prediction model selected.</p>
      </div>
    );
  }

  // we only enable the predict button if items are actually selected
  // this prevents sending empty requests to the api
  const totalSelected = selectedJobs.length;

  return ( 
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      
      {/* header with controls */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-slate-100">Predictions</h2>
        
        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-400 font-medium">Model:</label>
          <select
            value={predictionType}
            onChange={(e) => onPredictionTypeChange(e.target.value as 'profitability' | 'duration' | 'none')}
            className="px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-100 focus:ring-2 focus:ring-blue-500/50 focus:outline-none transition-colors"
          >
            <option value="profitability">Profitability</option>
            <option value="duration">Duration</option>
          </select>
          
          <button
            onClick={onPredict}
            disabled={loading || totalSelected === 0}
            className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Predict
          </button>
        </div>
      </div>

      {/* show the correct view based on the selected type */}
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
  
  // filter for jobs that have result, match based on IDs
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.id);
    return !!predictions[id];
  });
  
  const predictedCount = predictedJobs.length;

  // extract just the numbers for calculating stats
  const durations = predictedJobs
    .map((j) => {
      const id = String(j.id);
      return predictions[id]?.predicted_completion_days;
    })
    .filter((d) => typeof d === 'number') as number[];

  // calculate averages and min/max
  let avgDuration = 0;
  let minDuration = 0;
  let maxDuration = 0;

  if (durations.length > 0) {
    const sum = durations.reduce((a, b) => a + b, 0);
    avgDuration = sum / durations.length;
    minDuration = Math.min(...durations);
    maxDuration = Math.max(...durations);
  }

  // Handle various states (loading, error, empty)
  if (loading) return <p className="text-sm text-slate-400 animate-pulse">Running predictions…</p>;
  if (error) return <p className="text-sm text-red-400">Error: {error}</p>;
  if (totalSelected === 0) return <p className="text-sm text-slate-400">Select jobs to run duration predictions.</p>;
  if (predictedCount === 0) return <p className="text-sm text-slate-400">Click "Predict" to see results.</p>;

  return (
    <div>
      <div className="flex items-center gap-4 mb-6 text-sm text-slate-400">
        <div>Selected: <span className="text-slate-200">{totalSelected}</span></div>
        <div>Predicted: <span className="text-slate-200">{predictedCount}</span></div>
      </div>

      {/* simplified grid for the stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <StatsCard label="Average Duration" value={avgDuration} color="text-slate-100" />
        <StatsCard label="Shortest Estimate" value={minDuration} color="text-green-400" />
        <StatsCard label="Longest Estimate" value={maxDuration} color="text-red-400" />
      </div>

      <div className="bg-blue-900/10 border border-blue-900/20 rounded-lg p-4 text-sm text-blue-200/80">
        <p><strong>Note:</strong> These are estimates based on historical job data.</p>
      </div>
    </div>
  );
}

// simple helper component for the stats cards to avoid repetition
function StatsCard({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div className="bg-slate-800/50 rounded-lg p-5 flex flex-col items-center justify-center text-center border border-slate-700/50">
      <div className="text-xs font-medium text-slate-400 mb-1 uppercase tracking-wide">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {value.toFixed(1)} <span className="text-sm font-normal text-slate-500 ml-1">days</span>
      </div>
    </div>
  );
}

function ProfitabilityView({ predictions, selectedJobs, loading, error }: {
  predictions: Record<string, Prediction>;
  selectedJobs: Job[];
  loading: boolean;
  error?: string | null;
}) {
  const totalSelected = selectedJobs.length;
  
  // filter to find jobs that have been predicted
  const predictedJobs = selectedJobs.filter((j) => {
    const id = String(j.id);
    return !!predictions[id];
  });
  
  const predictedCount = predictedJobs.length;

  // counters for each category
  let highCount = 0;
  let mediumCount = 0;
  let lowCount = 0;
  
  let totalConfidence = 0;
  let confidenceCount = 0;

  // loop through results to populate counters and confidence sum
  predictedJobs.forEach((job) => {
    const id = String(job.id);
    const prediction = predictions[id];
    
    if (prediction) {
      if (prediction.class === 'High') highCount++;
      if (prediction.class === 'Medium') mediumCount++;
      if (prediction.class === 'Low') lowCount++;
      
      if (typeof prediction.confidence === 'number') {
        totalConfidence += prediction.confidence;
        confidenceCount++;
      }
    }
  });

  const avgConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;

  // Handle states
  if (loading) return <p className="text-sm text-slate-400 animate-pulse">Running predictions…</p>;
  if (error) return <p className="text-sm text-red-400">Error: {error}</p>;
  if (totalSelected === 0) return <p className="text-sm text-slate-400">Select jobs to run profitability predictions.</p>;
  if (predictedCount === 0) return <p className="text-sm text-slate-400">Click "Predict" to see results.</p>;

  // data object for the chart
  const chartData = {
    labels: ['Low', 'Medium', 'High'],
    datasets: [
      {
        label: 'Jobs',
        data: [lowCount, mediumCount, highCount],
        backgroundColor: [
          'rgba(239, 68, 68, 0.8)',   // red
          'rgba(234, 179, 8, 0.8)',   // yellow
          'rgba(34, 197, 94, 0.8)',   // green
        ],
        borderRadius: 4,
      },
    ],
  };

  return (
    <div>
      <div className="flex items-center gap-4 mb-4 text-sm text-slate-400">
        <div>Selected: <span className="text-slate-200">{totalSelected}</span></div>
        <div>Predicted: <span className="text-slate-200">{predictedCount}</span></div>
      </div>

      <div className="flex flex-col lg:flex-row gap-8">
        
        {/* chart container */}
        <div className="flex-1 min-h-[250px] bg-slate-800/30 rounded-lg p-4">
          <Bar
            data={chartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                x: { grid: { display: false } }
              }
            }}
          />
        </div>

        {/* breakdown stats */}
        <div className="w-full lg:w-64 flex flex-col justify-center gap-4">
          <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Breakdown</h4>
          
          <div className="space-y-2">
            <BreakdownRow label="High" count={highCount} color="text-green-400" />
            <BreakdownRow label="Medium" count={mediumCount} color="text-yellow-400" />
            <BreakdownRow label="Low" count={lowCount} color="text-red-400" />
          </div>

          <div className="pt-4 border-t border-slate-700/50 mt-2">
             <div className="text-sm text-slate-400 mb-1">Avg Confidence</div>
             <div className="text-3xl font-bold text-blue-400">
                {(avgConfidence * 100).toFixed(1)}%
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownRow({ label, count, color }: { label: string, count: number, color: string }) {
  return (
    <div className="flex justify-between items-center p-2 rounded bg-slate-800/40 border border-slate-700/30">
      <span className={`font-medium ${color}`}>{label}</span>
      <span className="font-bold text-slate-200">{count}</span>
    </div>
  );
}
