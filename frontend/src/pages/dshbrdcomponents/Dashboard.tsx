import { useState, useEffect, useRef } from 'react';
import { Briefcase } from '../../components/icons';
import {
  syncJobs,
  getJobs,
  predictProfitability,
  predictDuration,
} from '../../features/jobs/api';
import type { 
  Job, 
  Prediction, 
  SyncParams 
} from '../../features/jobs/api';

import { Toolbar } from './Toolbar';
import { JobsTable } from './JobsTable';
import { PredictionPanel } from './PredictionPanel';
import { Pagination } from './Pagination';
import { StatusAlert, SyncProgressCard } from './StatusCards';

// --- Types & Interfaces ---

/**
State for the filter toolbar inputs.
Values are strings to match HTML input elements.
 */
type FilterState = {
  fromDate: string;     // ISO Date string (YYYY-MM-DD)
  toDate: string;       // ISO Date string (YYYY-MM-DD)
  minRevenue: string;   // Numeric string input
  maxRevenue: string;   // Numeric string input
  order: 'asc' | 'desc';
  limit: string;        // Numeric string input
};

type PaginationState = {
  page: number;         // Current page (1-indexed)
  pageSize: number;     // Items per page
  totalPages: number;   // Total available pages
};

// Aggregated statistics from a set of predictions.
type PredictionSummaryData = {
  highCount: number;
  mediumCount: number;
  lowCount: number;
  count: number;
  avgConfidence?: number;
};

type LoadOptions = {
  // If provided, use these jobs instead of fetching from API
  preloadedJobs?: Job[]; 
  // Message to display on successful load (e.g. "Sync complete")
  statusPrefix?: string; 
};

type StatusMessage = {
  text: string;
  type: 'success' | 'error' | 'info';
};


// --- Helper Functions ---

/**
Extracts a numeric revenue value from a job object.
Handles different formats SimPRO might return (direct number, string, or nested object).
 */
function getRevenueValue(job: Job): number | null {
  if (!job) return null;

  if (typeof job.revenue === 'number' && Number.isFinite(job.revenue)) {
    return job.revenue;
  }

  return null;
}

/**
Filters and sorts jobs client-side.
Used when we have preloaded data (e.g., after a sync) or are using cached data fallback.
 */
function filterJobsLocally(
  jobs: Job[],
  filters: FilterState,
  page: number,
  pageSize: number,
) {
  // Ensure valid pagination params
  const safePage = page && page > 0 ? page : 1;
  const safePageSize = pageSize && pageSize > 0 ? pageSize : jobs.length || 1;

  let filtered = Array.isArray(jobs) ? [...jobs] : [];

  // Filter: Min Revenue
  if (filters.minRevenue) {
    const min = Number(filters.minRevenue);
    if (!Number.isNaN(min)) {
      filtered = filtered.filter((job) => {
        const revenue = getRevenueValue(job);
        return revenue != null && revenue >= min;
      });
    }
  }

  // Filter: Max Revenue
  if (filters.maxRevenue) {
    const max = Number(filters.maxRevenue);
    if (!Number.isNaN(max)) {
      filtered = filtered.filter((job) => {
        const revenue = getRevenueValue(job);
        return revenue != null && revenue <= max;
      });
    }
  }

  // Sort: Revenue
  filtered.sort((a, b) => {
    const revA = getRevenueValue(a);
    const revB = getRevenueValue(b);
    
    // Sort logic handles nulls by pushing them to the end
    const fallback = filters.order === 'desc' ? -Infinity : Infinity;
    const valA = revA != null ? revA : fallback;
    const valB = revB != null ? revB : fallback;

    if (valA === valB) return 0;
    
    // Toggle direction
    if (filters.order === 'desc') return valA > valB ? -1 : 1;
    return valA > valB ? 1 : -1;
  });

  // Paginate
  const start = (safePage - 1) * safePageSize;
  const paged = start >= 0 
    ? filtered.slice(start, start + safePageSize) 
    : filtered.slice(0, safePageSize);

  const total = filtered.length;
  const totalPages = safePageSize > 0 
    ? Math.max(1, Math.ceil(total / safePageSize)) 
    : 1;

  return { jobs: paged, total, totalPages };
}

// --- Main Component ---

export function Dashboard() {
  // -- State: UI & Data --
  const [filters, setFilters] = useState<FilterState>({
    fromDate: '',
    toDate: '',
    minRevenue: '',
    maxRevenue: '',
    order: 'desc',
    limit: '',
  });

  const [pagination, setPagination] = useState<PaginationState>({
    page: 1,
    pageSize: 20,
    totalPages: 1,
  });

  const [jobs, setJobs] = useState<Job[]>([]);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);
  
  // -- State: Loading Flags --
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  
  // -- State: Predictions --
  const [, setPredictionSummary] = useState<PredictionSummaryData | null>(null);
  const [predictionType, setPredictionType] = useState<'profitability' | 'duration' |'none'>('profitability');
  
  const [selectedJobIds, setSelectedJobIds] = useState<Array<string>>([]);
  
  // Profitability Prediction State
  const [profitabilityPredictions, setProfitabilityPredictions] = useState<Record<string, Prediction>>({});
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  
  // Duration Prediction State
  const [durationPredictions, setDurationPredictions] = useState<Record<string, Prediction>>({});
  const [, setDurationPredictionLoading] = useState(false); // Unused currently
  const [, setDurationPredictionError] = useState<string | null>(null); // Unused currently

  // -- Refs --
  // Cache fetched jobs to allow client-side filtering without re-fetching
  const cachedJobsRef = useRef<Job[]>([]);


  // -- Methods --

  const updateCache = (items: Job[] | undefined | null) => {
    if (Array.isArray(items) && items.length > 0) {
      cachedJobsRef.current = items;
    }
  };

  /**
  Main data loading function.
  Can load from API or use preloaded data (e.g. from Sync result).
  */
  const loadJobs = async (
    newPage?: number,
    newPageSize?: number,
    options?: LoadOptions,
  ) => {
    const pageToLoad = newPage ?? pagination.page;
    const sizeToLoad = newPageSize ?? pagination.pageSize;
    const hasPreloaded = Array.isArray(options?.preloadedJobs);

    // Only set loading to true if we are actually fetching over network
    if (!hasPreloaded) {
      setIsLoading(true);
    }

    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      let jobsData: Job[] = [];
      let totalPages = 1;
      let totalCount = 0;
      let usedCachedFallback = false;

      // Case A: Use preloaded data (e.g. from Sync)
      if (hasPreloaded && options?.preloadedJobs) {
        updateCache(options.preloadedJobs);
        const local = filterJobsLocally(
          options.preloadedJobs,
          filters,
          pageToLoad,
          sizeToLoad,
        );
        jobsData = local.jobs;
        totalPages = local.totalPages;
        totalCount = local.total;
      } 
      // Case B: Fetch from API
      else {
        const params: any = {
          sortField: 'revenue',
          order: filters.order,
          page: pageToLoad,
          pageSize: sizeToLoad,
        };

        if (filters.minRevenue) {
          const min = Number(filters.minRevenue);
          if (!Number.isNaN(min)) params.minRevenue = min;
        }
        if (filters.maxRevenue) {
          const max = Number(filters.maxRevenue);
          if (!Number.isNaN(max)) params.maxRevenue = max;
        }
        if (filters.limit) {
          const limit = Number(filters.limit);
          if (!Number.isNaN(limit)) params.limit = limit;
        }

        const data = await getJobs(params);
        const fetchedJobs = Array.isArray(data?.jobs) ? data.jobs : [];
        const reportedTotal =
          typeof data?.total === 'number' ? data.total : fetchedJobs.length;

        // Smart caching:
        // If we fetched new data, use it & update cache.
        if (fetchedJobs.length > 0) {
          updateCache(fetchedJobs);
          jobsData = fetchedJobs;
          totalPages = data?.totalPages ?? 1;
          totalCount = reportedTotal;
        } 
        // If API returned nothing but we have cache (e.g. offline/error fallback scenario)
        else if (cachedJobsRef.current.length > 0) {
          const fallback = filterJobsLocally(
            cachedJobsRef.current,
            filters,
            pageToLoad,
            sizeToLoad,
          );
          jobsData = fallback.jobs;
          totalPages = fallback.totalPages;
          totalCount = fallback.total;
          usedCachedFallback = true;
        } 
        // No data anywhere
        else {
          jobsData = [];
          totalPages = 1;
          totalCount = 0;
        }
      }

      setJobs(jobsData);
      setPagination((prev) => ({
        ...prev,
        page: pageToLoad,
        pageSize: sizeToLoad,
        totalPages: totalPages > 0 ? totalPages : 1,
      }));

      // Construct a user-friendly status message
      const showingCount = jobsData.length;
      const totalForMessage = totalCount > 0 ? totalCount : showingCount;
      let messageText: string;

      if (options?.statusPrefix) {
        // "Sync complete · Showing 50 of 200 jobs"
        const countText = totalForMessage > showingCount
          ? `${showingCount} of ${totalForMessage}`
          : `${showingCount}`;
        messageText = `${options.statusPrefix} · Showing ${countText} jobs`;
      } else if (usedCachedFallback) {
        // "Showing cached jobs · 50 of 200 jobs"
        const countText = totalForMessage > showingCount
          ? `${showingCount} of ${totalForMessage}`
          : `${showingCount}`;
        messageText = `Showing cached jobs · ${countText} jobs`;
      } else if (totalForMessage > showingCount) {
        messageText = `Loaded ${showingCount} of ${totalForMessage} jobs`;
      } else {
        messageText = `Loaded ${showingCount} jobs`;
      }

      setStatusMessage({
        text: messageText,
        type: 'success',
      });

    } catch (error: any) {
      const errorMsg =
        error?.response?.data?.error ||
        error?.message ||
        'Failed to load jobs';
      setStatusMessage({
        text: errorMsg,
        type: 'error',
      });
      setJobs([]);
    } finally {
      setIsLoading(false);
    }
  };

  /**
  Triggers a sync with the backend SimPRO service.
   */
  const handleSync = async (mode: 'update' | 'full' = 'update') => {
    setIsSyncing(true);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      // Prepare Sync Params
      const params: SyncParams = { mode };
      
      if (mode === 'update') {
        if (filters.fromDate) params.from = filters.fromDate;
        if (filters.toDate) params.to = filters.toDate;
      }

      const data = await syncJobs(params);
      
      // If sync returns jobs immediately, use them to populate the table (avoids a second call of loadJobs)
      const syncedJobs = Array.isArray(data?.jobs) ? data.jobs as Job[] : undefined;

      await loadJobs(1, pagination.pageSize, {
        preloadedJobs: syncedJobs,
        statusPrefix: data?.message || 'Sync complete',
      });

    } catch (error: any) {
      const errorMsg =
        error?.response?.data?.error || error?.message || 'Sync failed';
      setStatusMessage({
        text: errorMsg,
        type: 'error',
      });
    } finally {
      setIsSyncing(false);
    }
  };

  /**
  Runs the profitability ML model on selected jobs.
   */
  const runProfitabilityPrediction = async () => {
    setPredictionLoading(true);
    setPredictionError(null);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      if (!selectedJobIds || selectedJobIds.length === 0) {
        throw new Error('Select one or more jobs to run predictions');
      }

      const response = await predictProfitability({ jobIds: selectedJobIds });
      const preds: Prediction[] = response?.predictions ?? [];

      // Normalize into map keyed by specific Job ID string
      const map: Record<string, Prediction> = {};
      preds.forEach((p) => {
        const key = p.jobId == null ? '' : String(p.jobId);
        map[key] = p;
      });

      setProfitabilityPredictions((prev) => ({ ...prev, ...map }));

      // --- Calculate Summary Stats ---
      
      // Determine class (High/Medium/Low) - prefer model output, fallback to heuristic
      const classes = preds
        .map((p) => {
          return p.class ?? null;
        })
        .filter(Boolean) as string[];

      const highCount = classes.filter((c) => c === 'High').length;
      const mediumCount = classes.filter((c) => c === 'Medium').length;
      const lowCount = classes.filter((c) => c === 'Low').length;
      const count = classes.length;

      // Calculate confidence average
      const confidences = preds
        .map((p) => (typeof p.confidence === 'number' ? p.confidence : typeof p.probability === 'number' ? p.probability : null))
        .filter((v): v is number => typeof v === 'number');

      const avgConfidence = confidences.length > 0
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : undefined;

      setPredictionSummary({ highCount, mediumCount, lowCount, count, avgConfidence });
      
      setStatusMessage({ 
        text: `Predicted profitability for ${count} jobs`, 
        type: 'success' 
      });

    } catch (error: any) {
      setPredictionError(error?.message || 'Prediction failed');
      setStatusMessage({ text: error?.message || 'Prediction failed', type: 'error' });
    } finally {
      setPredictionLoading(false);
    }
  };

  /**
   * Runs the duration (completion time) ML model on selected jobs.
   */
  const runDurationPrediction = async () => {
    setPredictionLoading(true);
    setPredictionError(null);
    setStatusMessage(null);

    try {
      if (!selectedJobIds || selectedJobIds.length === 0) {
        throw new Error('Select one or more jobs to run predictions');
      }

      const response = await predictDuration({ jobIds: selectedJobIds });
      const preds = response?.predictions ?? [];

      const map: Record<string, Prediction> = {};
      preds.forEach((p: Prediction) => {
        const key = p.jobId == null ? '' : String(p.jobId);
        map[key] = p;
      });

      setDurationPredictions((prev) => ({ ...prev, ...map }));

      setStatusMessage({ 
        text: `Predicted completion time for ${preds.length} jobs`, 
        type: 'success' 
      });
    } catch (error: any) {
      setPredictionError(error?.message || 'Duration prediction failed');
      setStatusMessage({ 
        text: error?.message || 'Duration prediction failed', 
        type: 'error' 
      });
    } finally {
      setPredictionLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    setPagination((prev) => ({ ...prev, page: newPage }));
  };

  const handlePageSizeChange = (newPageSize: number) => {
    setPagination((prev) => ({ ...prev, page: 1, pageSize: newPageSize }));
  };

  // Reload jobs when pagination changes
  useEffect(() => {
    if (pagination.pageSize > 0) {
      loadJobs(pagination.page, pagination.pageSize);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pagination.page, pagination.pageSize]);

  // Handler for the "Predict" button click
  const handlePredict = async () => {
    if (predictionType === 'profitability') {
      await runProfitabilityPrediction();
    } else if (predictionType === 'duration') {
      await runDurationPrediction();
    } else {
      console.warn('No prediction type selected');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <header className="bg-slate-900/50 border-b border-slate-800 px-6 py-6">
        <div className="flex items-center gap-3">
          <Briefcase className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-2xl font-bold text-slate-100">
              Jobs Dashboard
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              SimPRO Business Data Forecasting
            </p>
          </div>
        </div>
      </header>

      <Toolbar
        filters={filters}
        onFilterChange={(nextFilters) =>
          setFilters((prev) => ({ ...prev, ...nextFilters }))
        }
        onSync={handleSync}
        onLoadJobs={() => loadJobs()}
        isSyncing={isSyncing}
        isLoading={isLoading}
      />

      <main className="px-6 py-6">
        <div className="flex flex-col gap-6">
          {/* Top Panel - Predictions and Visualizations */}
          <div className="w-full">
            <PredictionPanel
              predictionType={predictionType}
              selectedJobs={jobs.filter((j) => selectedJobIds.includes(String(j.id ?? '')))}
              predictions={predictionType === 'profitability' ? profitabilityPredictions : durationPredictions}
              loading={predictionLoading}
              error={predictionError}
              onPredict={handlePredict}
              onPredictionTypeChange={(t: 'profitability' | 'duration' | 'none') => setPredictionType(t)}
            />
          </div>

          {/* Bottom Panel - Jobs Table and Controls */}
          <div className="space-y-4 w-full">
            {statusMessage && (
              <StatusAlert message={statusMessage.text} type={statusMessage.type} />
            )}

            {isSyncing && <SyncProgressCard />}

            <JobsTable
              jobs={jobs}
              isLoading={isLoading}
              selectedJobIds={selectedJobIds}
              onSelectionChange={(ids) => setSelectedJobIds(ids)}
              predictions={predictionType === 'duration' ? durationPredictions : profitabilityPredictions}
              predictionType={predictionType}
            />

            <Pagination
              pagination={pagination}
              onPageChange={handlePageChange}
              onPageSizeChange={handlePageSizeChange}
              disabled={isLoading || isSyncing}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
