import { useState, useEffect, useRef } from 'react';
import { Briefcase } from '../../components/icons';
import { syncJobs, getJobs, predict } from '../../features/jobs/api';
import { Toolbar } from './Toolbar';
import { JobsTable } from './JobsTable';
import { Pagination } from './Pagination';
import {
  StatusAlert,
  SyncProgressCard,
  PredictionSummary,
} from './StatusCards';

type ApiJob = Record<string, any>;

type Prediction = {
  jobId: string | number;
  class?: string;
  confidence?: number;
  probability?: number;
  margin_est?: number;
  [key: string]: any;
};

type FilterState = {
  fromDate: string;
  toDate: string;
  minRevenue: string;
  maxRevenue: string;
  order: 'asc' | 'desc';
  limit: string;
};

type PaginationState = {
  page: number;
  pageSize: number;
  totalPages: number;
};

type PredictionSummaryData = {
  highCount: number;
  mediumCount: number;
  lowCount: number;
  count: number;
  avgConfidence?: number;
};

type LoadOptions = {
  preloadedJobs?: ApiJob[];
  statusPrefix?: string;
};

function getRevenueValue(job: ApiJob): number | null {
  if (!job) return null;

  const direct = job.revenue;
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }

  const incTax = job?.Total?.IncTax;
  if (typeof incTax === 'number' && Number.isFinite(incTax)) {
    return incTax;
  }
  if (typeof incTax === 'string') {
    const parsed = Number(incTax);
    if (Number.isFinite(parsed)) return parsed;
  }

  return null;
}

function filterJobsLocally(
  jobs: ApiJob[],
  filters: FilterState,
  page: number,
  pageSize: number,
) {
  const safePage = page && page > 0 ? page : 1;
  const safePageSize = pageSize && pageSize > 0 ? pageSize : jobs.length || 1;

  let filtered = Array.isArray(jobs) ? [...jobs] : [];

  if (filters.minRevenue) {
    const min = Number(filters.minRevenue);
    if (!Number.isNaN(min)) {
      filtered = filtered.filter((job) => {
        const revenue = getRevenueValue(job);
        return revenue != null && revenue >= min;
      });
    }
  }

  if (filters.maxRevenue) {
    const max = Number(filters.maxRevenue);
    if (!Number.isNaN(max)) {
      filtered = filtered.filter((job) => {
        const revenue = getRevenueValue(job);
        return revenue != null && revenue <= max;
      });
    }
  }

  filtered.sort((a, b) => {
    const revA = getRevenueValue(a);
    const revB = getRevenueValue(b);
    const fallback = filters.order === 'desc' ? -Infinity : Infinity;
    const valA = revA != null ? revA : fallback;
    const valB = revB != null ? revB : fallback;

    if (valA === valB) return 0;
    if (filters.order === 'desc') return valA > valB ? -1 : 1;
    return valA > valB ? 1 : -1;
  });

  const start = (safePage - 1) * safePageSize;
  const paged = start >= 0 ? filtered.slice(start, start + safePageSize) : filtered.slice(0, safePageSize);

  const total = filtered.length;
  const totalPages = safePageSize > 0 ? Math.max(1, Math.ceil(total / safePageSize)) : 1;

  return { jobs: paged, total, totalPages };
}

export function Dashboard() {
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

  const [jobs, setJobs] = useState<ApiJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{
    text: string;
    type: 'success' | 'error' | 'info';
  } | null>(null);
  const [predictionSummary, setPredictionSummary] =
    useState<PredictionSummaryData | null>(null);
  const cachedJobsRef = useRef<ApiJob[]>([]);

  const updateCache = (items: ApiJob[] | undefined | null) => {
    if (Array.isArray(items) && items.length > 0) {
      cachedJobsRef.current = items;
    }
  };

   const loadJobs = async (
    newPage?: number,
    newPageSize?: number,
    options?: LoadOptions,
  ) => {
    const pageToLoad = newPage ?? pagination.page;
    const sizeToLoad = newPageSize ?? pagination.pageSize;
    const hasPreloaded = Array.isArray(options?.preloadedJobs);

    if (!hasPreloaded) {
      setIsLoading(true);
    } else {
      setIsLoading(false);
    }

    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      let jobsData: ApiJob[] = [];
      let totalPages = 1;
      let totalCount = 0;
      let usedCachedFallback = false;

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
      } else {
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

        if (fetchedJobs.length > 0) {
          updateCache(fetchedJobs);
          jobsData = fetchedJobs;
          totalPages = data?.totalPages ?? 1;
          totalCount = reportedTotal;
        } else if (cachedJobsRef.current.length > 0) {
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
        } else {
          jobsData = fetchedJobs;
          totalPages = data?.totalPages ?? 1;
          totalCount = reportedTotal;
        }
      }

      setJobs(jobsData);
      setPagination((prev) => ({
        ...prev,
        page: pageToLoad,
        pageSize: sizeToLoad,
        totalPages: totalPages > 0 ? totalPages : 1,
      }));

      const showingCount = jobsData.length;
      const totalForMessage =
        totalCount > 0 ? totalCount : showingCount;

      let messageText: string;
      if (options?.statusPrefix) {
        const countText =
          totalForMessage > showingCount
            ? `${showingCount} of ${totalForMessage}`
            : `${showingCount}`;
        messageText = `${options.statusPrefix} · Showing ${countText} jobs`;
      } else if (usedCachedFallback) {
        const countText =
          totalForMessage > showingCount
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

  const handleSync = async (mode: 'update' | 'full' = 'update') => {
    setIsSyncing(true);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      const params: { from?: string; to?: string; mode: 'update' | 'full' } = {
        mode,
      };
      if (mode === 'update') {
        if (filters.fromDate) params.from = filters.fromDate;
        if (filters.toDate) params.to = filters.toDate;
      }

      const data = await syncJobs(params);
      const syncedJobs = Array.isArray(data?.jobs) ? data.jobs : undefined;

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

  const handlePredict = async () => {
    setIsLoading(true);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
  // send currently loaded jobs in request body for prediction
      const seenIds = new Set<string>();
      const jobIds = jobs
        .map((job) => {
          const id = job?.ID ?? job?.id ?? null;
          if (id === null || id === undefined) return null;
          const key = String(id);
          if (seenIds.has(key)) return null;
          seenIds.add(key);
          return id;
        })
        .filter((id): id is string | number => id !== null);

      if (jobIds.length === 0) {
        throw new Error('No job identifiers available to request predictions');
      }

      const response = await predict({ jobIds });
      const preds: Prediction[] = response?.predictions ?? [];

      // map predictions back to jobs using uppercase ID field
      const byId = new Map(preds.map((p) => [p.jobId, p]));
      const updatedJobs = jobs.map((j) => {
        const p = byId.get(j.ID) as Prediction | undefined;
        if (!p) return j;

        // determine class from prediction (trained model or fallback)
        const klass: string | null =
          (p.class as string) ??
          (typeof p.margin_est === 'number'
            ? p.margin_est >= 0.1
              ? 'High'
              : p.margin_est >= 0.03
              ? 'Medium'
              : 'Low'
            : null);

        // determine score type and value
        const scoreType =
          typeof p.confidence === 'number'
            ? 'confidence'
            : typeof p.probability === 'number'
            ? 'probability'
            : typeof p.margin_est === 'number'
            ? 'margin'
            : null;

        const score =
          (typeof p.confidence === 'number' && p.confidence) ||
          (typeof p.probability === 'number' && p.probability) ||
          (typeof p.margin_est === 'number' && p.margin_est) ||
          null;

        return {
          ...j,
          profitability: {
            class:
              klass ??
              j.profitability?.class ??
              j.profitability_class ??
              null,
            score,
            scoreType,
          },
        };
      });

      setJobs(updatedJobs);

      // calculate summary statistics
      const getClass = (p: any): string | null => {
        if (p?.class) return p.class;
        if (typeof p?.margin_est === 'number') {
          return p.margin_est >= 0.1
            ? 'High'
            : p.margin_est >= 0.03
            ? 'Medium'
            : 'Low';
        }
        return null;
      };

      const classes = preds
        .map((p: any) => getClass(p))
        .filter(Boolean) as string[];
      const highCount = classes.filter((c) => c === 'High').length;
      const mediumCount = classes.filter((c) => c === 'Medium').length;
      const lowCount = classes.filter((c) => c === 'Low').length;
      const count = classes.length;

      const confidences = preds
        .map((p: any) =>
          typeof p.confidence === 'number'
            ? p.confidence
            : typeof p.probability === 'number'
            ? p.probability
            : null
        )
        .filter((x: number | null) => typeof x === 'number') as number[];

      const avgConfidence =
        confidences.length > 0
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : undefined;

      setPredictionSummary({
        highCount,
        mediumCount,
        lowCount,
        count,
        avgConfidence,
      });

      setStatusMessage({
        text: `Predicted profitability for ${count} jobs`,
        type: 'success',
      });
    } catch (error: any) {
      // extract detailed error from backend response
      const respData = error?.response?.data;
      let errorMsg = 'Prediction failed';

      if (respData) {
        const parts: string[] = [];
        const baseMsg = respData.error || respData.message;
        if (baseMsg) parts.push(baseMsg);
        if (respData.mlStatus) parts.push(`ML status ${respData.mlStatus}`);
        if (respData.mlBody) {
          if (typeof respData.mlBody === 'string') {
            parts.push(respData.mlBody);
          } else if (respData.mlBody?.error || respData.mlBody?.message) {
            parts.push(respData.mlBody.error || respData.mlBody.message);
          }
        }
        errorMsg = parts.length > 0 ? parts.join(' - ') : errorMsg;
      } else if (error?.message) {
        errorMsg = error.message;
      }

      setStatusMessage({
        text: errorMsg,
        type: 'error',
      });
      setPredictionSummary(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    setPagination((prev) => ({ ...prev, page: newPage }));
  };

  const handlePageSizeChange = (newPageSize: number) => {
    setPagination((prev) => ({ ...prev, page: 1, pageSize: newPageSize }));
  };

  // auto-load when page or pageSize changes
  useEffect(() => {
    if (pagination.pageSize > 0) {
      loadJobs(pagination.page, pagination.pageSize);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pagination.page, pagination.pageSize]);

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
        onFilterChange={setFilters}
        onSync={handleSync}
        onLoadJobs={() => loadJobs()}
        onPredict={handlePredict}
        isSyncing={isSyncing}
        isLoading={isLoading}
      />

      <main className="px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel - Jobs Table and Controls */}
          <div className="space-y-4">
            {statusMessage && (
              <StatusAlert message={statusMessage.text} type={statusMessage.type} />
            )}

            {isSyncing && <SyncProgressCard />}

            {predictionSummary && (
              <PredictionSummary summary={predictionSummary} />
            )}

            <JobsTable jobs={jobs} isLoading={isLoading} />

            <Pagination
              pagination={pagination}
              onPageChange={handlePageChange}
              onPageSizeChange={handlePageSizeChange}
              disabled={isLoading || isSyncing}
            />
          </div>

          {/* Right Panel - Predictions and Visualizations */}
          <div className="space-y-4">
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-slate-100 mb-2">
                Predictions & Visualizations
              </h2>
              <p className="text-sm text-slate-400">
                This panel will display prediction results and data visualizations.
              </p>
              <div className="mt-6 border-t border-slate-800 pt-6">
                <div className="text-center text-slate-500 py-12">
                  <p className="text-lg font-medium mb-2">Coming Soon</p>
                  <p className="text-sm">
                    Prediction models and interactive charts will appear here.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
