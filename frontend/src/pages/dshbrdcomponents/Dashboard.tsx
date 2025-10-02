import { useState, useEffect } from 'react';
import { Briefcase } from 'lucide-react';
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

  const loadJobs = async (newPage?: number, newPageSize?: number) => {
    setIsLoading(true);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      const params: any = {
        sortField: 'revenue',
        order: filters.order,
        page: newPage ?? pagination.page,
        pageSize: newPageSize ?? pagination.pageSize,
      };

      if (filters.minRevenue) {
        params.minRevenue = Number(filters.minRevenue);
      }
      if (filters.maxRevenue) {
        params.maxRevenue = Number(filters.maxRevenue);
      }
      if (filters.limit) {
        params.limit = Number(filters.limit);
      }

      const data = await getJobs(params);
      setJobs(data?.jobs ?? []);
      setPagination((prev) => ({
        ...prev,
        page: newPage ?? prev.page,
        pageSize: newPageSize ?? prev.pageSize,
        totalPages: data?.totalPages ?? 1,
      }));
      setStatusMessage({
        text: `Loaded ${data?.jobs?.length ?? 0} jobs`,
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

  const handleSync = async () => {
    setIsSyncing(true);
    setStatusMessage(null);
    setPredictionSummary(null);

    try {
      const params: { from?: string; to?: string } = {};
      if (filters.fromDate) params.from = filters.fromDate;
      if (filters.toDate) params.to = filters.toDate;

      const data = await syncJobs(params);
      setStatusMessage({ text: data?.message || 'Synced', type: 'success' });
      // auto-refresh after sync
      await loadJobs(1);
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
      // pass jobs via params (backend uses query params, not body)
      const response = await predict({ jobs });
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
              simPRO business data forecasting
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

      <main className="px-6 py-6 space-y-4">
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
      </main>
    </div>
  );
}
