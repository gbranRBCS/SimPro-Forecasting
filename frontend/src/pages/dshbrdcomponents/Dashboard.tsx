import { useState, useEffect } from 'react';
import { Briefcase } from '../../components/icons';
import {
  syncJobs,
  getJobs,
  predictProfitability,
  predictDuration,
} from '../../features/jobs/api';
import type { Job, Prediction, SyncParams } from '../../features/jobs/api';

import { Toolbar } from './Toolbar';
import { JobsTable } from './JobsTable';
import { PredictionPanel } from './PredictionPanel';
import { Pagination } from './Pagination';
import { StatusAlert, SyncProgressCard } from './StatusCards';

// types for component state
export type FilterState = {
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

type PredictionType = 'profitability' | 'duration' | 'none';

type StatusMessage = {
  text: string;
  type: 'success' | 'error' | 'info';
};

// helper to filter jobs on the client side
// this is used when we have just synced data or want to filter the current list without fetching
function filterJobsLocally(jobs: Job[], filters: FilterState, page: number, pageSize: number) {
  let result = [...jobs];

  // filter by revenue
  if (filters.minRevenue) {
    result = result.filter((j) => (j.revenue ?? 0) >= Number(filters.minRevenue));
  }
  if (filters.maxRevenue) {
    result = result.filter((j) => (j.revenue ?? 0) <= Number(filters.maxRevenue));
  }

  // sort by revenue
  result.sort((a, b) => {
    const revA = a.revenue ?? 0;
    const revB = b.revenue ?? 0;
    return filters.order === 'asc' ? revA - revB : revB - revA;
  });

  // paginate
  const total = result.length;
  const start = (page - 1) * pageSize;
  const pagedJobs = result.slice(start, start + pageSize);
  
  return { 
    jobs: pagedJobs, 
    total, 
    totalPages: Math.ceil(total / pageSize) 
  };
}

export function Dashboard() {
  // state for data and api calls
  const [jobs, setJobs] = useState<Job[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null);

  // state for filtering and pagination
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

  // state for predictions
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [predictionType, setPredictionType] = useState<PredictionType>('profitability');
  const [predictions, setPredictions] = useState<Record<string, Prediction>>({});
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // load jobs from backend or use preloaded data
  async function loadJobs(pageToLoad: number, preloadedJobs?: Job[]) {
    // clear old messages
    setStatusMessage(null);
    
    // if not using preloaded data, show loading spinner
    if (!preloadedJobs) setIsLoading(true);

    try {
      let jobsToShow: Job[] = [];
      let total = 0;
      let pages = 1;

      if (preloadedJobs) {
        // use local filtering if we already have the data
        const result = filterJobsLocally(preloadedJobs, filters, pageToLoad, pagination.pageSize);
        jobsToShow = result.jobs;
        total = result.total;
        pages = result.totalPages;
      } else {
        // fetch from api
        const params: any = {
          sortField: 'revenue',
          order: filters.order,
          page: pageToLoad,
          pageSize: pagination.pageSize,
        };

        // add optional filters
        if (filters.minRevenue) params.minRevenue = Number(filters.minRevenue);
        if (filters.maxRevenue) params.maxRevenue = Number(filters.maxRevenue);
        if (filters.limit) params.limit = Number(filters.limit);

        const response = await getJobs(params);
        jobsToShow = response.jobs;
        total = response.total;
        pages = response.totalPages || 1;
      }

      setJobs(jobsToShow);
      setPagination(prev => ({ ...prev, page: pageToLoad, totalPages: pages }));

      // show success message
      if (!preloadedJobs) {
        setStatusMessage({ 
          text: `Loaded ${jobsToShow.length} jobs (Total: ${total})`, 
          type: 'success' 
        });
      }

    } catch (err: any) {
      console.error(err);
      setStatusMessage({ 
        text: err.message || 'Failed to load jobs', 
        type: 'error' 
      });
    } finally {
      setIsLoading(false);
    }
  }

  // handle syncing with simpro
  async function handleSync() {
    setIsSyncing(true);
    setStatusMessage({ text: 'Syncing with SimPRO...', type: 'info' });

    try {
      // prepare sync params
      const params: SyncParams = { mode: 'update' };
      if (filters.fromDate) params.from = filters.fromDate;
      if (filters.toDate) params.to = filters.toDate;

      const response = await syncJobs(params);
      
      setStatusMessage({ text: response.message, type: 'success' });
      
      // refresh list with synced data
      if (response.jobs) {
        await loadJobs(1, response.jobs);
      } else {
        await loadJobs(1);
      }

    } catch (err: any) {
      console.error(err);
      setStatusMessage({ 
        text: err.response?.data?.error || err.message || 'Sync failed', 
        type: 'error' 
      });
    } finally {
      setIsSyncing(false);
    }
  }

  // run predictions based on selected type
  async function handlePredict() {
    if (selectedJobIds.length === 0) {
      setPredictionError('Please select at least one job to predict');
      return;
    }

    setPredictionLoading(true);
    setPredictionError(null);
    setStatusMessage(null);

    try {
      let result;
      if (predictionType === 'profitability') {
        result = await predictProfitability({ jobIds: selectedJobIds });
      } else {
        result = await predictDuration({ jobIds: selectedJobIds });
      }

      // update predictions map with new results
      const newPredictions: Record<string, Prediction> = {};
      result.predictions.forEach((p: Prediction) => {
        newPredictions[String(p.jobId)] = p;
      });

      setPredictions(prev => ({ ...prev, ...newPredictions }));
      setStatusMessage({ 
        text: `Successfully predicted ${result.predictions.length} jobs`, 
        type: 'success' 
      });

    } catch (err: any) {
      console.error(err);
      setPredictionError(err.message || 'Prediction failed');
    } finally {
      setPredictionLoading(false);
    }
  }

  // reload when page changes
  useEffect(() => {
    loadJobs(pagination.page);
  }, [pagination.page]); 

  // header component
  function Header() {
    return (
      <header className="bg-slate-900/50 border-b border-slate-800 px-6 py-6 transition-all">
        <div className="flex items-center gap-3">
          <Briefcase className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-2xl font-bold text-slate-100">Jobs Dashboard</h1>
            <p className="text-sm text-slate-400 mt-0.5">SimPRO Business Data Forecasting</p>
          </div>
        </div>
      </header>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-200">
      <Header />

      <Toolbar
        filters={filters}
        onFilterChange={newFilters => setFilters(prev => ({ ...prev, ...newFilters }))}
        onSync={handleSync}
        onLoadJobs={() => loadJobs(1)}
        isSyncing={isSyncing}
        isLoading={isLoading}
      />

      <main className="px-6 py-6 flex flex-col gap-6 max-w-[1600px] mx-auto">
        
        {/* top section: prediction controls and charts */}
        <section className="w-full transition-all duration-300">
          <PredictionPanel
            predictionType={predictionType}
            selectedJobs={jobs.filter(j => selectedJobIds.includes(String(j.id)))}
            predictions={predictions}
            loading={predictionLoading}
            error={predictionError}
            onPredict={handlePredict}
            onPredictionTypeChange={setPredictionType}
          />
        </section>

        {/* main content: tables and alerts */}
        <section className="space-y-4 w-full">
          {statusMessage && (
            <StatusAlert message={statusMessage.text} type={statusMessage.type} />
          )}

          {isSyncing && <SyncProgressCard />}

          <JobsTable
            jobs={jobs}
            isLoading={isLoading}
            selectedJobIds={selectedJobIds}
            onSelectionChange={setSelectedJobIds}
            predictions={predictions}
            predictionType={predictionType}
          />

          <Pagination
            pagination={pagination}
            onPageChange={page => setPagination(prev => ({ ...prev, page }))}
            onPageSizeChange={size => setPagination(prev => ({ ...prev, page: 1, pageSize: size }))}
            disabled={isLoading || isSyncing}
          />
        </section>
      </main>
    </div>
  );
}