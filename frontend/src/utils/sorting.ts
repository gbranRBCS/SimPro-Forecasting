import type { Job } from '../features/jobs/api';

export type SortDirection = 'asc' | 'desc';
export type SortKey = 
  | 'customer' 
  | 'site' 
  | 'status' 
  | 'issued' 
  | 'revenue' 
  | 'profitability';

export function sortJobs(jobs: Job[], key: SortKey | null, direction: SortDirection | null, predictions?: Record<string, {class?: string}>): Job[] {
    if (! (key && direction)) {return jobs;}

    // Retrieve sort values from sortKey, default numerical values to 0 if null.
    const getSortValue = (job: Job, sortKey: SortKey): string | number => {
        switch (sortKey) {
            case 'customer':
                return (job.customerName || '').toLowerCase();
            case 'site':
                return (job.siteName || '').toLowerCase();
            case 'status':
                return (job.status_name || '').toLowerCase();
            case 'issued':
                const dateString = job.dateIssued;
                return dateString ? new Date(dateString).getTime() : 0;
            case 'revenue':
                return job.revenue ?? (job as any).Total?.IncTax ?? 0;
            case 'profitability':
                const profitClass = getProfitClass(job);
                if (profitClass === "high") return 3;
                if (profitClass === "medium") return 2;
                if (profitClass === "low") return 1;
                return 0;
            default:
                return 0;
        }
    };

    function getProfitClass(job: Job): string | undefined {
        if (!predictions) return undefined;

        // get job ID
        const id = job.id ?? (job as any).ID;
        const idKey = id != null ? String(id) : '';

        // find prediction value from ID
        const prediction = predictions[idKey];

        // extract class from prediction
        const profitClass = prediction?.class?.toLowerCase();

        return profitClass
    }

    function mergeSort(arr: Job[]): Job[] {
        if (arr.length <= 1) return arr;
        
        const mid = Math.floor(arr.length / 2);
        const L = mergeSort(arr.slice(0, mid));
        const R = mergeSort(arr.slice(mid));

        return merge(L, R);
    }

    function merge(left: Job[], right: Job[]): Job[] {
        const result: Job[] = [];
        let i = 0;
        let j = 0;

        // Compare left and right arrays, mergen in order
        while (i < left.length && j < right.length) {
            if (getSortValue(left[i], key!) < getSortValue(right[j], key!)) {
                result.push(left[i]);
                i++
            } else {
                result.push(right[j]);
                j++
            }
        }
        // Add any leftover values in left or right arrays
        while (i < left.length) {
            result.push(left[i]);
            i++;
        }
        while (j < right.length) {
            result.push(right[j]);
            j++;
        }
        return result;
    }

    const sorted = mergeSort(jobs);

    if (direction === "desc") {
        return sorted.reverse();
    }
    
    return sorted;
}