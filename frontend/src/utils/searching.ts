import type { Job } from '../features/jobs/api';

// Linear search through jobs, checking job description, customer name and site name. Not case sensitive.
export function searchJobs(jobs: Job[], searchTerm: string): Job[] {
    const term = searchTerm.trim().toLowerCase();
    if (!term) {
        return jobs;
    }
    else {
        const results: Job[] = [];
        for (let i = 0; i < jobs.length; i++) {
            let job = jobs[i];
            
            // Convert job fields to lower case
            const customer = (job.customerName || '').toLowerCase();
            const description = (job.Description || '').toLowerCase();
            const site = (job.siteName || '').toLowerCase();

            if (customer.includes(term) || description.includes(term) || site.includes(term)) {
                results.push(job);
            }
        }
        return results;
    }
}