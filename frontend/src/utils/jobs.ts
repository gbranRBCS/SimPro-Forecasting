export function toJobRowView(j: any) {
  return {
    id: j.ID,
    name: j.Name ?? j.RequestNo ?? `Job ${j.ID}`,
    customer: j.customerName ?? j.Customer?.CompanyName ?? "—",
    site: j.siteName ?? j.Site?.Name ?? "—",
    status: j.status?.Name ?? j.Stage ?? "—",
    revenue: j.revenue ?? j?.Total?.IncTax ?? null,
    issued: j.dateIssued ?? j.DateIssued ?? null,
    due: j.dateDue ?? j.DueDate ?? null,

    profitClass: j.profitability?.class ?? null,
    profitScore: j.profitability?.score ?? null,
  };
}

// pretty print
export function formatCurrency(v: number | null | undefined) {
  if (v == null) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "GBP" }).format(v);
}

export function formatDate(s: string | Date | null | undefined) {
  if (!s) return "—";
  const d = new Date(s as any);
  return isNaN(d as any) ? "—" : d.toLocaleDateString();
}

// colours/ icons
export function classBadgeProps(cls: string | null | undefined) {
  switch (cls) {
    case "High":
      return { tone: "success", label: "High" } as const;
    case "Medium":
      return { tone: "warning", label: "Medium" } as const;
    case "Low":
      return { tone: "destructive", label: "Low" } as const;
    default:
      return { tone: "neutral", label: "—" } as const;
  }
}

