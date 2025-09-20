export function toJobRowView(j: any) {
  // fallbacks for profitability info
  let profitClass = j.profitability?.class ?? j.profitability_class ?? null;
  let profitScore =
    j.profitability?.score ??
    (typeof j.netMarginPct === "number" ? j.netMarginPct : null);

  if (!profitClass && typeof j.netMarginPct === "number") {
    if (j.netMarginPct >= 0.10) profitClass = "High";
    else if (j.netMarginPct >= 0.03) profitClass = "Medium";
    else profitClass = "Low";
  }

  return {
    id: j.ID,
    name: j.Name ?? j.RequestNo ?? `Job ${j.ID}`,
    customer: j.customerName ?? j.Customer?.CompanyName ?? "—",
    site: j.siteName ?? j.Site?.Name ?? "—",
    status: j.status?.Name ?? j.Stage ?? "—",
    revenue: j.revenue ?? j?.Total?.IncTax ?? null,
    issued: j.dateIssued ?? j.DateIssued ?? null,
    due: j.dateDue ?? j.DueDate ?? null,

    profitClass,
    profitScore,
    profitEst: j.profit_est ?? null,
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

