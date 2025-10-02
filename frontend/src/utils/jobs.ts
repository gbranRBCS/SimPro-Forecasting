export function toJobRowView(j: any) {
  // profitability fallbacks
  let profitClass = j.profitability?.class ?? j.profitability_class ?? null;

  // if no nested class but we have a margin, infer a class
  const margin = typeof j.netMarginPct === "number" ? j.netMarginPct : null;
  if (!profitClass && margin != null) {
    if (margin > 0.64) profitClass = "High";
    else if (margin >= 0.44) profitClass = "Medium";
    else profitClass = "Low";
  }

  const profitScore =
    j.profitability?.score ??
    (typeof j.netMarginPct === "number" ? j.netMarginPct : null);

  const profitScoreType =
    j.profitability?.scoreType ??
    (typeof j.netMarginPct === "number" ? "margin" : null);

  return {
    id: j.ID ?? j.id,
    name: j.Name ?? j.RequestNo ?? `Job ${j.ID ?? j.id ?? ""}`,
    customer: j.customerName ?? j.Customer?.CompanyName ?? "—",
    site: j.siteName ?? j.Site?.Name ?? "—",
    status: j.statusName ?? j.status?.Name ?? j.Stage ?? "—",
    revenue: j.revenue ?? j?.Total?.IncTax ?? null,
    issued: j.dateIssued ?? j.DateIssued ?? null,
    due: j.dateDue ?? j.DueDate ?? null,

    profitClass,
    profitScore,
    profitScoreType,
    profitEst: j.profit_est ?? null,
  };
}

// pretty print
export function formatCurrency(v: number | null | undefined) {
  if (v == null) return "—";
  try {
    return new Intl.NumberFormat('en-AU', { style: "currency", currency: "GBP" }).format(v);
  } catch (e) {
    return "—";
  }
}

export function formatDate(s: string | Date | null | undefined) {
  if (!s) return "—";
  try {
    const d = new Date(s as any);
    return isNaN(d.getTime()) ? "—" : d.toLocaleDateString();
  } catch (e) {
    return "—";
  }
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
      return { tone: "default", label: "—" } as const;
  }
}

