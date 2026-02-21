export function toJobRowView(j: any) {
  const profitClass = j.profitability?.class ?? null;
  const profitScore = j.profitability?.score ?? null;
  const profitScoreType = j.profitability?.scoreType ?? null;

  return {
    id: j.id,
    name: j.descriptionText ?? `Job ${j.id ?? ""}`,
    customer: j.customerName ?? "—",
    site: j.siteName ?? "—",
    status: j.status_name ?? j.stage ?? "—",
    revenue: j.revenue ?? null,
    issued: j.dateIssued ?? null,
    due: j.dateDue ?? null,

    profitClass,
    profitScore,
    profitScoreType,
    profitEst: null,
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
    return isNaN(d.getTime()) ? "—" : d.toLocaleDateString('en-GB');
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

