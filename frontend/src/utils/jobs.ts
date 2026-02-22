export function toJobRowView(j: any) {
  const profitability = j.profitability;
  const profitClass = profitability?.class;
  const profitScore = profitability?.score;
  const profitScoreType = profitability?.scoreType;

  return {
    id: j.id,
    name: j.descriptionText,
    customer: j.customerName,
    site: j.siteName,
    status: j.status_name,
    revenue: j.revenue,
    issued: j.dateIssued,
    due: j.dateDue,

    profitClass,
    profitScore,
    profitScoreType,
    profitEst: null,
  };
}

// currency text is formatted for GBP
export function formatCurrency(value: number) {
  const formatter = new Intl.NumberFormat("en-AU", {
    style: "currency",
    currency: "GBP",
  });

  return formatter.format(value);
}

// Date values are shown using UK format dd/mm/yyyy
export function formatDate(value: string | Date) {
  const dateValue = new Date(value);
  return dateValue.toLocaleDateString("en-GB");
}

// profitability class decides badge tone and label.
export function classBadgeProps(cls: string) {
  switch (cls) {
    case "High":
      return { tone: "success", label: "High" } as const;
    case "Medium":
      return { tone: "warning", label: "Medium" } as const;
    case "Low":
      return { tone: "destructive", label: "Low" } as const;
    default:
      throw new Error("Unknown class");
  }
}

