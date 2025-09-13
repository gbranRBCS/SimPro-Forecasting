import { toJobRowView, formatCurrency, formatDate, classBadgeProps } from "./jobs";
describe("toJobRowView", () => {
  it("maps basic fields and revenue", () => {
    const row = toJobRowView({
      ID: 123,
      RequestNo: "Pink 42",
      Customer: { CompanyName: "Eton College" },
      Site: { Name: "Farrer Theatre" },
      Stage: "On site",
      Total: { IncTax: 117.14 },
    });
    expect(row.id).toBe(123);
    expect(row.name).toBe("Pink 42");
    expect(row.customer).toBe("Eton College");
    expect(row.site).toBe("Farrer Theatre");
    expect(row.status).toBe("On site");
    expect(row.revenue).toBe(117.14);
  });
});
describe("formatters", () => {
  it("formats currency", () => {
    expect(formatCurrency(100)).toMatch(/£\s?100/);
  });
  it("formats date", () => {
    expect(formatDate("2025-09-12")).toMatch(/\d{1,2}\/\d{1,2}\/\d{4}/);
  });
  it("badge props", () => {
    expect(classBadgeProps("High").tone).toBe("success");
  });
});