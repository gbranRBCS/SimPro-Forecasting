import { forwardRef, type ReactNode } from "react";
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & {
  size?: number | string;
};

function createIcon(node: ReactNode, displayName: string) {
  const Icon = forwardRef<SVGSVGElement, IconProps>(
    ({ size = 24, strokeWidth = 2, ...props }, ref) => (
      <svg
        ref={ref}
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        {...props}
      >
        {node}
      </svg>
    ),
  );

  Icon.displayName = displayName;

  return Icon;
}

export const Briefcase = createIcon(
  <>
    <path d="M16 20V4a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    <rect width="20" height="14" x="2" y="6" rx="2" />
  </>,
  "BriefcaseIcon",
);

export const Filter = createIcon(
  <path d="M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z" />,
  "FilterIcon",
);

export const RefreshCw = createIcon(
  <>
    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
    <path d="M21 3v5h-5" />
    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
    <path d="M8 16H3v5" />
  </>,
  "RefreshCwIcon",
);

export const Sparkles = createIcon(
  <>
    <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z" />
    <path d="M20 2v4" />
    <path d="M22 4h-4" />
    <circle cx="4" cy="20" r="2" />
  </>,
  "SparklesIcon",
);

export const Loader2 = createIcon(
  <path d="M21 12a9 9 0 1 1-6.219-8.56" />,
  "Loader2Icon",
);

export const ChevronLeft = createIcon(
  <path d="m15 18-6-6 6-6" />,
  "ChevronLeftIcon",
);

export const ChevronRight = createIcon(
  <path d="m9 18 6-6-6-6" />,
  "ChevronRightIcon",
);

export const ChevronDown = createIcon(
  <path d="m6 9 6 6 6-6" />,
  "ChevronDownIcon",
);

export const AlertCircle = createIcon(
  <>
    <circle cx="12" cy="12" r="10" />
    <line x1="12" x2="12" y1="8" y2="12" />
    <line x1="12" x2="12.01" y1="16" y2="16" />
  </>,
  "AlertCircleIcon",
);

export const CheckCircle = createIcon(
  <>
    <path d="M21.801 10A10 10 0 1 1 17 3.335" />
    <path d="m9 11 3 3L22 4" />
  </>,
  "CheckCircleIcon",
);

export const TrendingUp = createIcon(
  <>
    <path d="M16 7h6v6" />
    <path d="m22 7-8.5 8.5-5-5L2 17" />
  </>,
  "TrendingUpIcon",
);
