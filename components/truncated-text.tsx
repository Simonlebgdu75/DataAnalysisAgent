"use client";

import { useState } from "react";

type TruncatedTextProps = {
  text: string;
  limit?: number;
  className?: string;
};

export function TruncatedText({
  text,
  limit = 150,
  className,
}: TruncatedTextProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (text.length <= limit) {
    return <span className={className}>{text}</span>;
  }

  const truncated = `${text.slice(0, limit).trimEnd()}…`;

  return (
    <span className={className}>
      {isExpanded ? text : truncated}{" "}
      <button
        className="inline-toggle"
        type="button"
        onClick={() => setIsExpanded((current) => !current)}
      >
        {isExpanded ? "See less" : "See more"}
      </button>
    </span>
  );
}

