"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type ChatMarkdownProps = {
  children: string;
};

export function ChatMarkdown({ children }: ChatMarkdownProps) {
  return (
    <div className="message-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node, ...props }) => {
            void node;
            return <a {...props} rel="noreferrer" target="_blank" />;
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
