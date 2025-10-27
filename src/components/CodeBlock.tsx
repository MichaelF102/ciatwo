import type { PropsWithChildren } from 'react';
import Highlight, { defaultProps } from 'prism-react-renderer';
import nightOwl from 'prism-react-renderer/themes/nightOwl';

function CodeBlock({ children, language = 'python' }: PropsWithChildren<{ language?: string }>) {
  const code = typeof children === 'string' ? children.trim() : '';
  return (
    <Highlight {...defaultProps} theme={nightOwl} code={code} language={language as never}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre className={`${className} overflow-auto rounded-2xl p-4`} style={style}>
          <code>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line, key: i })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token, key })} />
                ))}
              </div>
            ))}
          </code>
        </pre>
      )}
    </Highlight>
  );
}

export default CodeBlock;
